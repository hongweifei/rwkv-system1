"""How many training rows does the head actually need?  Measure the curve, don't guess the cap.

Every task's ``n_train`` is a cap far below its pool -- MASSIVE 400 of 11514, amazon 1500 of 200000,
ag_news 4000 of 120000, xnli 400 of a file holding 392702 rows per language -- so "train on all of
it" is a natural next question.  It is also a ~490-hour extraction job at the measured per-example
cost (0.2 s for a 4-option question, 1.1 s for MASSIVE's 60-option block, 2 s for banking77's 77
labels) plus ~40 GB of feature caches, which this box cannot pay.

So measure whether the cap binds before paying for any of it.  The head is a 53,889-parameter
readout (width 64) on frozen 768-dimensional features, and a readout that small may well saturate
hundreds of rows in.  This trains on fractions of the *cached* train split, three seeds each, and
prints test accuracy against rows seen:

    flat from 25% to 100%    -> the cap is not the constraint, and more extraction buys nothing
    still climbing at 100%   -> more rows are worth extraction hours, and the knee says how many

Free: nothing is extracted and no backbone runs, only heads on subsets of features already on disk.
Three seeds, because a curve whose points differ by less than the task's seed sd is not a curve.

Dev stays whole in every arm: it picks the epoch and fits the temperature, so shrinking it would
change two things at once.  The ``100%`` arm is the shipped recipe on the cached split, so it should
reproduce the chain's number for that task -- the check that this measures the pipeline.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_sweep import mean_sd, run_one  # noqa: E402

from rwkv_system1.cache_layout import CACHE, DEFAULT_BACKBONE, ROOT, load_features  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.targets import get_soft_targets  # noqa: E402


SEEDS = (0, 1, 2)
#: Fractions of the cached train split.  Geometric rather than even, because a knee low on the
#: curve is the interesting case and even spacing would miss it.
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
#: Where the cap sits furthest below the pool, i.e. where more rows could plausibly pay.  Pass a
#: comma-separated list to run others; the default is deliberately not "every cached task".
TASKS = ["massive_en", "ag_news", "banking77", "amazon_reviews_en", "xnli_en", "sst5"]
#: Seed for the row subset.  Fixed, so the 5%-row point of two different days is the same rows.
SUBSET_SEED = 12345


def take(fs, fraction: float):
    """``(subset, mask)`` -- the mask so a soft target is subset by the same rows, one source of
    truth rather than two functions that have to keep their randperm in lockstep."""
    n = max(1, int(round(len(fs) * fraction)))
    if n >= len(fs):
        return fs, None
    gen = torch.Generator().manual_seed(SUBSET_SEED)
    mask = torch.zeros(len(fs), dtype=torch.bool)
    mask[torch.randperm(len(fs), generator=gen)[:n]] = True
    return fs.subset(mask), mask


def cached_tasks() -> list[str]:
    from rwkv_system1.task_registry import REAL_TASKS

    have: dict[str, set[str]] = {}
    for p in CACHE.glob("*.pt"):
        n = p.stem
        for s in ("train", "dev", "test"):
            if f"_{s}_" in n:
                have.setdefault(n.split(f"_{s}_")[0], set()).add(s)
    return [k for k in REAL_TASKS if have.get(f"{DEFAULT_BACKBONE}_{k}") == {"train", "dev", "test"}]


def main(argv: list[str]) -> int:
    requested = [t for t in (argv[0].split(",") if argv else []) if t]
    tasks = requested or [t for t in TASKS if t in cached_tasks()]
    if not tasks:
        raise SystemExit(f"none of {requested or TASKS} has all three splits cached")
    # A second argument narrows the fractions.  Once the cap has been raised the pipeline's own eval
    # already gives the 100% point, so "is the curve still climbing at the cap" needs only the
    # smaller fractions -- and on a wide-option task the full four-point sweep is hours of head
    # training, not seconds.
    fractions = (tuple(float(x) for x in argv[1].split(",")) if len(argv) > 1 else FRACTIONS)
    base = Config()
    dest = ROOT / "experiments" / "results" / "data_scaling.json"
    print(f"backbone {DEFAULT_BACKBONE}   seeds {SEEDS}   "
          f"fractions {[f'{100 * f:g}%' for f in fractions]}\n")

    rows: list[dict] = []
    for key in tasks:
        full = {s: load_features(DEFAULT_BACKBONE, key, s, base)
                for s in ("train", "dev", "test")}
        soft_full = get_soft_targets(DEFAULT_BACKBONE, key, base, full["train"])
        print(f"\n=== {key} ===  {len(full['train'])} cached train rows, "
              f"{len(full['test'])} test", flush=True)
        for fraction in fractions:
            arm = f"{100 * fraction:g}%"
            tr, mask = take(full["train"], fraction)
            soft = soft_full if (soft_full is None or mask is None) else soft_full[mask]
            splits = {"train": tr, "dev": full["dev"], "test": full["test"]}
            for seed in SEEDS:
                row = run_one(key, arm, {}, seed, base, splits, soft)
                row["n_train"] = len(tr)
                rows.append(row)
        del full, soft_full

    report(rows, tasks)
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


def report(rows: list[dict], tasks: list[str]) -> None:
    """Test accuracy against rows seen, per task, with the paired delta against the full split."""
    for key in tasks:
        sel = [r for r in rows if r["task"] == key]
        if not sel:
            continue
        # Ordered by the fraction each arm names, so a narrowed sweep still prints in order.
        arms = sorted({r["arm"] for r in sel}, key=lambda a: float(a.rstrip("%")))
        print(f"\n########## {key}: accuracy vs training rows (T=1) ##########")
        print(f"{'rows':>7}{'acc':>19}{'d_acc vs 100%':>17}{'ece':>10}{'nll':>10}")
        whole = {r["seed"]: r for r in sel if r["arm"] == "100%"}
        for arm in arms:
            hits = [r for r in sel if r["arm"] == arm]
            if not hits:
                continue
            acc, sd = mean_sd([r["t1"]["accuracy"] for r in hits])
            delta = "--"
            if arm != "100%" and whole:
                ds = [r["t1"]["accuracy"] - whole[r["seed"]]["t1"]["accuracy"]
                      for r in hits if r["seed"] in whole]
                ma, sa = mean_sd(ds)
                se = sa / math.sqrt(len(ds)) if not math.isnan(sa) else float("nan")
                delta = f"{100 * ma:+.2f}±{100 * se:.2f}pt"
            ece, _ = mean_sd([r["t1"]["ece"] for r in hits])
            nll, _ = mean_sd([r["t1"]["nll"] for r in hits])
            print(f"{hits[0]['n_train']:>7}{100 * acc:>13.2f}%±{100 * sd:>5.2f}pt"
                  f"{delta:>17}{ece:>10.4f}{nll:>10.4f}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
