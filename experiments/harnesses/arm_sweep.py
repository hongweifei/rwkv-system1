"""Which recipe wins across the whole suite: the shipped one, a narrower head, or no RLCD stage?

Two earlier measurements each pointed somewhere but were too thin to change a default on:

* ``head_width_sweep.py`` swept 64/128/256/512 on two tasks.  The likelihood preferred the
  narrowest head on both and 512 was no better than 256, so the head is not capacity-limited --
  but the accuracy gaps were 2-3 test rows, which is not a result.
* the RLCD ablation (``rlcd_variants.py``) found the stage's accuracy gain is partly just thirty
  more epochs, that an equal-epochs control is *bit-identical* to plain cross-entropy on a
  hard-label task, and that on that task the stage costs calibration.  Three tasks again.

So this runs the three candidate recipes over every task whose features are cached, three seeds
each, and reports them side by side:

    shipped    ``Config()`` as it stands -- 256 wide, 60 warm-up + 30 RLCD epochs
    w64        ``head_d_model = 64``
    no-rlcd    ``rlcd_epochs = 0`` -- the same 60 warm-up epochs, no second stage

Same ``train`` and ``evaluate`` the pipeline calls, same cached features, same gold targets via
``get_soft_targets``, so a cell is the pipeline's own number rather than a parallel
implementation of it.  The shipped arm is retrained here rather than read back from ``runs/``, and
its accuracy should reproduce the chain's eval column exactly -- that agreement is the check that
this harness is measuring the pipeline and not itself.

Per-task differences of a point or two are still test-set noise; the statistic worth reading is the
mean of the paired per-task deltas, which is why the last table prints it with its standard error
over tasks.  Features are cached, so nothing is extracted.

Run one chunk of tasks per process, as with ``rlcd_variants.py``: a long single-process run of this
shape took a SIGSEGV once the box ran out of commit charge.  Each invocation merges into
``experiments/results/arm_sweep.json``, replacing only the rows it recomputes.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from rwkv_system1.cache_layout import (CACHE, DEFAULT_BACKBONE, ROOT, head_kwargs,
                                       load_features)
from rwkv_system1.decision import DecisionHead
from rwkv_system1.pipeline.config import Config
from rwkv_system1.targets import get_soft_targets
from rwkv_system1.task_registry import REAL_TASKS
from rwkv_system1.train import evaluate, train


SEEDS = (0, 1, 2)

ARMS: list[tuple[str, dict]] = [
    ("shipped", {}),
    ("w64", {"head_d_model": 64}),
    ("no-rlcd", {"rlcd_epochs": 0}),
]

METRICS = ("accuracy", "ece", "nll")


def quiet(*_a, **_k) -> None:
    pass


def cached_tasks() -> list[str]:
    """Tasks with all three splits on disk, in registry order."""
    have: dict[str, set[str]] = {}
    for p in CACHE.glob("*.pt"):
        n = p.stem
        for s in ("train", "dev", "test"):
            if f"_{s}_" in n:
                have.setdefault(n.split(f"_{s}_")[0], set()).add(s)
    return [k for k in REAL_TASKS if have.get(f"{DEFAULT_BACKBONE}_{k}") == {"train", "dev", "test"}]


def train_head(cfg: Config, tr, dv, soft, log=quiet):
    """One head trained exactly the way `cmd_train` trains it.

    Same seed for the weight init and the batch order, same recipe read off ``cfg``.  Extracted
    because the seed-ensemble harness needs the trained head (to average its probabilities), and a
    second copy of this call is how a harness ends up measuring a recipe that is not the shipped
    one.
    """
    torch.manual_seed(cfg.seed)
    head = DecisionHead(**head_kwargs(cfg, tr))
    history = train(
        head, tr, dv,
        soft_targets=soft,
        warmup_soft_targets=cfg.warmup_soft_targets,
        warmup_epochs=cfg.warmup_epochs,
        rlcd_epochs=cfg.rlcd_epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        group_size=cfg.group_size,
        pg_weight=cfg.pg_weight,
        w_sph=cfg.w_sph,
        w_rps=cfg.w_rps,
        exploration=cfg.exploration,
        explore_sigma=cfg.explore_sigma,
        permute_options=cfg.permute_options,
        seed=cfg.seed,
        log=log,
    )
    return head, history


def run_one(key: str, arm: str, overrides: dict, seed: int, base: Config,
            splits: dict, soft: torch.Tensor | None) -> dict:
    cfg = replace(base, seed=seed, **overrides)
    tr, dv, te = splits["train"], splits["dev"], splits["test"]
    t0 = time.perf_counter()
    head, history = train_head(cfg, tr, dv, soft)
    elapsed = time.perf_counter() - t0
    raw = evaluate(head, te, temperature=1.0)
    temp = history.temperature_for(int(te.qtype), int(te.n_options))
    fitted = evaluate(head, te, temperature=temp)
    row = {
        "task": key, "arm": arm, "seed": seed, "seconds": round(elapsed, 1),
        "n_test": len(te), "fitted_temperature": temp,
        "t1": {m: raw[m] for m in METRICS},
        "fit": {m: fitted[m] for m in METRICS},
        "params": sum(p.numel() for p in head.parameters()),
    }
    print(f"  {key:<18}{arm:<9}seed {seed}  acc {raw['accuracy']:.4f}  ece {raw['ece']:.4f}"
          f"  nll {raw['nll']:.4f}  T {temp:.2f} -> ece {fitted['ece']:.4f}  [{elapsed:.0f}s]",
          flush=True)
    return row


def mean_sd(values: list[float]) -> tuple[float, float]:
    v = [x for x in values if x is not None and not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    return (m, float("nan")) if len(v) < 2 else (m, st.stdev(v))


def selected(argv: list[str]) -> list[str]:
    return [t for t in (argv[0].split(",") if argv else []) if t]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tasks = selected(argv) or cached_tasks()
    if not tasks:
        raise SystemExit("no task has all three splits cached")
    dest = ROOT / "experiments" / "results" / "arm_sweep.json"
    previous: list[dict] = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else []
    fresh = {(t, a) for t in tasks for a, _ in ARMS}
    rows = [r for r in previous if (r["task"], r["arm"]) not in fresh]
    if len(rows) != len(previous):
        print(f"replacing {len(previous) - len(rows)} rows of {dest.name}, "
              f"keeping {len(rows)}")

    base = Config()
    print(f"backbone {DEFAULT_BACKBONE}   {len(tasks)} tasks   seeds {SEEDS}   "
          f"arms {[a for a, _ in ARMS]}")
    print(f"recipe: warmup {base.warmup_epochs}, RLCD {base.rlcd_epochs}, "
          f"targets {base.targets!r}, width {base.head_d_model}\n")

    for key in tasks:
        splits = {s: load_features(DEFAULT_BACKBONE, key, s, base)
                  for s in ("train", "dev", "test")}
        soft = get_soft_targets(DEFAULT_BACKBONE, key, base, splits["train"])
        tr, te = splits["train"], splits["test"]
        print(f"\n=== {key} ===  {len(tr)} train / {len(splits['dev'])} dev / {len(te)} test, "
              f"{te.n_options} options, {te.qtype_name}", flush=True)
        for arm, overrides in ARMS:
            for seed in SEEDS:
                rows.append(run_one(key, arm, overrides, seed, base, splits, soft))
        del splits, soft

    report(rows, tasks, [a for a, _ in ARMS])
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


def report(rows: list[dict], tasks: list[str], arms: list[str],
           reference: str = "shipped") -> None:
    """Per-task means, then the paired deltas, then those deltas averaged over tasks.

    ``arms`` is passed in rather than read from this module's ``ARMS`` so that a harness with a
    different arm model -- one that varies the data as well as the recipe, say -- can reuse the
    whole table.  ``reference`` is the arm the deltas are taken against.

    The per-task table is what a chunked run can print early, but a point or two on one task is
    test-set noise -- the cross-task mean of the paired deltas is the only number here with enough
    rows behind it to act on, which is why it gets its own table and a standard error.
    """
    done = [t for t in tasks
            if all(any(r["task"] == t and r["arm"] == a for r in rows) for a in arms)]
    if not done:
        print("\nno task has all arms complete; nothing to compare")
        return
    # Every column is formatted to a fixed width and padded afterwards, so the tables line up
    # whatever the numbers turn out to be.  The widths are derived rather than literals: an arm
    # named 'teacher-s1s2' is longer than 'w64', and a literal 11 let the header collide with the
    # cells.
    w = 18
    aw = max(11, max(len(a) for a in arms) + 1)
    pw = max(21, 2 * aw)
    compared = [a for a in arms if a != reference]

    def mean_of(key: str, arm: str, split: str, metric: str) -> float:
        return mean_sd([r[split][metric] for r in rows
                        if r["task"] == key and r["arm"] == arm])[0]

    for split, label in (("t1", "T=1"), ("fit", "after the dev-fitted temperature")):
        for metric, heading, fmt in (("accuracy", "accuracy", "{:.2f}%"),
                                     ("ece", "top-label ECE (lower better)", "{:.4f}")):
            print(f"\n########## {heading}, {label}, mean over {len(SEEDS)} seeds ##########")
            print(f"{'task':<{w}}" + "".join(f"{a:>{aw}}" for a in arms))
            for key in done:
                cells = []
                for arm in arms:
                    m = mean_of(key, arm, split, metric)
                    cells.append(fmt.format(100 * m if metric == "accuracy" else m))
                print(f"{key:<{w}}" + "".join(f"{c:>{aw}}" for c in cells))

    print("\n########## paired vs shipped, per task (T=1) ##########")
    print("(positive accuracy delta = better; negative ECE delta = better)")
    print(f"{'task':<{w}}" + "".join(f"{a:>{pw}}" for a in compared))
    paired: dict[str, dict[str, list[float]]] = {}
    for key in done:
        deltas = {}
        for arm in compared:
            per_seed = {m: [] for m in ("accuracy", "ece")}
            for seed in SEEDS:
                a = next(r for r in rows
                         if r["task"] == key and r["arm"] == arm and r["seed"] == seed)
                b = next(r for r in rows
                         if r["task"] == key and r["arm"] == reference and r["seed"] == seed)
                for m in per_seed:
                    per_seed[m].append(a["t1"][m] - b["t1"][m])
            deltas[arm] = {m: sum(v) / len(v) for m, v in per_seed.items()}
            for m, v in deltas[arm].items():
                paired.setdefault(arm, {}).setdefault(m, []).append(v)
        print(f"{key:<{w}}" + "".join(
            f"{100 * deltas[a]['accuracy']:+.2f}pt {deltas[a]['ece']:+.4f}".rjust(pw)
            for a in compared))

    print(f"\n########## those deltas averaged over the tasks (n={len(done)}) ##########")
    print(f"{'arm':<{w}}{'d_acc':>18}{'d_ece':>18}{'acc wins':>11}{'ece wins':>11}")
    for arm in compared:
        accs, eces = paired[arm]["accuracy"], paired[arm]["ece"]
        ma, sa = mean_sd(accs)
        me, se = mean_sd(eces)
        sea = sa / math.sqrt(len(accs))
        see = se / math.sqrt(len(eces))
        print(f"{arm:<{w}}{100 * ma:>+13.2f}±{100 * sea:.2f}pt{me:>+13.4f}±{see:.4f}"
              f"{sum(1 for x in accs if x > 0):>7}/{len(accs)}"
              f"{sum(1 for x in eces if x < 0):>7}/{len(eces)}")


if __name__ == "__main__":
    raise SystemExit(main())
