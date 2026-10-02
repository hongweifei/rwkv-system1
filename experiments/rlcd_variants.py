"""Which RLCD variant is better: the shipped one, the one the write-up describes, or neither?

Three questions in one run, same features, same head, same warm-up, only the stage-2 recipe
differs:

* ``ce``      -- no RLCD stage at all (``rlcd_epochs=0``);
* ``proper``  -- the stage runs but the policy-gradient term is switched off
                 (``pg_weight=0``), leaving the strictly proper score as the whole objective;
* ``shipped`` -- proper score + REINFORCE on the verifier ``1[a == y]``, sampling the reported
                 distribution;
* ``gauss*``  -- as ``shipped``, but the action is drawn as ``argmax(logits + sigma*N(0,1))``.

``proper`` is not an approximation of the write-up's variant, it *is* it.  The write-up says the
policy gradient's reward is the strictly proper score.  That reward is a function of the reported
distribution and the label, so it takes the same value for every draw ``a``: ``R(a) = R``, and the
PG term's expected gradient ``sum_a grad q(a) R(a)`` collapses to ``R grad sum_a q(a) = R grad 1 =
0``.  Feeding it the proper score therefore contributes zero signal and only variance; the verifier
``1[a == y]`` is the only reward here that varies with the draw.  So ``proper`` should land on the
same numbers as ``pg_weight=0`` would by construction -- which is what makes the arm worth running
rather than arguing about.

``gauss`` is the other half of the write-up.  Sampling the reported distribution is exactly
``argmax(logits + Gumbel)``, so ``gauss`` changes the noise law and not the objective: both are
unbiased REINFORCE for the same expected reward, differing in which actions get tried and in the
variance of the estimate.  ``sigma=1.283`` is the variance-matched choice -- a Gumbel(0,1) has
standard deviation ``pi/sqrt(6) = 1.2825`` -- and is run so the Gaussian arm cannot lose merely by
exploring less.

Reported at T=1 (the headline the pipeline prints) and after the dev-fitted temperature, because
that fit is where most of the published calibration comes from.  Every arm is run under three seeds
on a shared feature cache, so the comparison is paired and the seed noise cancels in the deltas.
n=3 resolves large effects only; the per-seed rows are dumped so a smaller one is not read as real.

Run it one task, or one arm group, per process::

    python experiments/rlcd_variants.py laya_urgency
    python experiments/rlcd_variants.py banking77 proper,shipped

Each invocation merges into ``experiments/results/rlcd_variants.json``, replacing only the rows it
recomputes.  That is not tidiness -- a single process doing all 54 runs took a SIGSEGV about
twenty-five minutes in, the same native-level failure the extraction pipeline works around by
chunking into five-task processes, and a crash should cost one chunk rather than the run.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from rwkv_system1.cache_layout import head_kwargs, load_features
from rwkv_system1.decision import DecisionHead
from rwkv_system1.pipeline.config import Config
from rwkv_system1.targets import get_soft_targets
from rwkv_system1.train import evaluate, train


BACKBONE = "rwkv7-g1d-0.1b"

#: ``laya_urgency`` is ordinal and ships gold posteriors, so the proper score has real work to do
#: there.  ``banking77`` (77 labels) and ``ag_news`` (4) ship hard labels only, where the log-score
#: term *is* cross-entropy and whatever the stage adds has to come from the spherical term and the
#: PG term -- and ``ag_news``' 1000-row test split keeps the paired deltas quieter than the 400- and
#: 500-row ones do.
TASKS = ["laya_urgency", "banking77", "ag_news"]
SEEDS = [0, 1, 2]

ARMS: list[tuple[str, dict]] = [
    ("ce", {"rlcd_epochs": 0}),
    # Equal-epochs control.  Every RLCD arm gets 60 warm-up epochs *plus* 30 stage-2 epochs, so
    # "RLCD beats cross-entropy" and "thirty more epochs beat cross-entropy" predict the same
    # number.  This arm runs cross-entropy for all 90 and separates them.
    ("ce90", {"warmup_epochs": 90, "rlcd_epochs": 0}),
    ("proper", {"pg_weight": 0.0}),
    ("shipped", {}),
    ("gauss1.0", {"exploration": "gaussian", "explore_sigma": 1.0}),
    ("gauss1.28", {"exploration": "gaussian", "explore_sigma": 1.2825}),
]

METRICS = ["accuracy", "ece", "ece_entropy", "nll", "posterior_l1"]


def quiet(*_a, **_k) -> None:
    pass


def run_one(key: str, arm: str, overrides: dict, seed: int, base: Config,
            splits: dict, soft: torch.Tensor | None) -> dict:
    cfg = replace(base, seed=seed, **overrides)
    tr, dv, te = splits["train"], splits["dev"], splits["test"]
    torch.manual_seed(cfg.seed)
    head = DecisionHead(**head_kwargs(cfg, tr))
    t0 = time.perf_counter()
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
        log=quiet,
    )
    elapsed = time.perf_counter() - t0

    raw = evaluate(head, te, temperature=1.0)
    temp = history.temperature_for(int(te.qtype), int(te.n_options))
    fit = evaluate(head, te, temperature=temp)
    row = {
        "task": key, "arm": arm, "seed": seed, "seconds": round(elapsed, 1),
        "fitted_temperature": temp,
        "t1": {m: raw.get(m) for m in METRICS},
        "fit": {m: fit.get(m) for m in METRICS},
        "dev_fitted_t1": evaluate(head, dv, temperature=1.0)["accuracy"],
    }
    print(
        f"  {key:<14}{arm:<11}seed {seed}  acc {raw['accuracy']:.4f}  ece {raw['ece']:.4f}"
        f"  nll {raw['nll']:.4f}  postL1 {raw.get('posterior_l1', float('nan')):.4f}"
        f"  T {temp:.2f} -> acc {fit['accuracy']:.4f} ece {fit['ece']:.4f}"
        f"  [{elapsed:.0f}s]",
        flush=True,
    )
    return row


def mean_sd(values: list[float]) -> tuple[float, float]:
    v = [x for x in values if x is not None and not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var)


def selected(argv: list[str]) -> tuple[list[str], list[str]]:
    """``[tasks] [arms]``, each comma-separated and each optional; empty means all of them."""
    tasks = [t for t in (argv[0].split(",") if argv else TASKS) if t]
    arms = [a for a in (argv[1].split(",") if len(argv) > 1 else []) if a]
    return tasks, arms


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tasks, only_arms = selected(argv)
    unknown = set(only_arms) - {a for a, _ in ARMS}
    if unknown:
        raise SystemExit(f"unknown arm(s) {sorted(unknown)}; choose from {[a for a, _ in ARMS]}")
    arms = [a for a in ARMS if not only_arms or a[0] in only_arms]

    base = Config()
    dest = Path(__file__).resolve().parent / "results" / "rlcd_variants.json"
    previous: list[dict] = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else []
    fresh = {(t, a) for t in tasks for a, _ in arms}
    rows = [r for r in previous if (r["task"], r["arm"]) not in fresh]
    if len(rows) != len(previous):
        print(f"replacing {len(previous) - len(rows)} rows of {dest.name}, "
              f"keeping {len(rows)} for other tasks or arms")

    print(f"backbone {BACKBONE}   tasks {tasks}   seeds {SEEDS}   arms {[a for a, _ in arms]}")
    print(f"recipe: warmup {base.warmup_epochs} epochs -> RLCD {base.rlcd_epochs} epochs, "
          f"targets {base.targets!r}, group {base.group_size}\n")

    for key in tasks:
        splits = {s: load_features(BACKBONE, key, s, base) for s in ("train", "dev", "test")}
        soft = get_soft_targets(BACKBONE, key, base, splits["train"])
        tr, te = splits["train"], splits["test"]
        print(f"\n=== {key} ===  {len(tr)} train / {len(splits['dev'])} dev / {len(te)} test, "
              f"{te.n_options} options, qtype {te.qtype_name}, "
              f"soft targets {'gold' if soft is not None else 'none (hard labels)'}", flush=True)
        if te.true_probs is not None:
            print(f"    Bayes accuracy ceiling {float(te.true_probs.max(-1).values.mean()):.4f}")
        else:
            print("    no analytic posterior shipped: a majority-class guess scores "
                  f"{float((te.labels == te.labels.bincount().argmax()).float().mean()):.4f}")
        for arm, overrides in arms:
            for seed in SEEDS:
                rows.append(run_one(key, arm, overrides, seed, base, splits, soft))
        # Released before the next task's splits are read: holding two 77-option caches at once
        # is the shape of allocation churn that ended the unchunked run.
        del splits, soft

    report(rows)
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


def report(rows: list[dict]) -> None:
    tasks = [t for t in TASKS if any(r["task"] == t for r in rows)]
    tasks += sorted({r["task"] for r in rows} - set(tasks))
    arms = [a for a, _ in ARMS if any(r["arm"] == a for r in rows)]

    for split in ("t1", "fit"):
        label = "T=1 (no temperature fitted)" if split == "t1" else "after the dev-fitted temperature"
        print(f"\n########## {label} ##########")
        print(f"{'task':<14}{'arm':<11}{'acc':>17}{'ece':>17}{'postL1':>17}")
        for key in tasks:
            for arm in arms:
                sel = [r for r in rows if r["task"] == key and r["arm"] == arm]
                if not sel:
                    continue          # a chunked run leaves arms missing from some tasks
                cells = []
                for m in ("accuracy", "ece", "posterior_l1"):
                    m_, sd_ = mean_sd([r[split].get(m) for r in sel])
                    cells.append(f"{m_:.4f}±{sd_:.4f}" if not math.isnan(m_) else "n/a")
                print(f"{key:<14}{arm:<11}{cells[0]:>17}{cells[1]:>17}{cells[2]:>17}")
            print()

    if "shipped" not in arms:
        print("(no `shipped` arm in this run, so no paired deltas)")
        return
    print("########## paired vs shipped, per-seed deltas ##########")
    print("(negative accuracy delta = worse; negative ece delta = better calibration)")
    print(f"{'task':<14}{'arm':<11}{'dacc':>18}{'dece':>18}{'dpostL1':>18}")
    for key in tasks:
        for arm in arms:
            if arm == "shipped":
                continue
            if len([r for r in rows if r["task"] == key and r["arm"] in (arm, "shipped")]) \
                    != 2 * len(SEEDS):
                continue          # only compare a task/arm pair that this run completed
            ds = []
            for seed in SEEDS:
                a = next(r for r in rows if r["task"] == key and r["arm"] == arm and r["seed"] == seed)
                b = next(r for r in rows
                         if r["task"] == key and r["arm"] == "shipped" and r["seed"] == seed)
                ds.append({m: (a["t1"].get(m) if a["t1"].get(m) is not None else float("nan"))
                           - (b["t1"].get(m) if b["t1"].get(m) is not None else float("nan"))
                           for m in ("accuracy", "ece", "posterior_l1")})
            cells = []
            for m in ("accuracy", "ece", "posterior_l1"):
                m_, sd_ = mean_sd([d[m] for d in ds])
                se = sd_ / math.sqrt(len(ds)) if not math.isnan(sd_) else float("nan")
                cells.append(f"{m_:+.4f}±{se:.4f}" if not math.isnan(m_) else "n/a")
            print(f"{key:<14}{arm:<11}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
