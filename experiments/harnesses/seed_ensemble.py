"""Does averaging three seeds' probabilities beat a single head?

The head is not capacity-limited and the features are frozen, so the remaining head-side lever is
variance: the same recipe at three seeds moves accuracy by a measured **0.6-1.9 pt** per task, and a
probability average is the cheapest way to take some of that back -- no new features, no new
parameters, and the backbone still runs exactly once (only the 0.26M head runs three times, which
costs 1-143 ms per whole test split).

**The baseline is the mean of the three single-head rows, not the best of them.** A single head is
one draw from that 0.6-1.9 pt spread; scoring the ensemble against the luckiest of its own three
draws would be flattery, and against seed 0 alone it would inherit whichever way seed 0 happened to
fall. So both columns come from the same three heads: `single` is their mean, `ensemble` is the
metric of their averaged probabilities, per task.

The fitted temperature is deliberately not re-fitted for the ensemble: the shipped pipeline fits one
scalar per (qtype, option-count band) on a single head's dev, and re-fitting it on the ensemble
would silently make that a different recipe as well as a different head count. So the comparison is
at T=1, which is also what the eval table headlines.

Run one chunk of tasks per process, as with ``arm_sweep.py``. Each invocation merges into
``experiments/results/seed_ensemble.json``, replacing only the tasks it recomputed.

Usage::

    python experiments/harnesses/seed_ensemble.py [task,task,...]
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_sweep import cached_tasks, quiet, train_head  # noqa: E402

from rwkv_system1.cache_layout import DEFAULT_BACKBONE, head_kwargs, load_features  # noqa: E402
from rwkv_system1.decision import DecisionHead  # noqa: E402
from rwkv_system1.features import FeatureSet  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.rlcd import (accuracy, entropy_confidence_ece, expected_calibration_error,  # noqa: E402
                               negative_log_likelihood)
from rwkv_system1.targets import get_soft_targets
from rwkv_system1.task_registry import REAL_TASKS
from rwkv_system1.train import head_logits

SEEDS = (0, 1, 2)
METRICS = ("accuracy", "ece", "nll", "ece_entropy")
DEST = Path(__file__).resolve().parents[1] / "results" / "seed_ensemble.json"


def score(probs: torch.Tensor, fs: FeatureSet) -> dict:
    """The pipeline's own metrics, computed from a probability matrix instead of a head."""
    labels = fs.labels
    out = {
        "accuracy": accuracy(probs, labels),
        "ece": expected_calibration_error(probs, labels),
        "ece_entropy": entropy_confidence_ece(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }
    if fs.true_probs is not None:
        out["posterior_l1"] = float((probs - fs.true_probs).abs().sum(-1).mean() / 2)
        out["posterior_kl"] = float(
            (fs.true_probs * (torch.log(fs.true_probs.clamp_min(1e-12))
                              - torch.log(probs.clamp_min(1e-12)))).sum(-1).mean()
        )
    return out


def probs_of(head: DecisionHead, te: FeatureSet) -> torch.Tensor:
    return head.probabilities(head_logits(head, te), 1.0)


def measure(key: str, base: Config) -> dict:
    """Three heads, one row per task: the mean of the singles and the average of the probabilities."""
    cfg = replace(base, seed=SEEDS[0])
    splits = {s: load_features(DEFAULT_BACKBONE, key, s, cfg) for s in ("train", "dev", "test")}
    te = splits["test"]
    soft = get_soft_targets(DEFAULT_BACKBONE, key, cfg, splits["train"])
    per_seed, probs = [], []
    for seed in SEEDS:
        head, _ = train_head(replace(base, seed=seed), splits["train"], splits["dev"], soft,
                             log=quiet)
        p = probs_of(head, te)
        probs.append(p)
        per_seed.append(score(p, te))
        del head
    single = {m: st.mean([r[m] for r in per_seed]) for m in per_seed[0]}
    single["accuracy_sd"] = st.stdev([r["accuracy"] for r in per_seed])
    ens = score(torch.stack(probs).mean(0), te)
    row = {"task": key, "n_test": len(te), "n_options": int(te.n_options),
           "single": single, "ensemble": ens,
           "seeds": [{m: r[m] for m in METRICS} for r in per_seed]}
    print(f"  {key:<20} acc {single['accuracy']:.4f}-> {ens['accuracy']:.4f}"
          f"  ece {single['ece']:.4f}-> {ens['ece']:.4f}"
          f"  nll {single['nll']:.4f}-> {ens['nll']:.4f}"
          f"  (seed sd {single['accuracy_sd']:.4f})", flush=True)
    return row


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    wanted = [t.strip() for t in (argv[0].split(",") if argv else []) if t.strip()]
    if not wanted:
        wanted = cached_tasks()
    # A `$PY -c` list coming through bash carries a trailing \r on its last field, and `cmd_extract`
    # only survives that because it strips; a harness that merely filters truthy values would drop
    # that task silently.  Name the unknown ones instead -- a dropped task is a missing row, not an
    # error, and a missing row reads exactly like a task that was never measured.
    unknown = [t for t in wanted if t not in REAL_TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown!r}; a stray \\r or typo in --tasks?")
    base = Config()
    rows = json.loads(DEST.read_text(encoding="utf-8")) if DEST.exists() else []
    rows = [r for r in rows if r["task"] not in set(wanted)]
    print(f"backbone {DEFAULT_BACKBONE}   seeds {SEEDS}   tasks {wanted}\n")
    for key in wanted:
        rows.append(measure(key, base))
    rows.sort(key=lambda r: r["task"])
    DEST.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"\n########## single head (mean of {len(SEEDS)}) vs 3-seed probability average ##########")
    print(f"{'task':<20}{'acc single':>11}{'acc ens':>9}{'d_acc':>8}   "
          f"{'ece single':>11}{'ece ens':>9}{'d_ece':>9}   {'seed sd':>8}")
    d = {m: [] for m in METRICS}
    for r in rows:
        for m in METRICS:
            d[m].append(r["ensemble"][m] - r["single"][m])
        print(f"{r['task']:<20}{r['single']['accuracy'] * 100:>10.2f}%"
              f"{r['ensemble']['accuracy'] * 100:>8.2f}%{d['accuracy'][-1] * 100:>+8.2f}   "
              f"{r['single']['ece']:>11.4f}{r['ensemble']['ece']:>9.4f}"
              f"{d['ece'][-1]:>+9.4f}   {r['single']['accuracy_sd']:>8.4f}")

    def describe(m, scale=100.0, suffix="pt"):
        vals = d[m]
        wins = sum(v > 0 for v in vals) if m == "accuracy" else sum(v < 0 for v in vals)
        sd = st.stdev(vals) if len(vals) > 1 else float("nan")
        print(f"  {m:<12} mean {st.mean(vals) * scale:+.2f}{suffix}  sd {sd * scale:.2f}"
              f"  se {sd / math.sqrt(len(vals)) * scale:.2f}  wins {wins}/{len(vals)}")

    print(f"\n{len(rows)} tasks:")
    for m in ("accuracy", "ece", "ece_entropy"):
        describe(m)
    describe("nll", scale=1.0, suffix="nats")
    print(f"\nwrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
