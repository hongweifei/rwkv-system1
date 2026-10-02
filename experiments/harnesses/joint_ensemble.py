"""Does the 3-seed ensemble still pay once the head is already a family joint head?

Both levers were measured against the same shipped baseline, never against each other:

    per-task head -> 3-seed probability ensemble   +0.80pt acc, -0.027 nats (34/34)   seed_ensemble.py
    per-task head -> per-family joint head          +9.67pt acc, -0.38 nats (10/10)   joint_family.py

The two are mechanically independent -- one adds training rows, the other averages seeds -- so
stacking them is plausible.  It is also exactly the kind of "both measured positive, so both help"
claim this project has been wrong about before, and the ensemble's gain is small enough that the
joint arm's own seed spread could swallow it.

Arms, all on MASSIVE's ten languages off the shipped (pinned-pooled) caches:

    per-task         one head per language, mean of its 3 seeds      (baseline; re-derived here,
                                                                     must match joint_family.py)
    per-task ens     3 per-task heads averaged                         (must match seed_ensemble.py)
    joint            one head over all ten languages, mean of seeds
    joint ens        3 joint heads averaged

The first two rows are controls, not results: if this harness disagrees with either finished sweep
on its own arm, the harness is broken and the ensemble column means nothing.

No temperature is re-fitted -- every metric is read at T=1, which is the shipped headline and the
only setting the ensemble can be scored at without inventing a per-average calibration.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_sweep import train_head  # noqa: E402

from rwkv_system1.cache_layout import (DEFAULT_BACKBONE, data_version, feature_inputs,  # noqa: E402
                                       load_features)
from rwkv_system1.decision import DecisionHead  # noqa: E402
from rwkv_system1.features import concat_joint  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.rlcd import (accuracy, entropy_confidence_ece,  # noqa: E402
                               expected_calibration_error, negative_log_likelihood)
from rwkv_system1.task_registry import REAL_TASKS  # noqa: E402
from rwkv_system1.train import evaluate, head_logits  # noqa: E402

SEEDS = (0, 1, 2)
METRICS = ("accuracy", "ece", "nll")
DEST = Path(__file__).resolve().parents[1] / "results" / "joint_ensemble.json"
FAMILY = [k for k in REAL_TASKS if k.startswith("massive_")]


def score(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    return {
        "accuracy": accuracy(probs, labels),
        "ece": expected_calibration_error(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "ece_entropy": entropy_confidence_ece(probs, labels),
    }


def probs_of(head: DecisionHead, fs) -> torch.Tensor:
    return head.probabilities(head_logits(head, fs), 1.0)


def load_checked(base: Config) -> dict:
    data = {s: {k: load_features(DEFAULT_BACKBONE, k, s, base) for k in FAMILY}
            for s in ("train", "dev", "test")}
    # Every number this harness prints is a statement about the vectors on disk, and the disk is
    # shared with the shipped pipeline.  One sidecar whose ``inputs`` disagree with what the current
    # code would produce means an edit landed after the last extraction -- the per-task row would
    # then read as a regression of the shipped head rather than as stale cache.
    bad = [k for k in FAMILY if data["train"][k].inputs
           != feature_inputs(base, DEFAULT_BACKBONE, data_version(k), k)]
    if bad:
        raise SystemExit(f"cache identity disagrees with the current code for {bad}; "
                         f"re-extract before measuring")
    return data


def measure(base: Config) -> tuple[list[dict], dict]:
    data = load_checked(base)
    JT = concat_joint([(k, data["train"][k]) for k in FAMILY])
    JD = concat_joint([(k, data["dev"][k]) for k in FAMILY])
    print(f"joint arm trains on {len(JT)} rows over {len(FAMILY)} languages; "
          f"per-task arm on {sum(len(data['train'][k]) for k in FAMILY) // len(FAMILY)} "
          f"rows per language", flush=True)

    # task -> arm -> seed -> metric dict, plus the raw probabilities for the averaging arms
    per_seed: dict[str, dict[str, list[dict]]] = {k: {"per-task": [], "joint": []} for k in FAMILY}
    raw: dict[str, dict[str, list[torch.Tensor]]] = {k: {"per-task": [], "joint": []} for k in FAMILY}

    for seed in SEEDS:
        t0 = time.perf_counter()
        joint, _ = train_head(replace(base, seed=seed), JT, JD, None, log=lambda *a: None)
        jt = time.perf_counter() - t0
        for k in FAMILY:
            te = data["test"][k]
            raw[k]["joint"].append(probs_of(joint, te))
            per_seed[k]["joint"].append(evaluate(joint, te, temperature=1.0))
        del joint
        print(f"  seed {seed}: joint trained in {jt:.0f}s", flush=True)

        for k in FAMILY:
            head, _ = train_head(replace(base, seed=seed), data["train"][k], data["dev"][k],
                                 None, log=lambda *a: None)
            te = data["test"][k]
            raw[k]["per-task"].append(probs_of(head, te))
            per_seed[k]["per-task"].append(evaluate(head, te, temperature=1.0))
            del head
        print(f"  seed {seed}: {len(FAMILY)} per-task heads done", flush=True)

    rows = []
    for k in FAMILY:
        labels = data["test"][k].labels
        arms = {
            "per-task": {m: st.mean([r[m] for r in per_seed[k]["per-task"]]) for m in
                         ("accuracy", "ece", "nll", "ece_entropy")},
            "per-task_ens": score(torch.stack(raw[k]["per-task"]).mean(0), labels),
            "joint": {m: st.mean([r[m] for r in per_seed[k]["joint"]]) for m in
                      ("accuracy", "ece", "nll", "ece_entropy")},
            "joint_ens": score(torch.stack(raw[k]["joint"]).mean(0), labels),
        }
        rows.append({"task": k, "n_test": int(len(labels)), "arms": arms})
        print(f"  {k:<14} per-task {arms['per-task']['accuracy'] * 100:6.2f}%  "
              f"ens {arms['per-task_ens']['accuracy'] * 100:6.2f}%  |  "
              f"joint {arms['joint']['accuracy'] * 100:6.2f}%  "
              f"ens {arms['joint_ens']['accuracy'] * 100:6.2f}%", flush=True)
    return rows, data


def compare(rows: list[dict], a: str, b: str, unit: str = "pt") -> None:
    """Paired per-task delta b - a, printed with its se and win count."""
    scale = 100.0 if unit == "pt" else 1.0
    better = {"accuracy": 1, "ece": -1, "nll": -1, "ece_entropy": -1}
    for m in METRICS:
        d = [(r["arms"][b][m] - r["arms"][a][m]) for r in rows]
        wins = sum(v * better[m] > 0 for v in d)
        sd = st.stdev(d)
        print(f"  {a} -> {b:<15} {m:<9} mean {st.mean(d) * scale:+.3f}{unit}  "
              f"se {sd / math.sqrt(len(d)) * scale:.3f}  better {wins}/{len(d)}")


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    unknown_flag = flags - {"--identity-only"}
    if unknown_flag:
        raise SystemExit(f"unknown flag(s): {sorted(unknown_flag)}")
    tasks = [t.strip() for a in argv if not a.startswith("--") for t in a.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in REAL_TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}")
    global FAMILY
    if tasks:
        FAMILY = tasks
    base = Config()
    if "--identity-only" in flags:
        load_checked(base)
        print(f"identity of all {len(FAMILY)} caches agrees with the current code")
        return 0
    print(f"backbone {DEFAULT_BACKBONE}   seeds {SEEDS}   family {FAMILY}\n", flush=True)
    rows, data = measure(base)
    DEST.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    # The fingerprint of what was actually read, kept next to the numbers: a table whose cache has
    # since been superseded is still readable, months later, as the table it was.
    (DEST.with_suffix(".identity.json")).write_text(
        json.dumps({k: data["train"][k].inputs for k in FAMILY}, indent=1, ensure_ascii=False),
        encoding="utf-8")

    print("\n########## controls: this harness vs the two finished sweeps ##########")
    compare(rows, "per-task", "per-task_ens")
    print("\n########## the question: does averaging still pay on a joint head? ##########")
    compare(rows, "joint", "joint_ens")
    print("\n########## and the size of each lever, per language ##########")
    compare(rows, "per-task", "joint")
    print(f"\nwrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
