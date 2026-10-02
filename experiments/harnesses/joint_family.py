"""One head per *family* instead of one per task: does sharing across languages pay?

The four-head-versus-one-joint question was measured once, on `laya_*` + `ag_news` (2026-09-22), and
the verdict flipped depending on whether the head carried slot embeddings -- with them off, joint
won where a single head is data-starved (+5 pt on `laya_category`, 225 rows) and lost on the one
score task, because one readout has to serve four rubrics and one temperature bucket pool.  That is
not the same question as this one: here the tasks are **one task in several languages**, so the
option list, the instruction text and the label space are identical and only the state differs.
Sharing is therefore much less of a stretch, and MASSIVE is exactly where a head is data-starved
(1,100 rows over 60 intents per language).

Arms, at the same seeds and off the same cached features:

    per-task   one head per language, what ships
    per-family one head trained on every language of the family at once

The joint arm also pools the temperature bucket (one scalar for the family rather than one per
language) -- that is part of what is being tested, not an oversight, and it is what cost
``laya_category`` ~5 pt of ECE in the earlier comparison.  Both arms report accuracy / ECE / NLL at
T=1 (the shipped headline) plus each arm's own fitted temperature.

Usage::

    python experiments/harnesses/joint_family.py [massive|xnli|amazon]
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

from rwkv_system1.cache_layout import DEFAULT_BACKBONE, load_features  # noqa: E402
from rwkv_system1.features import concat_joint  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.task_registry import REAL_TASKS  # noqa: E402
from rwkv_system1.train import evaluate

SEEDS = (0, 1, 2)
DEST = Path(__file__).resolve().parents[1] / "results" / "joint_family.json"

#: Families that share one option list and one question type across their tasks, which is the whole
#: premise -- a joint head over mismatched label spaces is the thing that already lost.
FAMILIES = {
    "massive": [k for k in REAL_TASKS if k.startswith("massive_")],
    "xnli": [k for k in REAL_TASKS if k.startswith("xnli_")],
    "amazon": [k for k in REAL_TASKS if k.startswith("amazon_reviews_")],
}


def load_all(tasks, cfg):
    return {s: {k: load_features(DEFAULT_BACKBONE, k, s, cfg) for k in tasks}
            for s in ("train", "dev", "test")}


def measure(family: str, base: Config) -> dict:
    tasks = FAMILIES[family]
    cfg = replace(base, seed=SEEDS[0])
    data = load_all(tasks, cfg)
    shared = {k: tuple(data["train"][k].options) for k in tasks}
    if len({v for v in shared.values()}) != 1:
        raise SystemExit(f"{family} is not one option list, so a joint head is not being tested: "
                         + json.dumps({k: v[:3] for k, v in shared.items()}, ensure_ascii=False))
    JT = concat_joint([(k, data["train"][k]) for k in tasks])
    JD = concat_joint([(k, data["dev"][k]) for k in tasks])
    print(f"\n=== {family}: {len(tasks)} tasks, one option list of {JT.features.shape[1]} ===")
    print(f"joint  {len(JT)} train rows / {len(JD)} dev   per-task "
          + " ".join(f"{k.split('_')[-1]}:{len(data['train'][k])}" for k in tasks))

    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        joint, jhist = train_head(replace(base, seed=seed), JT, JD, None, log=lambda *a: None)
        print(f"  seed {seed} joint trained in {time.perf_counter() - t0:.0f}s", flush=True)
        for k in tasks:
            te = data["test"][k]
            tq = (int(te.qtype), int(te.n_options))
            t1 = evaluate(joint, te, temperature=1.0)
            tj = jhist.temperature_for(*tq)
            rows.append({"family": family, "task": k, "seed": seed, "arm": "joint",
                         "temperature": tj,
                         "t1": {m: t1[m] for m in ("accuracy", "ece", "nll")},
                         "fit": {m: evaluate(joint, te, temperature=tj)[m]
                                 for m in ("accuracy", "ece", "nll")}})
        del joint, jhist
        for k in tasks:
            head, hist = train_head(replace(base, seed=seed), data["train"][k], data["dev"][k],
                                    None, log=lambda *a: None)
            te = data["test"][k]
            t1 = evaluate(head, te, temperature=1.0)
            tj = hist.temperature_for(int(te.qtype), int(te.n_options))
            rows.append({"family": family, "task": k, "seed": seed, "arm": "per-task",
                         "temperature": tj,
                         "t1": {m: t1[m] for m in ("accuracy", "ece", "nll")},
                         "fit": {m: evaluate(head, te, temperature=tj)[m]
                                 for m in ("accuracy", "ece", "nll")}})
            del head, hist
    return {"family": family, "tasks": tasks, "n_joint_rows": len(JT), "rows": rows}


def summarise(result: dict) -> None:
    rows, tasks = result["rows"], result["tasks"]
    def get(arm, task, key, metric):
        vals = [r[key][metric] for r in rows
                if r["arm"] == arm and r["task"] == task and key in r]
        return st.mean(vals)

    print(f"\n########## {result['family']}: per-task head vs one head for the whole family ##########")
    print(f"{'task':<16}{'acc/task':>9}{'acc joint':>10}{'d_acc':>8}   "
          f"{'ece/task':>9}{'ece joint':>10}{'d_ece':>9}   {'nll d':>8}")
    d_acc, d_ece, d_nll = [], [], []
    for k in tasks:
        a1, aj = get("per-task", k, "t1", "accuracy"), get("joint", k, "t1", "accuracy")
        e1, ej = get("per-task", k, "t1", "ece"), get("joint", k, "t1", "ece")
        n1, nj = get("per-task", k, "t1", "nll"), get("joint", k, "t1", "nll")
        d_acc.append(aj - a1)
        d_ece.append(ej - e1)
        d_nll.append(nj - n1)
        print(f"{k:<16}{a1 * 100:>8.2f}%{aj * 100:>9.2f}%{(aj - a1) * 100:>+8.2f}   "
              f"{e1:>9.4f}{ej:>10.4f}{ej - e1:>+9.4f}   {nj - n1:>+8.4f}")

    def line(name, vals, better, scale=100.0, unit="pt"):
        wins = sum(v * better > 0 for v in vals)
        sd = st.stdev(vals) if len(vals) > 1 else float("nan")
        print(f"  {name:<10} mean {st.mean(vals) * scale:+.2f}{unit}  sd {sd * scale:.2f}"
              f"  se {sd / math.sqrt(len(vals)) * scale:.2f}  joint better {wins}/{len(vals)}")

    print(f"\n{len(tasks)} tasks in the family (mean over {len(SEEDS)} seeds each arm):")
    line("accuracy", d_acc, +1)
    line("ece", d_ece, -1, scale=1.0, unit="")
    line("nll", d_nll, -1, scale=1.0, unit="nats")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    family = argv[0] if argv else "massive"
    if family not in FAMILIES:
        raise SystemExit(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")
    print(f"backbone {DEFAULT_BACKBONE}   seeds {SEEDS}   family {family}")
    result = measure(family, Config())
    prior = json.loads(DEST.read_text(encoding="utf-8")) if DEST.exists() else []
    prior = [r for r in prior if r.get("family") != family]
    DEST.write_text(json.dumps([*prior, result], indent=1), encoding="utf-8")
    summarise(result)
    print(f"\nwrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
