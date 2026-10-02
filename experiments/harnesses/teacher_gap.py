"""Is the online teacher worth paying for on this task?  Teacher vs student, per task.

The teacher is not a global improvement: measured 2026-09-23 it beats the student by 69pt on
``massive_en`` and by 35pt on ``sst5``, and *loses* by 5pt on ``ag_news`` where the student is
already at 89-95%.  So which tasks to spend on is a per-task question, and this is how it gets
answered -- before the calls are paid for, not after.

Samples ``N_ROWS`` test rows, asks through :func:`rwkv_system1.teacher.fetch` -- the same code the
real run uses, so this verifies the production path rather than a parallel one -- and prints:

    teacher   argmax accuracy of the teacher's distribution against the gold label
    student   the shipped head's accuracy on the same rows
    top       the teacher's mean top probability
    H/Hmax    its mean entropy over ``log k``: 1.0 is uniform, 0 is a point mass

``H/Hmax`` is what decides whether the teacher can add anything *beyond* its label.  A wide task
comes back near 0 (banking77's 77 labels were one option at 100 and the rest at 0), so there the
teacher is a labeller; a narrow ambiguous one comes back spread out (``go_emotions`` 0.64), and
there the proper-score stage has something real to calibrate to.

A single row of the table is a 20-row accuracy, so it carries ~11pt of standard error at p=0.5.
Read the pattern and re-run with ``N_ROWS`` higher before acting on one cell.

Usage, a couple of tasks per process -- ``build_examples`` loads each corpus in full::

    set -a; . ./.env; set +a
    python experiments/harnesses/teacher_gap.py sst5,emotion
"""

from __future__ import annotations

import math
import sys

import torch

from rwkv_system1.cache_layout import DEFAULT_BACKBONE, load_features, load_head
from rwkv_system1.examples import build_examples
from rwkv_system1.pipeline.config import Config
from rwkv_system1.teacher import DEFAULT_MODEL, fetch
from rwkv_system1.train import evaluate


N_ROWS = 20


def probe(key: str, cfg: Config, model: str, n_rows: int) -> dict | None:
    data = build_examples(cfg, [key])
    examples = data[key]["test"][:n_rows]
    question = examples[0].question
    gold = [e.answer for e in examples]

    probs, stats = fetch([e.state for e in examples], question.instructions,
                         list(question.options), model=model, workers=cfg.teacher_workers)
    hit = sum(1 for p, g in zip(probs, gold) if int(p.argmax()) == g)

    fs = load_features(DEFAULT_BACKBONE, key, "test", cfg)
    head, _ = load_head(DEFAULT_BACKBONE, key, cfg)
    # subset() takes a bool mask and refuses an index tensor, by design.
    first = torch.zeros(len(fs), dtype=torch.bool)
    first[:len(examples)] = True
    student = evaluate(head, fs.subset(first))["accuracy"]

    tops = probs.max(-1).values
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1) / math.log(len(question.options))
    return {
        "task": key, "k": len(question.options), "n": len(examples),
        "teacher_acc": hit / len(examples), "student_acc": student,
        "mean_top": float(tops.mean()), "mean_norm_entropy": float(ent.mean()),
        "seconds": stats["mean_seconds"], "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
    }


def main(argv: list[str]) -> int:
    cfg = Config()
    tasks = argv[0].split(",") if argv else ["laya_urgency", "sst5", "emotion"]
    model = argv[1] if len(argv) > 1 else DEFAULT_MODEL
    print(f"teacher {model}, {N_ROWS} test rows per task\n")
    print(f"{'task':<18}{'k':>4}{'n':>4}{'teacher':>9}{'student':>9}{'gap':>9}"
          f"{'top':>7}{'H/Hmax':>8}{'s/call':>8}")
    for key in tasks:
        r = probe(key, cfg, model, N_ROWS)
        print(f"{r['task']:<18}{r['k']:>4}{r['n']:>4}{100 * r['teacher_acc']:>8.1f}%"
              f"{100 * r['student_acc']:>8.1f}%{100 * (r['teacher_acc'] - r['student_acc']):>+8.1f}pt"
              f"{r['mean_top']:>7.2f}{r['mean_norm_entropy']:>8.3f}{r['seconds']:>8.1f}", flush=True)
        print(f"    {r['prompt_tokens']} prompt + {r['completion_tokens']} completion tokens "
              f"for {r['n']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
