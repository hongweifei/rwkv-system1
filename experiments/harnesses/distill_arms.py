"""Which half of the teacher is doing the work: its answer, or its hesitation?

Mixing both into one arm can only produce "worse".  On sst5 the teacher is 54.6% accurate on the
very rows it would be teaching from, while the labels are 100% right, so the answer half is strictly
worse than what it replaces; the only thing that could pay is the *shape* of its distribution, the
mass it puts on the options it did not pick.  These arms separate them, with the true labels as the
referee throughout:

    labels        the shipped recipe on the true labels (the reference)
    teacher-hard  the teacher's argmax as the *training* labels, shipped recipe otherwise
    teacher-s2    the teacher's distribution, stage 2 only
    teacher-s1s2  the teacher's distribution, stage 1 and stage 2

``teacher-hard`` needs no new loss: soft cross-entropy against a one-hot *is* cross-entropy against
the argmax, so it is the existing hard-label path on substituted labels.  If ``teacher-hard`` and
``teacher-s1s2`` land on top of each other, the distribution's shape carries nothing and this whole
direction is closed -- which is the question worth spending a run on.

Dev and test keep the true labels in every arm.  Dev picks the epoch and fits the temperature, test
is the verdict; pseudo-labelling either of those would measure the teacher against itself.

Three seeds: one seed could not resolve the earlier 2.3pt gap against this project's 0.6-1.9pt
seed-to-seed sd on a task, which is why the earlier answer had to be "suggestive".

Free to run where the teacher distributions are already cached -- it only retrains heads.  Reuses
``arm_sweep``'s per-arm runner and table rather than growing a second copy of them.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_sweep import report, run_one  # noqa: E402

from rwkv_system1.cache_layout import DEFAULT_BACKBONE, ROOT, load_features  # noqa: E402
from rwkv_system1.examples import build_examples  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.targets import get_soft_targets  # noqa: E402
from rwkv_system1.teacher import targets_for_split  # noqa: E402


SEEDS = (0, 1, 2)
TASKS = ["sst5"]
REFERENCE = "labels"

#: ``(name, recipe overrides, where the *training* labels come from)``.
ARMS: list[tuple[str, dict, str]] = [
    ("labels", {}, "true"),
    ("teacher-hard", {"targets": "hard"}, "teacher"),
    ("teacher-s2", {"targets": "teacher"}, "true"),
    ("teacher-s1s2", {"targets": "teacher", "warmup_soft_targets": True}, "true"),
]


def main(argv: list[str]) -> int:
    tasks = [t for t in (argv[0].split(",") if argv else TASKS) if t]
    base = Config()
    dest = ROOT / "experiments" / "results" / "distill_arms.json"
    previous: list[dict] = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else []
    fresh = {(t, a) for t in tasks for a, _, _ in ARMS}
    rows = [r for r in previous if (r["task"], r["arm"]) not in fresh]
    if len(rows) != len(previous):
        print(f"replacing {len(previous) - len(rows)} rows of {dest.name}, keeping {len(rows)}")

    print(f"backbone {DEFAULT_BACKBONE}   tasks {tasks}   seeds {SEEDS}")
    print(f"arms {[a for a, _, _ in ARMS]}   reference {REFERENCE!r}\n")

    for key in tasks:
        splits = {s: load_features(DEFAULT_BACKBONE, key, s, base)
                  for s in ("train", "dev", "test")}
        examples = build_examples(base, [key])[key]
        # The teacher's own answer on the training rows, from the cache.  `get_soft_targets` below
        # loads the corpus a second time to get the prompt's instructions, which is why this mode
        # is not run over many wide tasks at once.
        teacher = targets_for_split(key, "train", examples["train"], base,
                                    model=base.teacher_model, workers=base.teacher_workers)
        agree = float((teacher.probs.argmax(-1) == splits["train"].labels).float().mean())
        print(f"\n=== {key} ===  teacher agrees with the labels on {100 * agree:.1f}% of the "
              f"{len(splits['train'])} training rows")
        pseudo = replace(splits["train"], labels=teacher.probs.argmax(-1).long(),
                         true_probs=None)

        for arm, overrides, source in ARMS:
            cfg = replace(base, **overrides)
            arm_splits = dict(splits)
            if source == "teacher":
                arm_splits["train"] = pseudo
            soft = get_soft_targets(DEFAULT_BACKBONE, key, cfg, arm_splits["train"])
            for seed in SEEDS:
                rows.append(run_one(key, arm, overrides, seed, base, arm_splits, soft))
        del splits, examples, teacher, pseudo

    report(rows, tasks, [a for a, _, _ in ARMS], reference=REFERENCE)
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
