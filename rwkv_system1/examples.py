"""Turning a task key into decision examples, and examples into extraction queries.

Core, not CLI: the teacher prompt needs the rendered questions, so a module below the pipeline
has to be able to build them.  ``Config`` is imported for the annotation only, which is what keeps
this from depending on ``pipeline``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import corpus
from .cache_layout import ROOT
from .decision import DecisionExample
from .task_registry import REAL_TASKS, SYNTHETIC_TASKS, all_task_keys

if TYPE_CHECKING:
    from .pipeline.config import Config


def build_examples(cfg: Config, keys=None) -> dict[str, dict[str, list[DecisionExample]]]:
    out = {}
    for key in (keys or all_task_keys()):
        if key in REAL_TASKS:
            task = REAL_TASKS[key]
            # Config supplies the default sizes; a task may pin its own, because the corpora
            # differ wildly in how many rows they have (urgency pools four workflows, category
            # has one).
            kwargs = {"n_train": cfg.n_train, "n_dev": cfg.n_dev, "n_test": cfg.n_test}
            kwargs.update(task.kwargs)
            if "path" in kwargs:
                kwargs["path"] = ROOT / kwargs["path"]
            print(f"  loading real corpus {key}")
            out[key] = task.loader(**kwargs)
            continue
        spec = SYNTHETIC_TASKS[key]
        tr, dv, te = corpus.split(
            spec,
            cfg.n_train,
            cfg.n_dev,
            cfg.n_test,
            seed=7,
            n_cues=cfg.n_cues,
            p_neutral=cfg.p_neutral,
            n_filler=cfg.n_filler,
        )
        out[key] = {"train": tr, "dev": dv, "test": te}
    return out


def queries_for(examples: list[DecisionExample]) -> list[tuple[str, str, list[str]]]:
    return [
        (e.question.type.label, e.question.instructions, e.question.options)
        for e in examples
    ]
