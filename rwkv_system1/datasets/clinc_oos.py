"""CLINC150: 151 intent labels, the fresh holdout for the gather-rule decision.

**Not a suite task, and it must not become one.**  The 34 registered tasks have already been used to
choose the head width, the RLCD recipe, the joint-head question and now the marker-gather rule, so
in practice they are validation sets and their numbers carry that.  This corpus never entered any of
those sweeps; it exists to be extracted **once** under a recipe that is already frozen and reported,
and the rule for using it is that nothing may be chosen on it afterwards.  That is why it is absent
from ``REAL_TASKS``: being in the registry is what puts a corpus into every later sweep.

Why this one: ``pooled`` pays off by the size of the *label space* (12 of 12 tasks at >= 28 options
gained, mean +9.8 pt; the 16 smaller ones a wash of +0.7), and that was found on MASSIVE's 60
intents, banking77's 77 and go_emotions' 28.  A fourth at 151 labels is the decisive test -- if the
gain does not appear here, the pin is overfit to the tasks that chose it.

The label *names* are read from the parquet's ``huggingface`` metadata (the column itself is an
int64) rather than hardcoding 151 strings the upstream repo could renumber.
"""
from __future__ import annotations

import json
import random

import pyarrow.parquet as pq

from ..decision import DecisionExample, DecisionType, Question
from ._hub import SPLIT_SEED, _by_question, _local, _parquet

CLINC_REPO = "clinc/clinc_oos"
#: `plus` is the full 150-intent version with every class present; `small` drops classes and
#: `imbalanced` skews the counts, neither of which is what a label-space question wants.
CLINC_CONFIG = "plus"


def intent_names(repo_file: str) -> list[str]:
    """The ``ClassLabel`` names of a split file, in label-index order."""
    blob = pq.ParquetFile(_local(CLINC_REPO, repo_file)).schema_arrow.metadata.get(b"huggingface")
    if blob is None:
        raise RuntimeError(f"{repo_file} carries no huggingface metadata; no label names to read")
    return list(json.loads(blob.decode("utf-8"))["info"]["features"]["intent"]["names"])


def load_clinc_oos(
    n_train: int = 1000,
    n_dev: int = 300,
    n_test: int = 400,
    config: str = CLINC_CONFIG,
) -> dict[str, list[DecisionExample]]:
    """CLINC150 as a 151-way choice question, with the corpus's own three splits.

    The one label that is a rejection option rather than an intent (``oos``) stays in the option
    list: `plus` balances it like any other class, so the question remains a plain 151-way
    classification and dropping rows would be a choice this corpus was not asked to make.

    File rows are grouped by intent and are therefore shuffled before slicing -- the same defect
    that made ``amazon_reviews`` a single class and half of MASSIVE's intents unanswerable.
    """
    names = intent_names(f"{config}/train-00000-of-00001.parquet")
    question = Question(
        name="clinc_intent",
        instructions="Which intent is the user expressing?",
        type=DecisionType.CHOICE,
        options=sorted(names),
    )

    def build(split: str, limit: int) -> list[DecisionExample]:
        rows = _parquet(CLINC_REPO, f"{config}/{split}-00000-of-00001.parquet")
        random.Random(SPLIT_SEED).shuffle(rows)
        return _by_question(rows[:limit], lambda r: r["text"], lambda r: names[r["intent"]],
                            question)

    out = {
        "train": build("train", n_train),
        "dev": build("validation", n_dev),
        "test": build("test", n_test),
    }
    covered = len({e.answer for e in out["train"]})
    print(f"    clinc_oos/{config}: {len(names)} labels | {len(out['train'])} train rows covering "
          f"{covered} labels, {len(out['dev'])} dev, {len(out['test'])} test")
    return out
