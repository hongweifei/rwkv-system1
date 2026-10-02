"""Banking77 customer-request topics: 77 labels, the widest option set here."""
from __future__ import annotations

import random
from ..decision import DecisionExample, DecisionType, Question
from ._hub import SPLIT_SEED, _by_question, _parquet


BANKING77_REPO = "mteb/banking77"



def load_banking77(
    n_train: int = 1200,
    n_dev: int = 400,
    n_test: int = 500,
) -> dict[str, list[DecisionExample]]:
    """Banking77 intent topics -- 77 ways.

    The label count is the one Laya's 0.425 was measured on; their 72-vs-77 note is about *Jev*,
    and both are the same underlying problem, which is this task's real difficulty: 77 option
    names cost 574 tokens of prompt per example, so the option block, not the customer text
    (~20 tokens), dominates the cost, and each option is scored from a marker that also saw the
    76 before it.
    """
    train_rows = _parquet(BANKING77_REPO, "data/train-00000-of-00001.parquet")
    test_rows = _parquet(BANKING77_REPO, "data/test-00000-of-00001.parquet")
    options = sorted({r["label_text"] for r in train_rows} | {r["label_text"] for r in test_rows})
    question = Question(
        name="banking77",
        instructions="What is the customer asking about?",
        type=DecisionType.CHOICE,
        options=options,
    )
    random.Random(SPLIT_SEED).shuffle(train_rows)
    random.Random(SPLIT_SEED + 1).shuffle(test_rows)
    out = {
        "train": _by_question(train_rows[:n_train], lambda r: r["text"], lambda r: r["label_text"], question),
        "dev": _by_question(train_rows[n_train:n_train + n_dev], lambda r: r["text"], lambda r: r["label_text"], question),
        "test": _by_question(test_rows[:n_test], lambda r: r["text"], lambda r: r["label_text"], question),
    }
    print(f"    banking77: {len(train_rows)} train pool, {len(test_rows)} test | {len(options)} topics")
    return out

