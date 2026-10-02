"""BoolQ: 2-way passage question answering."""
from __future__ import annotations

from ..decision import DecisionExample, DecisionType, Question
from ._hub import _head


def load_boolq(
    n_train: int = 1000,
    n_dev: int = 300,
    n_test: int = 600,
) -> dict[str, list[DecisionExample]]:
    """BoolQ as a two-way question, passage and question both in the state.

    Only the state may vary per row, so the hypothesis-shaped side of the task has to travel
    there; the option list stays the two fixed texts.
    """
    question = Question(
        name="boolq",
        instructions="Does the passage answer the question?",
        type=DecisionType.BOOLEAN,
        options=["the passage does not answer the question", "the passage answers the question"],
        labels=["false", "true"],
    )

    def build(split: str, limit: int) -> list[DecisionExample]:
        rows = _head("google/boolq", f"data/{split}-00000-of-00001.parquet", limit)
        return [
            DecisionExample(state=f"{r['passage']}\n{r['question']}", question=question,
                            answer=int(bool(r["answer"])), true_probs=None,
                            meta={"source": "boolq"})
            for r in rows
        ]

    print(f"    boolq: train {n_train}, dev from train, test from validation (no public test labels)")
    pool = build("train", n_train + n_dev)
    return {"train": pool[:n_train], "dev": pool[n_train:], "test": build("validation", n_test)}

