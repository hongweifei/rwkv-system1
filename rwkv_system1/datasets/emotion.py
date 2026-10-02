"""DAIR emotion: 6-way emotion classification."""
from __future__ import annotations

from ..decision import DecisionExample, DecisionType, Question
from ._hub import SPLIT_SEED, _head


EMOTION_REPO = "dair-ai/emotion"



EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]



def load_emotion(
    n_train: int = 1000,
    n_dev: int = 300,
    n_test: int = 600,
) -> dict[str, list[DecisionExample]]:
    """DAIR emotion, 6 ways -- the task Laya reports 0.600 on, so it is directly comparable."""
    question = Question(
        name="emotion",
        instructions="Which emotion does this text express?",
        type=DecisionType.CHOICE,
        options=EMOTION_LABELS,
    )

    def build(split: str, limit: int, seed: int | None = None) -> list[DecisionExample]:
        rows = _head(EMOTION_REPO, f"split/{split}-00000-of-00001.parquet", limit, seed)
        return [
            DecisionExample(
                state=r["text"].strip(),
                question=question,
                answer=int(r["label"]),
                true_probs=None,
                meta={"source": "emotion"},
            )
            for r in rows
            if len(r["text"].strip()) > 3
        ]

    # dev is the dataset's own validation split, not the tail of the train slice: `dair-ai/emotion`
    # ships one, so carving dev out of train spent 300 rows of the budget and left dev at 300.
    out = {"train": build("train", n_train),
           "dev": build("validation", n_dev, SPLIT_SEED + 3),
           "test": build("test", n_test)}
    print(f"    emotion: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()))
    return out

