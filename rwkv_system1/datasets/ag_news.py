"""AG News: 4-way news topic, the public benchmark with a published Laya number."""
from __future__ import annotations

from ..decision import DecisionExample, DecisionType, Question
from ._hub import _head


AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]



def load_ag_news(
    n_train: int = 4000,
    n_dev: int = 500,
    n_test: int = 1000,
    max_chars: int = 1200,
) -> dict[str, list[DecisionExample]]:
    """AG News -- the one real public benchmark with a published Laya number (0.953).

    Labels are hard, so there is no soft target and no analytic posterior: accuracy is the
    honest metric, and ECE is measured against single labels.
    """
    question = Question(
        name="topic",
        instructions="Which topic is this news article about?",
        type=DecisionType.CHOICE,
        options=AG_NEWS_LABELS,
    )

    def build(split: str, limit: int) -> list[DecisionExample]:
        out = []
        for row in _head("fancyzhx/ag_news", f"data/{split}-00000-of-00001.parquet", limit):
            text = row["text"].strip().replace("\n", " ")
            if len(text) < 40:
                continue
            out.append(
                DecisionExample(
                    state=text[:max_chars],
                    question=question,
                    answer=int(row["label"]),
                    true_probs=None,
                    meta={"source": "ag_news"},
                )
            )
        return out

    test_pool = build("test", n_dev + n_test)
    print(f"    ag_news: {n_train} train, {n_dev} dev, {n_test} test")
    return {
        "train": build("train", n_train),
        "dev": test_pool[:n_dev],
        "test": test_pool[n_dev : n_dev + n_test],
    }

