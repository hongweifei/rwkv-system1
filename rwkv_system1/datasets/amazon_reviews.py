"""Amazon reviews as a 5-level star-rating score question, six languages."""
from __future__ import annotations

from ..decision import DecisionExample, DecisionType, Question
from ._hub import SPLIT_SEED, _head


REVIEWS_REPO = "mteb/amazon_reviews_multi"



REVIEW_LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]



def load_amazon_reviews_multi(
    lang: str = "en",
    n_train: int = 1500,
    n_dev: int = 400,
    n_test: int = 500,
) -> dict[str, list[DecisionExample]]:
    """Amazon product reviews as a 5-level *score* question, one language at a time.

    The suite had exactly one ordinal task (`laya_urgency`), so the RPS term -- the only rule
    that knows level 2 is nearer level 1 than level 5 -- was barely exercised.  Star ratings are
    ordinal by construction, and this repo ships the same task in six of the twelve declared
    languages, which is the axis that was missing.
    """
    question = Question(
        name="rating",
        instructions="How many stars does this review give?",
        type=DecisionType.SCORE,
        options=REVIEW_LEVELS,
    )

    def build(split: str, limit: int) -> list[DecisionExample]:
        # This repo's per-language files are grouped by class: `en/train.jsonl` holds 200000 rows
        # across 5 classes and its *first 1900 are all class 0*.  Slicing before shuffling -- which
        # is what this did -- therefore produced a single-class train and a single-class test, so
        # the five-star question always answered "1 star", scored whatever accuracy a constant
        # gets, and measured nothing.  `_head` shuffles before it slices and does it by index, so
        # the rows are the ones a shuffle-then-slice would pick without the 200000-row list that
        # used to be materialised to take 3500 of them.
        rows = _head(REVIEWS_REPO, f"{lang}/{split}.jsonl", limit, SPLIT_SEED)
        return [
            DecisionExample(
                state=r["text"].strip().replace("\n", " ")[:1200],
                question=question,
                answer=int(r["label"]),
                true_probs=None,
                meta={"source": f"reviews/{lang}", "lang": lang},
            )
            for r in rows[:limit]
            if len(r["text"].strip()) > 10
        ]

    # The dataset's own three splits.  Until 2026-09-27 this used `validation` as *test* and carved
    # dev out of train -- which spent the row budget twice and, with a `test.jsonl` sitting there
    # unused, made the reported split the wrong one.  `REVIEWS_DATA_VERSION` goes to 3 because dev
    # and test keep their *sizes* while their rows change, and the cache name records only sizes.
    out = {"train": build("train", n_train),
           "dev": build("validation", n_dev),
           "test": build("test", n_test)}
    print(f"    amazon_reviews_multi/{lang}: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()))
    return out

