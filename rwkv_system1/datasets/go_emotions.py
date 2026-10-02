"""GoEmotions: 28 emotions with the annotator vote distribution as the target."""
from __future__ import annotations

import random
from ..decision import DecisionExample, DecisionType, Question
from ._hub import SPLIT_SEED, _parquet


GO_EMOTIONS_REPO = "google-research-datasets/go_emotions"



GO_EMOTIONS_META = {
    "text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id",
    "example_very_unclear",
}



def load_go_emotions(
    n_train: int = 2000,
    n_dev: int = 400,
    n_test: int = 500,
) -> dict[str, list[DecisionExample]]:
    """GoEmotions with its **annotator distribution** as the target, not a hard label.

    This is the only public corpus here besides typed-decisions that says what the truth
    *is*, rather than what one annotator wrote down: `raw` is one row per (comment, rater), so
    grouping by comment id gives a vote count per emotion and the share is a real posterior.
    That is what the posterior_l1/KL columns and the Bayes ceiling need, and it is the honest
    target a strictly proper rule can be held to.

    Two consequences of the task's construction, stated so its numbers are not read as failure:
    the dominant emotion is only the argmax of a spread-out vote (so the ceiling is well below
    1), and `example_very_unclear` rows are dropped rather than labelled -- 28 emotions over
    Reddit comments is genuinely ambiguous.  Splits are carved from `raw/train` because that is
    the only file carrying votes; `simplified/` has hard labels only and is not mixed in.
    """
    rows = _parquet(GO_EMOTIONS_REPO, "raw/train-00000-of-00001.parquet")
    emotions = [c for c in rows[0] if c not in GO_EMOTIONS_META]
    agg: dict[str, dict] = {}
    for r in rows:
        rec = agg.get(r["id"])
        if rec is None:
            rec = agg[r["id"]] = {"text": r["text"], "votes": [0] * len(emotions)}
        # `example_very_unclear` is per rater, so a comment stays usable while at least one
        # rater found it clear -- the same rule `simplified` applies when it picks a dominant
        # label, which is why that file can be ignored rather than merged.
        rec["clear"] = rec.get("clear", False) or not bool(r["example_very_unclear"])
        for i, e in enumerate(emotions):
            rec["votes"][i] += int(bool(r[e]))

    items = [(v["text"], v["votes"]) for v in agg.values()
             if v.get("clear") and sum(v["votes"]) > 0]
    random.Random(SPLIT_SEED).shuffle(items)
    question = Question(
        name="go_emotions",
        instructions="Which emotion does this comment express?",
        type=DecisionType.CHOICE,
        options=emotions,
    )

    def build(chunk) -> list[DecisionExample]:
        out = []
        for text, votes in chunk:
            total = sum(votes)
            probs = [v / total for v in votes]
            out.append(
                DecisionExample(
                    state=text,
                    question=question,
                    answer=max(range(len(votes)), key=votes.__getitem__),
                    true_probs=probs,
                    meta={"source": "go_emotions", "votes": total},
                )
            )
        return out

    need = n_train + n_dev + n_test
    if len(items) < need:
        raise ValueError(f"asked {need} comments, only {len(items)} carry votes")
    print(f"    go_emotions: {len(agg)} comments from {len(rows)} rater rows | "
          f"{len(emotions)} emotions | {len(items)} usable")
    return {
        "train": build(items[:n_train]),
        "dev": build(items[n_train:n_train + n_dev]),
        "test": build(items[n_train + n_dev:need]),
    }

