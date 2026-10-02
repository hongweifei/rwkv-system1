"""XNLI: 3-way entailment, one language at a time."""
from __future__ import annotations

from ..decision import DecisionExample, DecisionType, Question
from ._hub import SPLIT_SEED, _head


XNLI_REPO = "facebook/xnli"



XNLI_LABELS = ["entailment", "neutral", "contradiction"]



def load_xnli(
    lang: str = "en",
    n_train: int = 400,
    n_dev: int = 100,
    n_test: int = 300,
) -> dict[str, list[DecisionExample]]:
    """XNLI natural-language inference, 3 ways, one language at a time.

    Both the premise and the hypothesis vary per row, and only the *state* may vary, since
    the option list and the instructions are shared by every row of a task.
    """
    question = Question(
        name="xnli",
        instructions="Does the premise entail the hypothesis, contradict it, or neither?",
        type=DecisionType.CHOICE,
        options=XNLI_LABELS,
    )

    def build(split: str, limit: int, seed: int | None = None) -> list[DecisionExample]:
        rows = _head(XNLI_REPO, f"{lang}/{split}-00000-of-00001.parquet", limit, seed)
        return [
            DecisionExample(
                state=f"{r['premise']}\n{r['hypothesis']}",
                question=question,
                answer=int(r["label"]),
                true_probs=None,
                meta={"source": f"xnli/{lang}", "lang": lang},
            )
            for r in rows
            if len(r["premise"].strip()) > 3
        ]

    # dev is the dataset's own validation split rather than the next `n_dev` rows of train: the
    # carved dev spent the row budget twice and was 100 rows, and 500 is where this project has
    # measured that a temperature fit starts to be earned.  Shuffled before slicing because a
    # re-vendored file could be grouped -- the check that said these three are balanced is in the
    # session notes, not in the code.
    out = {"train": build("train", n_train),
           "dev": build("validation", n_dev, SPLIT_SEED + 3),
           "test": build("test", n_test)}
    print(f"    xnli/{lang}: " + ", ".join(f"{k} {len(v)}" for k, v in out.items())
          + f" of that language's {n_train}/{n_dev}/{n_test} requested")
    return out

