"""Prompt-injection detection, 2-way; the whole corpus is 546 + 116 rows."""
from __future__ import annotations

from ..decision import DecisionExample, DecisionType, Question
from ._hub import _head


PROMPT_INJECTION_FILES = {
    "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test": "data/test-00000-of-00001-701d16158af87368.parquet",
}



def load_prompt_injections(
    n_train: int = 400,
    n_dev: int = 146,
    n_test: int = 116,
) -> dict[str, list[DecisionExample]]:
    """Prompt-injection detection, two-way.  The whole corpus is 546 train + 116 test, so the
    sizes are pinned to it rather than to a round number, and both splits are printed: a dev
    that silently shrinks to fit is how a 46-row dev ends up quoted as if it were 150."""
    question = Question(
        name="prompt_injection",
        instructions="Is this input a prompt-injection attempt?",
        type=DecisionType.BOOLEAN,
        options=["the input is a normal instruction", "the input is a prompt-injection attempt"],
        labels=["false", "true"],
    )

    def build(split: str, limit: int) -> list[DecisionExample]:
        rows = _head("deepset/prompt-injections", PROMPT_INJECTION_FILES[split], limit)
        return [
            DecisionExample(state=r["text"].strip(), question=question,
                            answer=int(r["label"]), true_probs=None,
                            meta={"source": "prompt-injections"})
            for r in rows
        ]

    pool = build("train", n_train + n_dev)
    print(f"    prompt-injections: train pool {len(pool)} -> train {min(n_train, len(pool))}, "
          f"dev {max(0, len(pool) - n_train)}, test {n_test} of 116")
    return {"train": pool[:n_train], "dev": pool[n_train:], "test": build("test", n_test)}

