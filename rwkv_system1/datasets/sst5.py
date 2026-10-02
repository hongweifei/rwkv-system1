"""SST-5 movie-review sentiment as a 5-level score question."""
from __future__ import annotations

from ..decision import DecisionExample, DecisionType, Question
from ._hub import _head, _jsonl


def load_sst5(
    n_train: int = 1000,
    n_dev: int = 300,
    n_test: int = 600,
) -> dict[str, list[DecisionExample]]:
    """SST-5 sentiment as a 5-level score question -- the ordinal task in Laya's own suite.

    Level names are read off ``label_text`` per index rather than hardcoded, because the index
    order is the part a hardcode would get wrong silently.
    """
    names: dict[int, str] = {}
    for split in ("train", "test"):
        for r in _jsonl("SetFit/sst5", f"{split}.jsonl"):
            names.setdefault(int(r["label"]), r["label_text"])
    options = [names[i] for i in sorted(names)]
    question = Question(
        name="sentiment",
        instructions="How positive is this movie review?",
        type=DecisionType.SCORE,
        options=options,
    )

    def build(split: str, limit: int) -> list[DecisionExample]:
        rows = _head("SetFit/sst5", f"{split}.jsonl", limit)
        return [
            DecisionExample(state=r["text"].strip(), question=question,
                            answer=int(r["label"]), true_probs=None,
                            meta={"source": "sst5"})
            for r in rows
        ]

    print(f"    sst5: levels {options}")
    return {"train": build("train", n_train), "dev": build("dev", n_dev),
            "test": build("test", n_test)}

