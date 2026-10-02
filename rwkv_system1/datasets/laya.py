"""Laya's own training corpus (typed-decisions) -- the one that ships soft targets."""
from __future__ import annotations

import json
import random
from ..decision import DecisionExample, DecisionType, Question, render_criteria_options
from ._hub import SPLIT_SEED, _parquet


TYPED_DECISIONS_REPO = "LocalLLaMA/typed-decisions"



_QTYPE_MAP = {"choice": DecisionType.CHOICE, "score": DecisionType.SCORE, "noul": DecisionType.BOOLEAN}



def load_typed_decisions(
    question_id: str = "urgency",
    workflows: list[str] | None = None,
    n_train: int = 900,
    n_dev: int = 300,
    n_test: int = 400,
) -> dict[str, list[DecisionExample]]:
    """Laya's own typed-decisions corpus, including its *soft* gold probabilities.

    This is the data behind their published 0.766. ``gold`` carries a probability distribution
    per question, aggregated over several samples (the corpus also ships ``label_agreement``
    with a ``total_variation`` per question), which is exactly the kind of target a strictly
    proper rule needs in order to have anything to calibrate to. Feeding these in place of
    cross-fitted targets is the difference between RLCD having signal and having none.

    ``question_id`` must have the same rubric in every workflow pooled. Only ``urgency`` is
    shared by all four (one 4-level rubric); ``action`` and ``disposition`` have per-workflow
    criteria and are refused rather than silently mixed.
    """
    train_rows = _parquet(TYPED_DECISIONS_REPO, "all/train-00000-of-00001.parquet")
    test_rows = _parquet(TYPED_DECISIONS_REPO, "all/test-00000-of-00001.parquet")

    def build(rows: list[dict]) -> list[DecisionExample]:
        spec: tuple[list[str], list[str], dict] | None = None
        out: list[DecisionExample] = []
        for row in rows:
            if workflows and row["workflow"] not in workflows:
                continue
            questions = json.loads(row["questions"])
            gold = json.loads(row["gold"])
            if question_id not in questions or question_id not in gold:
                continue
            q, g = questions[question_id], gold[question_id]
            rendered, labels = render_criteria_options(q["type"], q.get("criteria"))
            if spec is None:
                spec = (rendered, labels, q)
            elif rendered != spec[0]:
                raise ValueError(
                    f"question {question_id!r} has different criteria across workflows; "
                    f"refusing to pool them"
                )
            probs = g.get("probabilities") or {}
            target = [float(probs.get(k, 0.0)) for k in labels]
            total = sum(target)
            if total <= 0:
                continue
            target = [v / total for v in target]
            label = str(g.get("label"))
            answer = (
                labels.index(label)
                if label in labels
                else max(range(len(target)), key=target.__getitem__)
            )
            out.append(
                DecisionExample(
                    state=json.dumps(json.loads(row["state"]), ensure_ascii=False),
                    question=None,  # attached below once the spec is known
                    answer=answer,
                    true_probs=target,
                    meta={
                        "source": "typed-decisions",
                        "workflow": row["workflow"],
                        "question": question_id,
                        "gold_confidence": g.get("confidence"),
                    },
                )
            )
        if spec is None:
            raise ValueError(f"no rows carried question {question_id!r}")
        rendered, labels, q = spec
        question = Question(
            name=question_id,
            instructions=q["instructions"],
            type=_QTYPE_MAP[q["type"]],
            options=rendered,
            labels=labels,
        )
        for ex in out:
            ex.question = question
        return out

    train_all = build(train_rows)
    test = build(test_rows)
    if n_train + n_dev > len(train_all):
        raise ValueError(
            f"asked {n_train}+{n_dev} but only {len(train_all)} train rows for {question_id!r}"
        )
    # The parquet is grouped by workflow -- four blocks of 300 -- so a contiguous slice makes
    # one split *one workflow*. Measured on `urgency`: train is 38% level 1 while a contiguous
    # dev became 53% level 3, which dragged dev accuracy under the 0.25 chance line and fitted
    # the temperature to a single domain. Shuffle both sides, seeded, so every split is a mixed
    # sample; `test` gets the same treatment because it is ordered too.
    random.Random(SPLIT_SEED).shuffle(train_all)
    random.Random(SPLIT_SEED + 1).shuffle(test)
    print(
        f"    typed-decisions/{question_id}: {len(train_all)} train, {len(test)} test | "
        f"mean gold confidence "
        f"{sum(e.meta['gold_confidence'] or 0 for e in train_all) / len(train_all):.3f}"
    )
    return {
        "train": train_all[:n_train],
        "dev": train_all[n_train : n_train + n_dev],
        "test": test[:n_test],
    }

