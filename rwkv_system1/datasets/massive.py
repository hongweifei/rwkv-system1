"""MASSIVE intent recognition: 60 intents, one language at a time."""
from __future__ import annotations

import random
from ..decision import DecisionExample, DecisionType, Question
from ._hub import SPLIT_SEED, _by_question, _jsonl


MASSIVE_REPO = "mteb/amazon_massive_intent"



def load_massive(
    lang: str = "en",
    n_train: int = 1000,
    n_dev: int = 400,
    n_test: int = 500,
    n_intents: int | None = None,
) -> dict[str, list[DecisionExample]]:
    """MASSIVE intent recognition -- 60 ways.

    Laya's MASSIVE number is on 20 options sampled *per case* (``N_OPTS = 20`` in their
    benchmark script); a fixed full 60 is the harder question and the one our cache format can
    actually represent, since one extraction shares one option list.  Utterances average ~8
    tokens, so every token of cost here is the option block, not the text -- that is what makes
    a 60-way task affordable where Banking77's 77 labels are not.

    ``n_intents`` shrinks the *label space* and nothing else: the same corpus, the same texts, the
    same recipe, fewer options, with the rows whose label fell outside the subset dropped (they
    would be unanswerable).  It exists to test whether the ``pooled`` gather rule's gain depends on
    how many options there are -- the rollout found +9.6 pt here at 60 options and ~0 across the
    small-option tasks, and this makes that comparison within one task instead of across tasks.
    Not a benchmark knob: it changes what the split contains, so the caller must keep it out of the
    registry's sizes or the cache name will collide with the real task's.
    """
    train_rows = _jsonl(MASSIVE_REPO, f"train/{lang}.json.gz")
    val_rows = _jsonl(MASSIVE_REPO, f"validation/{lang}.json.gz")
    test_rows = _jsonl(MASSIVE_REPO, f"test/{lang}.json.gz")
    # The option list is the union of every split's labels, which is why these three are read whole.
    options = sorted({r["label"] for r in train_rows + val_rows + test_rows})
    if n_intents is not None:
        options = sorted(random.Random(SPLIT_SEED + 2).sample(options, n_intents))
    question = Question(
        name="massive_intent",
        instructions="Which intent is the user expressing?",
        type=DecisionType.CHOICE,
        options=options,
    )
    # Both files are grouped, not just the train one: `test/en.json.gz` holds 2974 rows over 59
    # intents and its *first 300 rows carry only 30 of them*, in scenario blocks (`alarm_*`,
    # `audio_volume_mute`, `iot_hue_*`, `general_*`, `datetime_*`).  Shuffling only the train file
    # -- which is what this did -- left the test slice covering half the label space, with one or
    # two of its intents absent from train and therefore unanswerable.  It also made the reported
    # accuracy a number about 30 intents under a 60-intent label, which is not comparable with
    # Laya's published figure.
    random.Random(SPLIT_SEED).shuffle(train_rows)
    random.Random(SPLIT_SEED + 1).shuffle(test_rows)
    # The validation file is grouped by intent too, so dev has to be shuffled before it is sliced
    # or the temperature gets fitted on a handful of intents (which is exactly the defect that made
    # the test slice cover 30 of 60).
    random.Random(SPLIT_SEED + 3).shuffle(val_rows)
    if n_intents is not None:
        keep = set(options)
        train_rows = [r for r in train_rows if r["label"] in keep]
        val_rows = [r for r in val_rows if r["label"] in keep]
        test_rows = [r for r in test_rows if r["label"] in keep]
    out = {
        "train": _by_question(train_rows[:n_train], lambda r: r["text"], lambda r: r["label"], question),
        # dev is the dataset's own validation split, not the next `n_dev` rows of train: carving it
        # out spent the row budget twice and left a 100-row dev, and this project has measured that
        # a rescaling is earned only from ~500 dev rows.
        "dev": _by_question(val_rows[:n_dev], lambda r: r["text"], lambda r: r["label"], question),
        "test": _by_question(test_rows[:n_test], lambda r: r["text"], lambda r: r["label"], question),
    }
    width = f"{len(options)} intents" + (f" (n_intents={n_intents})" if n_intents else "")
    print(f"    massive/{lang}: {len(train_rows)} train pool, {len(val_rows)} validation, "
          f"{len(test_rows)} test | {width}")
    return out

