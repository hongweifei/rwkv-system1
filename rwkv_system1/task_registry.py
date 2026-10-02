"""The task registry: which tasks exist, their sizes, and their data versions.

Both families live here, side by side and named for what they are: ``SYNTHETIC_TASKS`` are the
generated corpora from :mod:`rwkv_system1.corpus` that exist to test *calibration* (they carry an
analytic posterior, and a character-bag model beats this project's own model on them, so their
accuracy is never quoted), and ``REAL_TASKS`` are the real corpora in :mod:`rwkv_system1.datasets`
that exist to test *capability*.  Adding or looking up a task should mean opening one file.

The module lives beside ``cache_layout`` rather than inside ``datasets/`` because it is not a
source: every file in that package is one corpus, and this is the table that registers them (and
carries the ``data_version`` each cache name depends on).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import corpus
from .datasets.ag_news import load_ag_news
from .datasets.amazon_reviews import load_amazon_reviews_multi
from .datasets.banking77 import load_banking77
from .datasets.boolq import load_boolq
from .datasets.emotion import load_emotion
from .datasets.go_emotions import load_go_emotions
from .datasets.laya import load_typed_decisions
from .datasets.massive import load_massive
from .datasets.prompt_injections import load_prompt_injections
from .datasets.sst5 import load_sst5
from .datasets.xnli import load_xnli


@dataclass(frozen=True)
class RealTask:
    loader: Callable
    kwargs: dict = field(default_factory=dict)
    #: Bump when the loader changes what a split *contains* for the same requested sizes --
    #: either the text it emits or which rows land in which split. The cache name records only
    #: the counts, so without a bump a re-shuffled split keeps the old file's name.
    #: v1 *is* the shuffled definition: the pre-shuffle caches and heads were deleted rather
    #: than versioned, because that slicing made one split a whole workflow and no result
    #: quoted from it survived.
    data_version: int = 1
    #: Which marker-gather rule this task runs under, or ``None`` for the shipped default.  Pinned
    #: per task rather than derived from the option count, because a threshold in code is a knob
    #: nothing can audit; an explicit value on ``Config`` or the CLI still wins over this, so the
    #: measuring arms in ``experiments/chains`` stay reproducible (``cache_layout.resolve_rule``).
    #:
    #: ``pooled`` pays off by the size of the *label space* and is a wash below it, measured two
    #: ways on 2026-09-25.  Across the suite: all 12 tasks with >= 28 options gained, 12/12, mean
    #: +9.8 pt (`banking77` 77 options +14.6, the ten MASSIVE languages at 60 +4.7 to +20.9,
    #: `go_emotions` 28 +4.0) against +0.7 pt over the other 16, where the losses are real
    #: (`prompt_injections` -3.5 with its ECE going 0.028 -> 0.095, `laya_urgency` -4.3).  Within
    #: one task: shrinking `massive_en` to 10 of its 60 intents -- same corpus, same texts, same
    #: recipe -- turns +9.6 pt into -2.0 pt, which is what rules out "the gain is just about how
    #: much headroom the task has".
    feature_rule: str | None = None
    #: Tasks sharing this name are served by **one** head, trained on their pooled train and dev
    #: rows; test stays per task.  None (the default) means a head of its own.
    #:
    #: Pinned per group, not derived from the prefix, because the premise is that every member
    #: answers the same question over the same option list -- ``concat_joint`` pads when the widths
    #: differ and would happily join mismatched rubrics into one readout.
    #:
    #: Measured 2026-10-01 on the three families that qualify, 3 seeds, same pinned-pooled caches
    #: (``experiments/results/joint_family.json``):
    #:
    #: * MASSIVE, 10 languages x 1,100 rows over 60 intents: **+9.67 pt** accuracy (se 0.73), every
    #:   language positive (+7.2 to +13.7), NLL **-0.38 nats** on 10/10, ECE a wash (+0.01 absolute,
    #:   4/10).  Both arms fitted T=1.20, so the pooled temperature bucket costs nothing here.
    #: * XNLI, 8 languages x 500 rows over 3 labels: +3.51 pt accuracy but **+0.04 absolute ECE**
    #:   (1/8 better) and NLL *worse* by 0.02 nats (2/8) -- its per-task fitted temperatures span
    #:   1.00 to 5.00, which one scalar cannot serve.  Not grouped.
    #: * amazon reviews, 6 languages x 3,500 rows over 5 stars: mildly positive on all three
    #:   (+1.26 pt, se 0.93; ECE -0.02; NLL -0.05 nats).  Real but inside one standard error of the
    #:   family mean, so it is left alone until it is measured again rather than grouped on a
    #:   maybe.
    #:
    #: The statistical unit is the family, not the language: the ten MASSIVE numbers are one head
    #: read ten ways, not ten replicates.
    joint_group: str | None = None


#: Bumped to 2 on 2026-09-24.  Both files these families read are *grouped* -- MASSIVE's
#: ``test/<lang>.json.gz`` by intent, amazon's ``<lang>/<split>.jsonl`` by class -- and only some of
#: the slice sites shuffled before slicing, so v1 held:
#:
#: * an amazon train split *and* test split that were each one class (`en/train.jsonl` has 200000
#:   rows over 5 classes and its first 1900 are all class 0), which is why those six tasks report
#:   1.0000 accuracy and 0.0000 NLL over a five-star question -- a constant answers them perfectly
#:   and nothing is measured;
#: * a MASSIVE test split covering 30 of the 60 intents, with one or two of those absent from
#:   train; its accuracy is a number about half a label space and is not comparable with Laya's
#:   published figure.
#:
#: MASSIVE's *train* splitting was already correct, so its v1 train features are byte-identical to
#: what v2 produces and can be renamed rather than re-extracted; everything else must be re-run.
MASSIVE_DATA_VERSION = 2
#: v3 on 2026-09-27: dev moves to `validation.jsonl` and test to the dataset's own `test.jsonl`.
#: Both keep their *sizes*, so without the bump the new splits would land on the old names.
REVIEWS_DATA_VERSION = 3


#: The pin for a task whose label space is large enough that the `pooled` gather pays -- see
#: ``RealTask.feature_rule`` for the measurement behind it.  Named once so the twelve uses read as
#: one decision rather than twelve coincidences.
POOLED = "pooled"

#: The one group that shares a head.  Same reason as ``POOLED``: named once, so the ten uses are
#: legible as a single decision -- see ``RealTask.joint_group`` for what it was measured against.
MASSIVE_JOINT = "massive"



REAL_TASKS: dict[str, RealTask] = {
    # Pools all four workflows: the only question whose rubric is identical across them.
    "laya_urgency": RealTask(
        load_typed_decisions,
        {"question_id": "urgency", "n_train": 900, "n_dev": 300, "n_test": 400},
    ),
    # customer_service only -> 300 train / 100 test rows available.
    "laya_category": RealTask(
        load_typed_decisions,
        {
            "question_id": "category",
            "workflows": ["customer_service"],
            "n_train": 225,
            "n_dev": 75,
            "n_test": 100,
        },
    ),
    "laya_needs_human": RealTask(
        load_typed_decisions,
        {
            "question_id": "needs_human",
            "workflows": ["customer_service"],
            "n_train": 225,
            "n_dev": 75,
            "n_test": 100,
        },
    ),
    # n_train raised 4000 -> 16000 on 2026-09-24: the data-scaling curve was still climbing at the
    # old cap (500/1000/2000/4000 rows -> 81.10/83.47/84.50/88.17% accuracy), and this task holds
    # the largest external gap (89.0% against Laya's 95.3%).  Only n_train moves: dev and test come
    # from ag_news' own test file and do not depend on it, so the comparison against the 4000-row
    # table is on the identical test split.
    "ag_news": RealTask(load_ag_news, {"n_train": 16000, "n_dev": 500, "n_test": 1000}),
    # Chinese and English first (user's call, 2026-09-22); the other 12 MASSIVE languages and 13
    # XNLI languages stay unwritten until these four are measured.
    "massive_zh": RealTask(
        load_massive, {"lang": "zh-CN", "n_train": 1100, "n_dev": 500, "n_test": 300},
        data_version=MASSIVE_DATA_VERSION, feature_rule=POOLED, joint_group=MASSIVE_JOINT,
    ),
    # Sizes are pinned low on purpose: at 60 and 77 options the prompt is the cost, not the text.
    "massive_en": RealTask(
        load_massive, {"lang": "en", "n_train": 1100, "n_dev": 500, "n_test": 500},
        data_version=MASSIVE_DATA_VERSION, feature_rule=POOLED, joint_group=MASSIVE_JOINT,
    ),
    "xnli_zh": RealTask(load_xnli, {"lang": "zh", "n_train": 500, "n_dev": 500, "n_test": 300}),
    "xnli_en": RealTask(load_xnli, {"lang": "en", "n_train": 500, "n_dev": 500, "n_test": 300}),
    "emotion": RealTask(load_emotion, {"n_train": 1300, "n_dev": 500, "n_test": 600}),
    # n_train raised 1200 -> 4800 on 2026-09-24: the steepest curve measured, still climbing hard
    # at the old cap (150/300/600/1200 rows -> 5.40/22.40/32.13/45.93%, i.e. +13.8pt on the last
    # doubling) with 8x of the 9993-row pool still unused.  Dev is carved from the train pool, so
    # raising n_train shifts which rows dev holds; it stays disjoint from test, which is what the
    # comparison needs.
    "banking77": RealTask(load_banking77, {"n_train": 4800, "n_dev": 400, "n_test": 500},
                          feature_rule=POOLED),
    # A real annotator distribution (`go_emotions`) and the ordinal axis (`sst5` as a score
    # question, exactly as Laya use it), plus their two remaining two-way tasks.  probe: the
    # trio closes out the suite Laya publish a table for.
    "go_emotions": RealTask(load_go_emotions, {"n_train": 2000, "n_dev": 400, "n_test": 500},
                            feature_rule=POOLED),
    # n_train raised 1000 -> 4000 on 2026-09-24: still climbing at the old cap (125/250/500/1000
    # rows -> 24.06/26.44/27.78/35.50%), and this is a five-level *score* question, so the RPS term
    # gets more to work with too.  dev and test are separate files and are untouched.
    "sst5": RealTask(load_sst5, {"n_train": 4000, "n_dev": 300, "n_test": 600}),
    "boolq": RealTask(load_boolq, {"n_train": 1000, "n_dev": 300, "n_test": 600}),
    "prompt_injections": RealTask(
        load_prompt_injections, {"n_train": 400, "n_dev": 146, "n_test": 116}
    ),
}



MASSIVE_DECLARED = ["de", "fr", "es", "pt", "ru", "ar", "ja", "ko"]  # en and zh-CN are above



XNLI_DECLARED = ["de", "fr", "es", "ru", "ar", "vi"]  # en and zh are above



REVIEWS_DECLARED = ["en", "de", "es", "fr", "ja", "zh"]



#: Raised 400/100 -> 1000/300 on 2026-09-24, together with ``massive_zh``.  The knee was measured
#: first, on the v2 caches, and MASSIVE is the one family where it is nowhere near reached: at
#: 100/200/400 rows ``massive_de`` reads 13.33/13.33/19.33%, ``massive_ar`` 5.16/8.53/14.14% and
#: ``massive_zh`` (96/192/384) 17.24/23.63/32.53% -- every one of them still climbing at the cap,
#: against a 1.67% floor for 60 options.  1000 is where ``massive_en`` already sat, so the family
#: is now sized the same way in every language.  ``n_dev`` goes to 300 to triple the pool the
#: crossfit targets are built from; it is deliberately *not* 500, because the ag_news lesson is
#: that a dev only earns the reported rescaling from 500 rows up, and paying 500 for a column the
#: headline does not quote is not worth the extra extraction.  Dev is carved from the train pool
#: here, so both moves shift which rows land in dev; test is a separate file and is untouched.
for _lang in MASSIVE_DECLARED:
    REAL_TASKS[f"massive_{_lang.replace('-', '_').lower()}"] = RealTask(
        load_massive, {"lang": _lang, "n_train": 1100, "n_dev": 500, "n_test": 300},
        data_version=MASSIVE_DATA_VERSION, feature_rule=POOLED, joint_group=MASSIVE_JOINT,
    )



for _lang in XNLI_DECLARED:
    REAL_TASKS[f"xnli_{_lang}"] = RealTask(
        load_xnli, {"lang": _lang, "n_train": 500, "n_dev": 500, "n_test": 300}
    )



#: Raised 1500/400 -> 3000/500 on 2026-09-24.  Same knee measurement: 375/750/1500 rows read
#: 40.20/44.67/47.80% and are still climbing on the last doubling.  Unlike MASSIVE this family is
#: cheap to grow -- 0.34 s per example measured off the v2 cache write times, against 0.78 s for a
#: 60-option MASSIVE row and 1.85 s for banking77's 77 -- so 2x the rows costs about 9 minutes per
#: language.  The pool is 200000 rows per language, so the cap is still ~1.75% of what is there.
for _lang in REVIEWS_DECLARED:
    REAL_TASKS[f"amazon_reviews_{_lang}"] = RealTask(
        load_amazon_reviews_multi, {"lang": _lang, "n_train": 3500, "n_dev": 500, "n_test": 500},
        data_version=REVIEWS_DATA_VERSION,
    )


#: The generated corpora (see :mod:`rwkv_system1.corpus`).  They are not registered as
#: ``RealTask`` entries because they are not loaded from disk: ``build_examples`` generates them
#: from a spec, with the split sizes coming from ``Config`` and the analytic posterior attached.
SYNTHETIC_TASKS = {
    "ticket_routing": corpus.TICKET_SPEC,
    "urgency": corpus.URGENCY_SPEC,
    "needs_human": corpus.NEEDS_HUMAN_SPEC,
    "topic_label": corpus.TOPIC_SPEC,
}


def all_task_keys() -> list[str]:
    return list(SYNTHETIC_TASKS) + list(REAL_TASKS)
