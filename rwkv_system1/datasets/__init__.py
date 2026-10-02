"""Real evaluation corpora, one module per source.

Each benchmark lives in its own file so a loader and its constants sit together; `_hub`
holds the mirror readers shared by all of them.  The table that registers these sources --
names, sizes, data versions -- is `rwkv_system1.task_registry`, next to the cache layout it
feeds, because it is not itself a source.
"""
from __future__ import annotations

from ._hub import (  # noqa: F401
    HF_MIRROR, SPLIT_SEED, _by_question, _head, _jsonl, _parquet,
)
from .laya import *  # noqa: F401,F403
from .ag_news import *  # noqa: F401,F403
from .massive import *  # noqa: F401,F403
from .banking77 import *  # noqa: F401,F403
from .xnli import *  # noqa: F401,F403
from .emotion import *  # noqa: F401,F403
from .go_emotions import *  # noqa: F401,F403
from .amazon_reviews import *  # noqa: F401,F403
from .sst5 import *  # noqa: F401,F403
from .boolq import *  # noqa: F401,F403
from .prompt_injections import *  # noqa: F401,F403
