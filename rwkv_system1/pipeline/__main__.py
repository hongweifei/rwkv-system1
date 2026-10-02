"""`python -m rwkv_system1.pipeline <command>` -- what the shell chains and the docs invoke.

    uv run python -m rwkv_system1.pipeline data
    uv run python -m rwkv_system1.pipeline extract --backbone rwkv7-g1d-0.1b
    uv run python -m rwkv_system1.pipeline train --tasks ag_news
    uv run python -m rwkv_system1.pipeline eval --tasks ag_news
    uv run python -m rwkv_system1.pipeline ablate --tasks ag_news
    uv run python -m rwkv_system1.pipeline predict --n 3
"""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
