"""The command line: Config, the example builder, and one module per command.

Exposes `main` and nothing else, deliberately.  What a caller needs from the data plane is in
`rwkv_system1.cache_layout` (what a cache is called and what invalidates it) and in
`rwkv_system1.task_registry` (which tasks exist); a wider re-export list here would be the kind
of forwarding shim this project has already deleted twice.
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
