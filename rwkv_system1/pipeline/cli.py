"""Argument parsing and dispatch -- one module per command lives in ``commands/``."""

from __future__ import annotations

import argparse
import json
import sys

from ..cache_layout import BACKBONES, DEFAULT_BACKBONE, RUNS
from ..features import GATHER_RULES
from ..task_registry import all_task_keys
from .commands.ablate import cmd_ablate
from .commands.data import cmd_data
from .commands.eval import cmd_eval
from .commands.extract import cmd_extract
from .commands.predict import cmd_predict
from .commands.train import cmd_train
from .config import Config


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a codepage that cannot print the CJK task names.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="rwkv_system1.pipeline")
    p.add_argument("command", choices=["data", "extract", "train", "eval", "ablate", "predict"])
    p.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    p.add_argument("--n", type=int, default=3, help="examples for predict")
    p.add_argument("--force", action="store_true", help="re-extract even if cached")
    p.add_argument("--tasks", default=None,
                   help="comma-separated subset of " + ",".join(all_task_keys()))
    p.add_argument("--targets", default=None, choices=["gold", "crossfit", "teacher", "hard"])
    p.add_argument("--teacher-model", default=None,
                   help="online teacher for --targets teacher (default qwen3.8-flash)")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-dev", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    # A *feature* knob: it selects a different cache and a different head, so it has to be given
    # to `extract`, not only to `train`.
    p.add_argument("--feature-rule", default=None, choices=sorted(GATHER_RULES),
                   help="how a marker position becomes a feature vector")
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--rlcd-epochs", type=int, default=None)
    # Head-side, and deliberately not in `Config`: the members of an ensemble are ordinary
    # single-seed heads, so this decides how many artifacts a command *reads*, not what any artifact
    # is. Putting it in the recipe file would let machine state change what a reported number means,
    # which is the bug the gather-rule knob just moved out of.
    p.add_argument("--ensemble-seeds", type=int, default=1,
                   help="average the softmaxes of N seed replicas (train and eval both take this)")
    # default=None rather than False so an absent flag leaves the config file's value alone.
    p.add_argument("--warmup-soft-targets", action="store_true", default=None,
                   help="train the warm-up on the soft target (distillation)")
    args = p.parse_args(argv)

    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in all_task_keys()]
        if unknown:
            raise SystemExit(f"unknown task(s) {unknown}; choose from {all_task_keys()}")
        args.task_list = wanted
    else:
        args.task_list = all_task_keys()

    cfg = Config()
    cfg_path = RUNS / f"{args.backbone}_config.json"
    recipe = ("warmup_epochs", "rlcd_epochs", "lr", "batch_size", "weight_decay", "targets")
    overridden = []
    if args.command != "data" and cfg_path.exists():
        for k, v in json.loads(cfg_path.read_text(encoding="utf-8")).items():
            if k == "feature_rule":
                # Deliberately not inherited.  This file is where a command *resumes* from, and a
                # rule recorded by one diagnostic run then silently overrode every task's pin --
                # which is how the first shipped table came out all-`last` with the pins in place.
                # The gather rule is decided by the task (``RealTask.feature_rule``) or named on
                # the command line, never by machine state; the arms that need one pass it to each
                # command anyway.
                if v is not None:
                    print(f"  note: {cfg_path.name} records feature_rule={v!r}, which is no longer"
                          f" inherited (the task's pin decides; use --feature-rule to override)")
                continue
            if hasattr(cfg, k):
                if k in recipe and v != getattr(Config(), k):
                    overridden.append(f"{k}={v!r}")
                setattr(cfg, k, v)
            else:
                # A key left by an older Config. Loading it would create an attribute no code
                # reads, which is how a dead knob keeps looking alive.
                print(f"  note: {cfg_path.name} carries {k!r}, which this Config no longer has")
    if overridden:
        # This file used to be rewritten by `train` too, so a one-off flag on a smoke test
        # silently became the standing recipe for every later run -- which is how four heads
        # were trained for 2 epochs and reported as collapsed.  Only `extract` writes it now.
        print(f"  note: {cfg_path.name} overrides the default recipe: {', '.join(overridden)}")
    # Explicit CLI flags win over whatever the last training run recorded.
    for opt in ("n_train", "n_dev", "n_test", "warmup_epochs", "rlcd_epochs", "targets",
                "teacher_model", "warmup_soft_targets", "feature_rule"):
        value = getattr(args, opt)
        if value is not None:
            setattr(cfg, opt, value)

    {"data": cmd_data, "extract": cmd_extract, "train": cmd_train,
     "eval": cmd_eval, "ablate": cmd_ablate, "predict": cmd_predict}[args.command](cfg, args)
    return 0
