"""Train the heads: one per task, or one per group when the registry says they share a head."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import torch

from ...cache_layout import (RUNS, group_members, head_inputs, head_kwargs, load_features,
                             run_name)
from ...decision import DecisionHead
from ...features import concat_joint
from ...targets import get_soft_targets
from ...train import train
from ..config import Config


def _recipe_note(name: str, inputs: dict, target) -> str | None:
    """Why an existing head at ``name`` is about to be replaced, or None if there is none."""
    if not target.exists():
        return None
    try:
        was = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))["head_inputs"]
    except (OSError, KeyError, ValueError):
        return f"  note: overwriting {name}.pt, whose recipe could not be read"
    # The file name does carry the recipe, so reaching here means the recipe matched; what is worth
    # saying out loud is *that* it matched while the file is being replaced, which is a retrain of
    # the same model.  Two-epoch smoke runs once landed on top of sixty-epoch heads and the
    # collapsed result was read as a model failure -- that is what the identity in the name is for.
    differs = [f"{k}: {was.get(k)!r} -> {inputs[k]!r}" for k in sorted(inputs)
               if was.get(k) != inputs[k]]
    if not differs:
        return f"  note: retraining {name}.pt under an identical recipe"
    return f"  note: overwriting a head trained with a different recipe ({'; '.join(differs)})"


def _fit(cfg: Config, tr, dv, soft):
    torch.manual_seed(cfg.seed)
    head = DecisionHead(**head_kwargs(cfg, tr))
    history = train(
        head, tr, dv,
        soft_targets=soft,
        warmup_soft_targets=cfg.warmup_soft_targets,
        warmup_epochs=cfg.warmup_epochs,
        rlcd_epochs=cfg.rlcd_epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        group_size=cfg.group_size,
        pg_weight=cfg.pg_weight,
        w_sph=cfg.w_sph,
        w_rps=cfg.w_rps,
        exploration=cfg.exploration,
        explore_sigma=cfg.explore_sigma,
        permute_options=cfg.permute_options,
        # ``train`` re-seeds internally, so omitting this pinned every shipped head's batch order to
        # seed 0 no matter what the config said -- while ``head_inputs`` recorded ``cfg.seed``, and
        # the harnesses passed their own seed, so a sweep measured a recipe the pipeline did not
        # run.  Harmless so far (all 147 artifacts on disk were trained at seed 0, where the default
        # and the config agree) and fatal the first time someone runs seeds 1 and 2 through here.
        seed=cfg.seed,
    )
    return head, history


def _save(cfg: Config, args, name: str, inputs: dict, head, history) -> None:
    target = RUNS / f"{name}.pt"
    note = _recipe_note(name, inputs, target)
    if note:
        print(note)
    torch.save(head.state_dict(), target)
    (RUNS / f"{name}.json").write_text(
        json.dumps({"config": asdict(cfg), "history": asdict(history),
                    "head_inputs": inputs}, indent=1),
        encoding="utf-8",
    )


def _group_targets(args, cfg: Config, members: list[str], parts: dict, force: bool):
    """Soft targets for the joined rows, in member order, or None when none of them has any.

    ``concat_joint`` lays its members side by side in list order, so the returned tensor is in that
    same order.  A group that mixes targets with one-hot members is refused rather than padded with
    zeros: a zero row is not a one-hot target, and silently training half a group against all-zero
    distributions is exactly the kind of wrong-number-no-message bug this pipeline keeps meeting.
    """
    ts = [get_soft_targets(args.backbone, k, cfg, parts[k], force=force) for k in members]
    if all(t is None for t in ts):
        return None
    if any(t is None for t in ts):
        mixed = [k for k, t in zip(members, ts) if t is None]
        raise SystemExit(
            f"group members without soft targets: {mixed}; the others have them, and joining "
            f"one-hot rows into a soft-target batch needs a decision this command will not make "
            f"silently")
    widths = {int(t.shape[1]) for t in ts}
    if len(widths) > 1:
        raise SystemExit(f"group members disagree on option count: {widths}")
    return torch.cat(ts, 0)


def _seed_cfgs(cfg: Config, n: int) -> list[Config]:
    """One Config per head in the ensemble, differing only in ``seed``.

    The set is always ``0 .. n-1`` because that is the set that was measured; honouring a
    ``--seed 5`` alongside ``--ensemble-seeds 3`` would produce a third ensemble nobody scored.
    """
    if n <= 1:
        return [cfg]
    if cfg.seed not in range(n):
        print(f"  note: --ensemble-seeds {n} trains seeds 0..{n - 1}; the recipe's seed="
              f"{cfg.seed} is not one of them and is ignored")
    return [replace(cfg, seed=s) for s in range(n)]


def _train_group(cfg: Config, args, members: list[str]) -> None:
    seeds = _seed_cfgs(cfg, getattr(args, "ensemble_seeds", 1))
    print(f"\n=== joint head for {'+'.join(members)}"
          f"{f' x {len(seeds)} seeds' if len(seeds) > 1 else ''} ===")
    tr = {k: load_features(args.backbone, k, "train", cfg) for k in members}
    dv = {k: load_features(args.backbone, k, "dev", cfg) for k in members}
    option_lists = {tuple(tr[k].options) for k in members}
    if len(option_lists) > 1:
        raise SystemExit(
            f"{members[0].split('_')[0]} is not one option list, so its head cannot be shared: "
            f"{ {k: len(v) for k, v in ((k, tr[k].options) for k in members)} }")
    JT = concat_joint([(k, tr[k]) for k in members])
    JD = concat_joint([(k, dv[k]) for k in members])
    soft = _group_targets(args, cfg, members, tr, force=args.force)
    print(f"  {len(JT)} train rows / {len(JD)} dev rows over {len(members)} caches, "
          f"{JT.features.shape[1]} options shared")
    # The caches are loaded once and every seed trains off the same rows: re-loading ten MASSIVE
    # caches per seed would cost minutes of I/O for nothing, and the soft targets are seed-free.
    for sc in seeds:
        if len(seeds) > 1:
            print(f"  seed {sc.seed}:")
        head, history = _fit(sc, JT, JD, soft)
        _save(sc, args, run_name(args.backbone, members[0], sc, JT),
              head_inputs(sc, JT, members[0], args.backbone), head, history)


def cmd_train(cfg: Config, args) -> None:
    RUNS.mkdir(exist_ok=True)
    trained: set[frozenset] = set()
    for key in args.task_list:
        members = group_members(key)
        if members:
            group = frozenset(members)
            if group in trained:
                print(f"\n=== {key} === served by the {members[0].split('_')[0]} joint head, "
                      f"already trained in this run")
                continue
            trained.add(group)
            _train_group(cfg, args, members)
            continue
        seeds = _seed_cfgs(cfg, getattr(args, "ensemble_seeds", 1))
        print(f"\n=== {key}{f' x {len(seeds)} seeds' if len(seeds) > 1 else ''} ===")
        tr = load_features(args.backbone, key, "train", cfg)
        dv = load_features(args.backbone, key, "dev", cfg)
        soft = get_soft_targets(args.backbone, key, cfg, tr, force=args.force)
        for sc in seeds:
            if len(seeds) > 1:
                print(f"  seed {sc.seed}:")
            head, history = _fit(sc, tr, dv, soft)
            _save(sc, args, run_name(args.backbone, key, sc, tr),
                  head_inputs(sc, tr, key, args.backbone), head, history)
    # Deliberately NOT rewriting `runs/<backbone>_config.json` here.  That file is the pipeline's
    # recipe, set once by `extract`; letting `train` rewrite it meant a one-off `--warmup-epochs 2`
    # on a smoke test became the standing recipe for every later command, and the four heads it
    # trained were reported as collapsed before the cause was found.  Each head's own `.json`
    # above still records exactly what it was trained with.
