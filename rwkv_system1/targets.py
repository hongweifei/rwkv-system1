"""What distribution the proper-score stage should calibrate to.

One concept, in one file -- and it lives here rather than in ``cache_layout`` because that file's
contract is identity only ("it never trains and never extracts; it decides names") while every
branch below does one of those two things: ``gold`` reads a tensor, ``crossfit`` trains fold
models, ``teacher`` loads corpora and calls an API.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch

from .cache_layout import (CACHE, data_version, feature_fingerprint, head_kwargs, load_features,
                           size_tag)
from .features import FeatureSet
from .train import cross_fit_targets

if TYPE_CHECKING:
    from .pipeline.config import Config


def get_soft_targets(backbone_name: str, key: str, cfg: Config, tr: FeatureSet,
                     force: bool = False):
    """The distribution the proper-score stage should calibrate to.

    ``gold`` uses the dataset's own probability targets when it ships them -- Laya's
    typed-decisions corpus does, aggregated over several samples, and that is what makes the
    proper rule meaningful. ``crossfit`` synthesises targets out of the model's own
    out-of-fold predictions; measured on the synthetic corpora those regress to the mean and
    the student faithfully inherits the error, so it is not the default. ``teacher`` asks an
    online model for a distribution over the same option set and temperature-fits it on dev --
    the only way to give a hard-label task a soft target, and measured negative on the one task
    it was tried on.  :mod:`rwkv_system1.teacher` carries the three-seed decomposition that
    closed this for clean-label tasks: the teacher's answer costs 2.67±0.33pt against the labels
    it replaces, and its hesitation adds 0.84±0.83pt, which is not resolved.  ``hard`` returns
    nothing and the stage falls back to one-hot labels, which reduces the log-score term to
    cross-entropy.
    """
    mode = cfg.targets
    if mode == "hard":
        return None
    if mode == "gold":
        if tr.true_probs is None:
            print(f"  soft targets: {key} ships no gold probabilities -> falling back to hard labels")
            return None
        t = tr.true_probs.float()
        print(f"  soft targets: {key} gold, mean max prob {float(t.max(-1).values.mean()):.3f}")
        return t
    if mode == "teacher":
        # `build_examples` is what carries the question instructions the prompt needs, so this
        # mode pays a corpus load even when the distributions are already cached; the paid calls
        # are not repeated.
        from .examples import build_examples
        from .teacher import soft_targets

        return soft_targets(key, build_examples(cfg, [key])[key], cfg, backbone_name, force=force)
    if mode != "crossfit":
        raise SystemExit(f"unknown targets mode {mode!r}; use gold, crossfit, teacher or hard")
    if not cfg.crossfit:
        return None
    # Any of these changing changes the targets, so hash the lot rather than trying to
    # spell them out in the filename.
    payload = {
        "features": feature_fingerprint(cfg, backbone_name, key),
        "head": [cfg.head_d_model, cfg.head_layers, cfg.head_dropout,
                 cfg.head_slot_embeddings],
        "folds": cfg.crossfit_folds,
        "epochs": cfg.crossfit_epochs,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "weight_decay": cfg.weight_decay,
    }
    tag = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    dest = CACHE / f"crossfit_{backbone_name}_{key}_{size_tag(key, cfg)}_{tag}.pt"
    if dest.exists() and not force:
        print(f"  cross-fit targets: cached {dest.name}")
        return torch.load(str(dest), map_location="cpu", weights_only=True)
    print(f"  cross-fit targets: {cfg.crossfit_folds} folds, {cfg.crossfit_epochs} epochs each")
    targets = cross_fit_targets(tr, head_kwargs(cfg, tr), n_folds=cfg.crossfit_folds,
                                epochs=cfg.crossfit_epochs)
    torch.save(targets, str(dest))
    return targets
