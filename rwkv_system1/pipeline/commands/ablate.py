"""Cross-entropy against cross-entropy + RLCD, on identical features."""

from __future__ import annotations

import json

import torch

from ...cache_layout import RUNS, head_kwargs, load_features
from ...targets import get_soft_targets
from ...decision import DecisionHead
from ...train import evaluate, train
from ..config import Config


def cmd_ablate(cfg: Config, args) -> None:
    """Does RLCD actually beat plain cross-entropy, on identical features and head?

    Reported side by side because the honest answer is allowed to be "not much".  The
    log-score term of the proper reward reduces to cross-entropy under one-hot labels,
    so any RLCD gain has to come from its bounded spherical term and from the fact that
    the temperature fit is applied to a differently-shaped head.
    """
    print(
        f"{'task':<17}{'variant':<10}{'acc':>8}{'Bayes':>8}{'ECE':>8}"
        f"{'ECEnt':>8}{'NLL':>8}{'postL1':>8}"
    )
    results = {}
    for key in args.task_list:
        tr = load_features(args.backbone, key, "train", cfg)
        dv = load_features(args.backbone, key, "dev", cfg)
        te = load_features(args.backbone, key, "test", cfg)
        # The posterior's own argmax against the labels, not mean(max posterior): see cmd_eval.
        bayes = float((te.true_probs.argmax(-1) == te.labels).float().mean())
        row = {}
        soft = get_soft_targets(args.backbone, key, cfg, tr, force=args.force)
        variants = [("ce only", 0, None)]
        variants.append(("ce+rlcd", cfg.rlcd_epochs, soft if soft is not None else "hard"))
        for variant, rlcd_epochs, soft_arg in variants:
            torch.manual_seed(cfg.seed)
            head = DecisionHead(**head_kwargs(cfg, tr))
            hist = train(
                head, tr, dv,
                soft_targets=None if soft_arg in (None, "hard") else soft_arg,
                warmup_epochs=cfg.warmup_epochs,
                rlcd_epochs=rlcd_epochs,
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
                # The literal 0 that used to sit here made every ablation arm run at seed 0 whatever
                # the config said, which is the defect this command exists to compare against.
                seed=cfg.seed,
                log=lambda *a, **k: None,
            )
            m = evaluate(
                head, te,
                temperature=hist.temperature_for(int(tr.qtype), int(tr.n_options)),
            )
            row[variant] = m
            print(
                f"{key:<17}{variant:<10}{m['accuracy']:>8.4f}{bayes:>8.4f}{m['ece']:>8.4f}"
                f"{m['ece_entropy']:>8.4f}{m['nll']:>8.4f}"
                f"{m.get('posterior_l1', float('nan')):>8.4f}"
            )
        d = row["ce+rlcd"]["ece"] - row["ce only"]["ece"]
        da = row["ce+rlcd"]["accuracy"] - row["ce only"]["accuracy"]
        results[key] = row
        print(f"{'':<17}{'delta':<10}{da:>+8.4f}{'':>8}{d:>+8.4f}")
    (RUNS / f"{args.backbone}_ablation.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )
