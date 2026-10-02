"""Sweep the head's width -- the 256 in Config was chosen, never measured.

The only measured part of that decision is the negative one: running the projection at the
backbone's width (768) would spend ~26M parameters fitting a few thousand examples, and a
per-option scorer at 256 reaches 0.79 where a set-attention stack at 768 collapses to chance.
Nobody swept the width itself.

Two tasks, deliberately at opposite ends of the data range: ag_news (4000 train rows) and
laya_category (225), three seeds each, full recipe (60 warm-up + 30 RLCD epochs), evaluated at
T=1.  Features are cached, so a cell costs ~15-40 s and no extraction happens.
"""
from __future__ import annotations

import statistics as st
import time

import torch

from rwkv_system1.cache_layout import head_kwargs, load_features
from rwkv_system1.decision import DecisionHead
from rwkv_system1.pipeline.config import Config
from rwkv_system1.train import evaluate, train

WIDTHS = (64, 128, 256, 512)
TASKS = (("ag_news", "n4000-500-1000"), ("laya_category", "n225-75-100"))
SEEDS = (0, 1, 2)

data = {}
for key, _ in TASKS:
    data[key] = {s: load_features("rwkv7-g1d-0.1b", key, s, Config())
                 for s in ("train", "dev", "test")}

print(f"{'task':<16}{'width':>6}{'params':>10}{'acc':>9}{'sd':>7}{'nll':>9}   seconds")
for key, _ in TASKS:
    for width in WIDTHS:
        cfg = Config()
        cfg.head_d_model = width
        accs, nlls, t0 = [], [], time.perf_counter()
        params = sum(p.numel() for p in DecisionHead(**head_kwargs(cfg, data[key]["train"])).parameters())
        for seed in SEEDS:
            torch.manual_seed(seed)
            head = DecisionHead(**head_kwargs(cfg, data[key]["train"]))
            train(head, data[key]["train"], data[key]["dev"],
                  soft_targets=data[key]["train"].true_probs,
                  warmup_epochs=cfg.warmup_epochs, rlcd_epochs=cfg.rlcd_epochs,
                  lr=cfg.lr, batch_size=cfg.batch_size, weight_decay=cfg.weight_decay,
                  group_size=cfg.group_size, pg_weight=cfg.pg_weight, w_sph=cfg.w_sph,
                  w_rps=cfg.w_rps, permute_options=cfg.permute_options, seed=seed,
                  log=lambda *a: None)
            m = evaluate(head, data[key]["test"])
            accs.append(m["accuracy"])
            nlls.append(m["nll"])
        sd = st.stdev(accs) if len(accs) > 1 else 0.0
        print(f"{key:<16}{width:>6}{params:>10,}{100 * st.mean(accs):>8.2f}%{100 * sd:>6.2f}"
              f"{st.mean(nlls):>9.4f}   {time.perf_counter() - t0:.0f}s", flush=True)
