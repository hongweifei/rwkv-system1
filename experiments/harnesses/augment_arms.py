"""Three-seed replication of the option-order augmentation, plus the slot-embedding control.

Arms, all through the real `train()` path (60 CE epochs, 30 RLCD epochs, per-bucket temperature
fit) so the numbers sit on the same footing as the shipped baseline table:

  slots          the shipped configuration -- slot embeddings, canonical order only
  slots+perm     B: same head, options handed to it in a random slot each step
  noslots        slot embeddings removed, canonical order only
  noslots+perm   B with the slot embeddings gone as well

Two tasks, chosen to cover the failure modes: `ag_news` (4-way choice, hard labels) and
`laya_urgency` (4-level score with teacher soft targets, so the RPS term is live and the
permutation has to preserve its level ordering).
"""
import json
import statistics as st
import time
from pathlib import Path

import torch

from rwkv_system1.cache_layout import head_kwargs, load_features
from rwkv_system1.pipeline.config import Config
from rwkv_system1.decision import DecisionHead
from rwkv_system1.features import FeatureSet
from rwkv_system1.train import train, evaluate, head_logits

BB = "rwkv7-g1d-0.1b"
TASKS = ["ag_news", "laya_urgency"]
SEEDS = (0, 1, 2)
ARMS = {"slots": (True, False), "slots+perm": (True, True),
        "noslots": (False, False), "noslots+perm": (False, True)}
N_PROBE = 400

data = {t: {s: load_features(BB, t, s, Config()) for s in ("train", "dev", "test")}
        for t in TASKS}


def rotated(fs, order):
    o = torch.tensor(order)
    inv = torch.empty_like(o)
    inv[o] = torch.arange(len(o))
    return FeatureSet(features=fs.features[:, o], labels=inv[fs.labels], n_options=fs.n_options,
                      qtype=fs.qtype, true_probs=None, states=list(fs.states),
                      options=fs.options, qtype_name=fs.qtype_name)


def probe(head, fs):
    """Re-order the options, then ask what the answer tracked and what accuracy survived."""
    k = fs.n_options
    order = [(i + 2) % k for i in range(k)]
    m = torch.zeros(len(fs), dtype=torch.bool)
    m[:min(N_PROBE, len(fs))] = True
    fs = fs.subset(m)
    base = head_logits(head, fs).argmax(-1)
    fs2 = rotated(fs, order)
    p = head_logits(head, fs2).argmax(-1)
    acc0 = float((base == fs.labels).float().mean()) * 100
    acc2 = float((p == fs2.labels).float().mean()) * 100
    return (acc0, acc2, float((torch.tensor(order)[p] == base).float().mean()) * 100)


res = {}
for arm, (slots, perm) in ARMS.items():
    cfg = Config()
    cfg.head_slot_embeddings = slots
    cfg.permute_options = perm
    for task in TASKS:
        rows = []
        t0 = time.perf_counter()
        for seed in SEEDS:
            torch.manual_seed(seed)
            head = DecisionHead(**head_kwargs(cfg, data[task]["train"]))
            hist = train(head, data[task]["train"], data[task]["dev"],
                         soft_targets=data[task]["train"].true_probs,
                         warmup_epochs=cfg.warmup_epochs, rlcd_epochs=cfg.rlcd_epochs, lr=cfg.lr,
                         batch_size=cfg.batch_size, weight_decay=cfg.weight_decay,
                         group_size=cfg.group_size, pg_weight=cfg.pg_weight, w_sph=cfg.w_sph,
                         w_rps=cfg.w_rps, permute_options=perm, seed=seed, log=lambda *a: None)
            te = data[task]["test"]
            m = evaluate(head, te)
            rows.append((m["accuracy"] * 100, m["ece"] * 100, m["nll"], *probe(head, te),
                         list(hist.temperatures.values())[0] if hist.temperatures else 1.0))
        res[(arm, task)] = rows
        cols = list(zip(*rows))
        mean = [st.mean(c) for c in cols]
        sd = [st.stdev(c) if len(c) > 1 else 0.0 for c in cols]
        acc, ece, nll, p0, p2, tx, tf = mean
        print(f"{arm:<14}{task:<16} acc {acc:>6.2f}±{sd[0]:.2f}  ECE {ece:>5.2f}±{sd[1]:.2f}"
              f"  NLL {nll:.4f}±{sd[2]:.4f}  | reorder: acc {p2:.1f}% (Δ{p2 - p0:+.1f})"
              f"  text {tx:.1f}%  | T {tf:.2f}  [{time.perf_counter() - t0:.0f}s]", flush=True)

# JSON beside the other results, path derived from this file: it used to `torch.save` to a
# hard-coded `D:/tmp/...`, which no other machine has and which turned a re-run into a silent
# write-somewhere-else.
dest = Path(__file__).resolve().parents[1] / "results" / "augment_arms.json"
dest.write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1), encoding="utf-8")
print(f"\nwrote {dest}")
