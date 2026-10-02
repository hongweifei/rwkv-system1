"""Joint head (one head on all four corpora) vs four per-task heads, over 3 seeds.

Both arms read the same cached 0.1b features and share the recipe (warm-up CE -> RLCD on the
gold soft targets -> per-bucket temperature fit); only *how many heads* and *which rows each
head sees* differ.  Head init is seeded before construction so a seed fixes init as well as
shuffling.
"""

import json
import os
import statistics as st
import time

import torch

from rwkv_system1.cache_layout import head_kwargs, load_features
from rwkv_system1.pipeline.config import Config
from rwkv_system1.features import concat_joint
from rwkv_system1.decision import DecisionHead
from rwkv_system1.train import train, evaluate

BB = "rwkv7-g1d-0.1b"
TASKS = ["laya_urgency", "laya_category", "laya_needs_human", "ag_news"]
SEEDS = (0, 1, 2)
QUICK = bool(int(os.environ.get("QUICK", "0")))

cfg = Config()
if QUICK:
    cfg.warmup_epochs, cfg.rlcd_epochs = 4, 2

tr = {k: load_features(BB, k, "train", cfg) for k in TASKS}
dv = {k: load_features(BB, k, "dev", cfg) for k in TASKS}
te = {k: load_features(BB, k, "test", cfg) for k in TASKS}
JT, JD = concat_joint([(k, tr[k]) for k in TASKS]), concat_joint([(k, dv[k]) for k in TASKS])

print(f"joint  N={len(JT)} dev={len(JD)} Kmax={JT.features.shape[1]} D={JT.feature_dim} "
      f"soft rows={JT.soft_rows}/{len(JT)}")
print("split  " + "  ".join(f"{k}:{len(tr[k])}" for k in TASKS))
nj = sum(p.numel() for p in DecisionHead(**head_kwargs(cfg, JT)).parameters())
print(f"head params: joint {nj/1e6:.2f}M x1   split {nj/1e6:.2f}M x4")


def run(head, tfs, dfs, soft, seed):
    torch.manual_seed(seed)
    head = head()
    hist = train(head, tfs, dfs, soft_targets=soft, warmup_epochs=cfg.warmup_epochs,
                 rlcd_epochs=cfg.rlcd_epochs, lr=cfg.lr, batch_size=cfg.batch_size,
                 weight_decay=cfg.weight_decay, group_size=cfg.group_size,
                 pg_weight=cfg.pg_weight, w_sph=cfg.w_sph, w_rps=cfg.w_rps,
                 seed=seed, log=lambda *a: None)
    return head, hist


res = {name: {t: [] for t in TASKS} for name in ("joint", *TASKS)}
tabs = {}
for seed in SEEDS:
    for arm in ("joint", "split"):
        t0 = time.perf_counter()
        if arm == "joint":
            head, hist = run(lambda: DecisionHead(**head_kwargs(cfg, JT)), JT, JD,
                             JT.true_probs, seed)
            for k in TASKS:
                t = hist.temperature_for(tr[k].qtype, tr[k].n_options)
                tabs.setdefault(("joint", k), set()).add(t)
                res["joint"][k].append((evaluate(head, te[k]), evaluate(head, te[k], temperature=t)))
        else:
            for k in TASKS:
                head, hist = run(lambda: DecisionHead(**head_kwargs(cfg, tr[k])), tr[k], dv[k],
                                 tr[k].true_probs, seed)
                t = hist.temperature_for(tr[k].qtype, tr[k].n_options)
                tabs.setdefault((k, k), set()).add(t)
                res[k][k].append((evaluate(head, te[k]), evaluate(head, te[k], temperature=t)))
        print(f"  seed {seed} {arm}: {time.perf_counter()-t0:.0f}s", flush=True)


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def cells(rows):
    a, nl, e1, ef = ([r[0]["accuracy"] for r in rows], [r[0]["nll"] for r in rows],
                     [r[0]["ece"] for r in rows], [r[1]["ece"] for r in rows])
    return (f"{st.mean(a):>8.4f}±{sd(a):<6.4f}{st.mean(nl):>8.4f}±{sd(nl):<6.4f}"
            f"{st.mean(e1):>8.4f}±{sd(e1):<6.4f}{st.mean(ef):>8.4f}±{sd(ef):<6.4f}")


print(f"\n{'model':<16}{'task':<18}{'n':>5}{'acc':>16}{'NLL':>16}{'ECE @T=1':>16}{'ECE fit':>16}  Tfit")
for k in TASKS:
    tf_j = sorted(tabs.get(("joint", k), set()))
    tf_s = sorted(tabs.get((k, k), set()))
    print(f"{'joint':<16}{k:<18}{len(te[k]):>5}{cells(res['joint'][k])}  {tf_j}")
    print(f"{k:<16}{k:<18}{len(te[k]):>5}{cells(res[k][k])}  {tf_s}")
    dj = [r[0]["accuracy"] for r in res["joint"][k]]
    ds = [r[0]["accuracy"] for r in res[k][k]]
    ej = [r[0]["ece"] for r in res["joint"][k]]
    es = [r[0]["ece"] for r in res[k][k]]
    print(f"{'-> joint - split':<34}{'':>5}{st.mean(dj)-st.mean(ds):>+8.4f}"
          f"{'':>8}{st.mean(ej)-st.mean(es):>+8.4f}")
out = {"seeds": list(SEEDS), "quick": QUICK,
       "per_seed": {n: {k: [{"acc": r[0]["accuracy"], "ece": r[0]["ece"], "nll": r[0]["nll"],
                             "ece_fitted": r[1]["ece"], "t": r[1]["temperature"]}
                         for r in rows]
                        for k, rows in d.items()} for n, d in res.items()}}
path = "/tmp/joint_vs_split_results.json"
if QUICK:
    path += ".quick"
json.dump(out, open(path, "w"), indent=1)
print(f"\nwrote {path}")
