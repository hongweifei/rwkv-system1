"""Does option-order augmentation survive at full data size, and what does it cost?

Two arms, full ag_news splits (4000 train / 500 dev / 1000 test), identical fixed budget of
60 cross-entropy epochs, no dev-NLL restore (that rule picks epoch 1 at small data and would
confound the comparison).  `shared` trains on the canonical option order, i.e. the shipped
design; `shared+aug` sees each row in 4 random option orders.

Reported per arm: test accuracy and ECE at T=1, then the rotation probe on 250 test rows --
whether the answer follows the option text or the slot, and how much accuracy is lost when the
option list is re-ordered.
"""
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from rwkv_system1.cache_layout import head_kwargs, load_features
from rwkv_system1.pipeline.config import Config
from rwkv_system1.decision import DecisionHead
from rwkv_system1.features import FeatureSet
from rwkv_system1.rlcd import cross_entropy_loss, expected_calibration_error
from rwkv_system1.train import head_logits, evaluate

cfg = Config()
BB, KEY, REPLICAS, EPOCHS = "rwkv7-g1d-0.1b", "ag_news", 4, 60
TR = load_features(BB, KEY, "train", cfg)
DV = load_features(BB, KEY, "dev", cfg)
TE = load_features(BB, KEY, "test", cfg)
K = TR.n_options
print(f"train {len(TR)} dev {len(DV)} test {len(TE)}  k={K}  options={TR.options}")


def orders(src, seed):
    """The canonical order plus REPLICAS-1 fresh per-row permutations."""
    feats, labels = [src.features], [src.labels]
    gen = torch.Generator().manual_seed(seed)
    for _ in range(REPLICAS - 1):
        perm = torch.stack([torch.randperm(K, generator=gen) for _ in range(len(src))])
        feats.append(src.features.gather(1, perm.unsqueeze(-1).expand(-1, -1, src.feature_dim)))
        labels.append((perm == src.labels.unsqueeze(1)).float().argmax(1))
    return torch.cat(feats), torch.cat(labels)


def fit(tr_feats, tr_labels, seed=0):
    torch.manual_seed(seed)
    head = DecisionHead(**head_kwargs(cfg, TR))
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    qt = torch.full((cfg.batch_size,), TR.qtype, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    n = tr_feats.shape[0]
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, cfg.batch_size):
            sel = perm[s:s + cfg.batch_size]
            xb = tr_feats[sel].float()
            yb = tr_labels[sel]
            loss = cross_entropy_loss(head(xb, qt[: len(yb)]), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 15 == 0:
            print(f"    epoch {ep:>3}/{EPOCHS}  {time.perf_counter() - t0:.0f}s", flush=True)
    head.eval()
    return head


def probe(head, n=250):
    m = torch.zeros(len(TE), dtype=torch.bool); m[:n] = True
    te = TE.subset(m)
    base = head_logits(head, te).argmax(-1)
    out = {}
    for name, order in (("rot2", [(i + 2) % K for i in range(K)]),
                        ("rev", list(reversed(range(K))))):
        o = torch.tensor(order)
        inv = torch.empty_like(o); inv[o] = torch.arange(len(o))
        fs2 = FeatureSet(features=te.features[:, o], labels=inv[te.labels], n_options=K,
                         qtype=te.qtype, true_probs=None, states=list(te.states),
                         options=te.options, qtype_name=te.qtype_name)
        p = head_logits(head, fs2).argmax(-1)
        out[name] = (float((o[p] == base).float().mean()) * 100,
                     float((p == base).float().mean()) * 100,
                     float((p == fs2.labels).float().mean()) * 100)
    return out


for name, build in (("shared", lambda: (TR.features, TR.labels)),
                    ("shared+aug", lambda: orders(TR, 0))):
    feats, labels = build()
    head = fit(feats, labels)
    m = evaluate(head, TE)
    tr_acc = float((head_logits(head, TR).argmax(-1) == TR.labels).float().mean()) * 100
    r = probe(head)
    print(f"\n{name:<12} rows {len(feats):>6}  train {tr_acc:.1f}%  test acc {m['accuracy'] * 100:.2f}%"
          f"  ECE@T=1 {m['ece']:.4f}  NLL {m['nll']:.4f}")
    print(f"{'':<12} rotate by 2: follows text {r['rot2'][0]:>5.1f}%  slot {r['rot2'][1]:>5.1f}%  "
          f"acc {r['rot2'][2]:>5.1f}%  (Δ {r['rot2'][2] - m['accuracy'] * 100:+5.1f} pt)")
    print(f"{'':<12} reversed   : follows text {r['rev'][0]:>5.1f}%  slot {r['rev'][1]:>5.1f}%  "
          f"acc {r['rev'][2]:>5.1f}%  (Δ {r['rev'][2] - m['accuracy'] * 100:+5.1f} pt)", flush=True)
