"""Which layout makes the head read the option text: augmentation (B) or a per-option suffix
pass (A)?  Same rows, same head, same `train()` path -- only the provenance of the marker
vectors and the option order the training rows use differ.

arms
  shared      the shipped design: every option in one causal suffix pass, one order
  shared+aug  same features, but the train split is replicated P times with a fresh option
              order per replica (B).  Dev and test stay in the canonical order.
  per-option  each option gets its own short suffix pass off the shared prefill, so a marker
              can only see its own option (A).  Re-extracted for exactly the rows the others use.

Then the probe that refuted permutation invariance: rotate the option columns, map the answer
back through the permutation, and ask whether it followed the text, the slot, or accuracy at all.
"""
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from rwkv_system1 import datasets
from rwkv_system1.cache_layout import checkpoint, head_kwargs, load_features
from rwkv_system1.pipeline.config import Config
from rwkv_system1.backbone import DecisionBackbone
from rwkv_system1.decision import DecisionHead
from rwkv_system1.features import FeatureSet
from rwkv_system1.rwkv7 import RWKV7
from rwkv_system1.train import train, evaluate, head_logits

BB, KEY = "rwkv7-g1d-0.1b", "ag_news"
N_TRAIN, N_DEV, N_TEST, REPLICAS = 400, 100, 64, 4
cfg = Config()

tr_full = load_features(BB, KEY, "train", cfg)
dv_full = load_features(BB, KEY, "dev", cfg)
te_full = load_features(BB, KEY, "test", cfg)
def rows(fs, n):
    m = torch.zeros(len(fs), dtype=torch.bool)
    m[:n] = True
    return fs.subset(m)


TR = rows(tr_full, N_TRAIN)
DV = rows(dv_full, N_DEV)
TE = rows(te_full, N_TEST)
K = TR.n_options
INSTRUCTIONS = datasets.load_ag_news(n_train=0, n_dev=0, n_test=8)["test"][0].question.instructions
print(f"rows: train {len(TR)} dev {len(DV)} test {len(TE)}  k={K}  options={TR.options}")


def extract_per_option(backbone, fs):
    """(N, k, D) where slot o's vector came from a pass containing only option o."""
    out = torch.zeros(len(fs), K, fs.feature_dim, dtype=torch.float16)
    blocks = []
    for o in range(K):
        suffix, markers = backbone.encode_query(fs.qtype_name, INSTRUCTIONS, [fs.options[o]])
        blocks.append((suffix, markers[-1]))
    ids = [backbone.encode_state(s) for s in fs.states]
    buckets = {}
    for i, t in enumerate(ids):
        buckets.setdefault(int(t.shape[1]), []).append(i)
    for length, idxs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        for start in range(0, len(idxs), 8):
            chunk = idxs[start:start + 8]
            pre = torch.cat([ids[i] for i in chunk])
            _, _, carried = backbone.model.forward(pre, return_hidden=False)
            for o, (suffix, pos) in enumerate(blocks):
                hidden, _, _ = backbone.model.forward(
                    suffix.repeat(len(chunk), 1), RWKV7.clone_state(carried), return_hidden=True
                )
                for r, i in enumerate(chunk):
                    out[i, o] = hidden[r, pos]
    return out


def stack(feats, src, labels=None):
    return FeatureSet(features=feats, labels=src.labels if labels is None else labels,
                      n_options=K, qtype=src.qtype,
                      true_probs=src.true_probs, states=list(src.states),
                      options=src.options, qtype_name=src.qtype_name)


def augmented(src, seed):
    """P replicas of the train split, each with its own option column order."""
    gen = torch.Generator().manual_seed(seed)
    feats, labels = [src.features], [src.labels]
    for _ in range(REPLICAS - 1):
        perm = torch.stack([torch.randperm(K, generator=gen) for _ in range(len(src))])
        feats.append(src.features.gather(1, perm.unsqueeze(-1).expand(-1, -1, src.feature_dim)))
        labels.append((perm == src.labels.unsqueeze(1)).float().argmax(1))
    return stack(torch.cat(feats), src, torch.cat(labels))


PO_CACHE = "D:/tmp/po_feats_ag_news.pt"
if os.path.exists(PO_CACHE):
    print("loaded the per-option features from disk (no re-extraction)")
    po_tr, po_dv, po_te = (t := torch.load(PO_CACHE, map_location="cpu", weights_only=True)).values()
else:
    backbone = DecisionBackbone(checkpoint(BB),
                                max_suffix_tokens=cfg.max_suffix_tokens)
    t0 = time.perf_counter()
    print("re-extracting these rows under the per-option layout ...", flush=True)
    po_tr = extract_per_option(backbone, TR)
    po_dv = extract_per_option(backbone, DV)
    po_te = extract_per_option(backbone, TE)
    torch.save({"train": po_tr, "dev": po_dv, "test": po_te}, PO_CACHE)
    print(f"  done in {time.perf_counter() - t0:.0f}s")
    del backbone

PO, PO_D, PO_T = stack(po_tr, TR), stack(po_dv, DV), stack(po_te, TE)
TRA = augmented(TR, 0)
print(f"augmented train: {len(TRA)} rows ({REPLICAS} orders), "
      f"gold-slot balance {[int((TRA.labels == j).sum()) for j in range(K)]}")


def train_plain(tr, dv, epochs=60, seed=0):
    """Cross-entropy only, fixed epoch count, no dev-NLL restore.

    ``train()`` keeps the best-dev-NLL epoch, and at this data size that is epoch 1 -- the
    head's confidence grows from the first step, so the rule that exists to stop us reporting
    an overfit final epoch instead reports a collapsed one.  For a layout comparison we need
    the same fixed budget in every arm.
    """
    from rwkv_system1.rlcd import cross_entropy_loss
    torch.manual_seed(seed)
    head = DecisionHead(**head_kwargs(cfg, tr))
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    qt = torch.full((len(tr),), tr.qtype, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for ep in range(1, epochs + 1):
        head.train()
        for xb, yb in DataLoader(TensorDataset(tr.features.float(), tr.labels), batch_size=cfg.batch_size,
                                 shuffle=True, generator=gen):
            loss = cross_entropy_loss(head(xb, qt[: len(yb)]), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in (1, 5, 10, 20, 40, epochs):
            head.eval()
            with torch.no_grad():
                dv_acc = float((head_logits(head, dv).argmax(-1) == dv.labels).float().mean())
            curve.append((ep, round(dv_acc, 3)))
    head.eval()
    return head, curve


def train_arm(tr, dv, seed=0):
    torch.manual_seed(seed)
    head = DecisionHead(**head_kwargs(cfg, tr))
    dv2 = FeatureSet(features=dv.features, labels=dv.labels, n_options=K, qtype=dv.qtype,
                     true_probs=dv.true_probs, states=list(dv.states), options=dv.options,
                     qtype_name=dv.qtype_name)
    hist = train(head, tr, dv2, warmup_epochs=cfg.warmup_epochs, rlcd_epochs=cfg.rlcd_epochs,
                 lr=cfg.lr, batch_size=cfg.batch_size, soft_targets=tr.true_probs, seed=seed,
                 log=lambda *a: None)
    return head, hist


def relabel(labels, order):
    """Gold index expressed in the rotated column layout."""
    inv = torch.empty_like(order)
    inv[order] = torch.arange(len(order))
    return inv[labels]


def probe(head, te):
    base = head_logits(head, te).argmax(-1)
    out = {"acc": float((base == te.labels).float().mean()) * 100}
    for name, order in (("rot2", torch.tensor([(i + 2) % K for i in range(K)])),
                        ("rev", torch.tensor(list(reversed(range(K)))))):
        fs2 = FeatureSet(features=te.features[:, order], labels=relabel(te.labels, order),
                         n_options=K, qtype=te.qtype, true_probs=None, states=list(te.states),
                         options=te.options, qtype_name=te.qtype_name)
        p = head_logits(head, fs2).argmax(-1)
        back = order[p]                       # the answer, in the original option index
        out[name] = (
            float((back == base).float().mean()) * 100,           # follows the TEXT
            float((p == base).float().mean()) * 100,              # follows the SLOT
            float((p == fs2.labels).float().mean()) * 100,        # accuracy, same rows
        )
    return out


ARMS = [("shared", TR, DV, TE), ("shared+aug", TRA, DV, TE), ("per-option", PO, PO_D, PO_T)]
counts = [int((TE.labels == j).sum()) for j in range(K)]
print(f"\ntest label distribution {counts} of {len(TE)} -> always-answer-the-majority gets "
      f"{100 * max(counts) / len(TE):.1f}%")
print(f"{'arm':<12}{'train_acc':>10}{'best_ep':>9}{'dev acc @ ep1/¼/½/end':>26}"
      f"{'preds/slot':>16}   {'rotate by 2':<22}  {'reversed':<22}")
for name, tr, dv, te in ARMS:
    head, curve = train_plain(tr, dv)
    fits = float((head_logits(head, tr).argmax(-1) == tr.labels).float().mean()) * 100
    pred = head_logits(head, te).argmax(-1)
    r = probe(head, te)
    print(f"{name:<12} train_acc {fits:>5.1f}%  preds/slot {str([int((pred == j).sum()) for j in range(K)]):>14}"
          f"  dev curve {curve}", flush=True)
    print(f"{'':<12} acc {r['acc']:>5.1f}%   rotate-by-2: text {r['rot2'][0]:>5.1f} slot {r['rot2'][1]:>5.1f} "
          f"acc {r['rot2'][2]:>5.1f}% (Δ{r['rot2'][2] - r['acc']:+5.1f})   reversed: text {r['rev'][0]:>5.1f} "
          f"slot {r['rev'][1]:>5.1f} acc {r['rev'][2]:>5.1f}% (Δ{r['rev'][2] - r['acc']:+5.1f})", flush=True)
