"""Validate the head/RLCD/temperature path without paying for backbone extraction.

Builds a well-posed synthetic decision task with an analytic posterior: exactly one
option carries a boost of size ``B * conf`` along a shared direction, the rest carry
noise, and the label is the index of the boosted option.  Given the observed scores the
posterior over "which option was boosted" is available in closed form, so both accuracy
and calibration can be scored against ground truth.

``conf`` controls difficulty per example: ``conf=1`` is a clear outlier, ``conf=0``
leaves all options exchangeable and the honest answer is 1/K.  A model that reports 0.99
on the low-confidence half is exactly the failure mode RLCD is supposed to fix.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rwkv_system1.decision import QTYPES, DecisionHead
from rwkv_system1.features import FeatureSet
from rwkv_system1.train import evaluate, train

BOOST = 3.0


def log_phi(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * x * x - 0.5 * torch.log(torch.tensor(2 * torch.pi))


def synth(n: int, k: int, d: int, seed: int) -> FeatureSet:
    g = torch.Generator().manual_seed(seed)
    conf = torch.rand(n, generator=g)
    labels = torch.randint(0, k, (n,), generator=g)
    scores = torch.randn(n, k, generator=g) + BOOST * conf[:, None] * (
        torch.arange(k) == labels[:, None]
    ).float()

    u = torch.ones(d)
    u = u / u.norm()
    feats = (scores[:, :, None] * u + 0.5 * torch.randn(n, k, d, generator=g)).to(torch.float16)

    # P(k is the boosted option | scores): the true option has mean BOOST*conf, the
    # others mean 0.
    shifted = scores[:, :, None] - BOOST * conf[:, None, None] * torch.eye(k)
    loglik = log_phi(shifted).sum(-1)
    true_probs = torch.softmax(loglik, dim=-1)

    return FeatureSet(
        features=feats,
        labels=labels,
        n_options=k,
        qtype=QTYPES["choice"],
        true_probs=true_probs,
        states=[""] * n,
        options=[f"opt{i}" for i in range(k)],
        qtype_name="choice",
    )


def main() -> int:
    k, d = 4, 1024
    tr, dv, te = synth(1200, k, d, 1), synth(300, k, d, 2), synth(300, k, d, 3)

    # How well could anything do? An oracle that knows the posterior picks its argmax.
    bayes = float(tr.true_probs.max(-1).values.mean())
    pair = float((tr.true_probs * (tr.true_probs > 0)).sum(-1).pow(2).sum(-1).mean())
    print(f"oracle (argmax of the true posterior) accuracy on train = {bayes:.4f}")
    print(f"trivial uniform-probability baseline ECE            = "
          f"{float((torch.full((len(tr), k), 1 / k).max(-1).values - (tr.true_probs.argmax(-1) == tr.labels).float()).abs().mean()):.4f}")

    head = DecisionHead(in_dim=d, d_model=256, n_layers=2, dropout=0.1)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"head: {n_params / 1e6:.2f}M params\n")

    t0 = time.perf_counter()
    hist = train(head, tr, dv, warmup_epochs=40, rlcd_epochs=25, batch_size=64, log=print)
    wall = time.perf_counter() - t0

    print(f"\nwall {wall:.1f}s   fitted T={hist.temperature:.3f}  bucket={hist.bucket}")
    print(f"\n{'':<12}{'acc':>8}{'Bayes':>8}{'ECE':>8}{'ECEraw':>8}{'NLL':>8}{'postL1':>8}")
    rows = [
        ("warm-up end", hist.warmup[-1]["dev_acc"], hist.warmup[-1]["dev_ece"], None),
        ("RLCD end", hist.rlcd[-1]["dev_acc"], hist.rlcd[-1]["dev_ece"], None),
    ]
    raw = evaluate(head, dv, 1.0)
    fit = evaluate(head, dv, hist.temperature)
    print(f"{'T=1':<12}{raw['accuracy']:>8.4f}{bayes:>8.4f}{raw['ece']:>8.4f}"
          f"{raw['ece']:>8.4f}{raw['nll']:>8.4f}{raw['posterior_l1']:>8.4f}")
    print(f"{'T=fitted':<12}{fit['accuracy']:>8.4f}{bayes:>8.4f}{fit['ece']:>8.4f}"
          f"{raw['ece']:>8.4f}{fit['nll']:>8.4f}{fit['posterior_l1']:>8.4f}")

    print("\nconfidence split on dev (does it admit when it cannot tell?):")
    from rwkv_system1.rlcd import reliability_curve
    from rwkv_system1.train import head_logits

    probs = head.probabilities(head_logits(head, dv), hist.temperature)
    hard = dv.true_probs.max(-1).values > 0.9
    for name, sel in (("clear  (true p>0.9)", hard), ("ambiguous", ~hard)):
        if bool(sel.any()):
            print(f"  {name:<22} n={int(sel.sum()):>4}  acc {float((probs[sel].argmax(-1) == dv.labels[sel]).float().mean()):.3f}"
                  f"  mean reported p {float(probs[sel].max(-1).values.mean()):.3f}"
                  f"  true p {float(dv.true_probs[sel].max(-1).values.mean()):.3f}")

    print("\nreliability curve on dev (reported top-p bin -> empirical accuracy):")
    correct = (probs.argmax(-1) == dv.labels).float()
    for conf, acc, n in reliability_curve(probs, dv.labels, n_bins=8):
        print(f"  conf {conf:.3f}  acc {acc:.3f}  n={n:>4}  gap {conf - acc:+.3f}")
    print(f"  overall: mean conf {float(probs.max(-1).values.mean()):.4f}"
          f"  mean acc {float(correct.mean()):.4f}  |gap| {abs(float(probs.max(-1).values.mean()) - float(correct.mean())):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
