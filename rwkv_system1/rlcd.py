"""RLCD -- the reward, the losses, temperature fitting, and the metrics.

``proper_reward`` is transcribed from Laya's ``laya/common.py``.  It is a *reward*, higher
is better, and it is **strictly proper**: its expectation is maximised exactly when the
reported distribution equals the true posterior.  It adds up three rules, each proper on
its own, because they fail in different ways:

* **log score** ``target . log q`` -- sharp, and unbounded, so one confidently wrong
  answer on an ambiguous input is punished hard.  Floored at ``log(1e-4)``.
* **spherical score** ``(target . q) / ||q||`` -- bounded, so it tolerates the label noise
  that makes raw log score push predictions onto vertices.
* **ranked probability score** -- applied to ``score`` questions only, because levels are
  *ordered*: predicting level 2 when the truth is level 1 is not the same mistake as
  predicting level 5, and a categorical rule cannot tell those apart.

MEASURED 2026-09-23: three tasks x three seeds x six recipes, 54 runs, same features and same
head throughout (``experiments/rlcd_variants.py``, rows in ``experiments/results/rlcd_variants.json``).
Paired per-seed deltas against the shipped recipe, at T=1:

    task          arm                      d_acc          d_ECE         d_postL1
    laya_urgency  no RLCD stage          -4.08±0.22pt   -0.18±0.76pt   +4.19±0.44pt
    laya_urgency  cross-entropy, 90 ep   -2.08±0.73pt   +1.62±0.18pt   +4.80±0.80pt
    laya_urgency  PG reward = the score   +0.08±0.42pt   +1.49±0.17pt   -0.09±0.02pt
    laya_urgency  Gaussian noise, s=1     -0.50±0.38pt   +0.35±0.37pt   -0.12±0.14pt
    laya_urgency  Gaussian noise, s=1.28  +0.00±0.14pt   +0.17±0.19pt   +0.04±0.01pt
    banking77     no RLCD stage         -10.93±2.28pt   +0.53±3.16pt        n/a
    banking77     cross-entropy, 90 ep   -4.27±1.23pt   -0.47±0.74pt        n/a
    banking77     PG reward = the score  -0.47±0.07pt   +0.50±0.81pt        n/a
    banking77     Gaussian noise, s=1    -0.40±0.20pt   +0.61±0.49pt        n/a
    banking77     Gaussian noise, s=1.28 -0.40±0.31pt   +0.53±0.37pt        n/a
    ag_news       no RLCD stage          +0.37±0.37pt   -2.28±0.83pt        n/a
    ag_news       cross-entropy, 90 ep   +0.37±0.37pt   -2.28±0.83pt        n/a
    ag_news       PG reward = the score  -0.13±0.09pt   -0.31±0.06pt        n/a
    ag_news       Gaussian noise, s=1    -0.03±0.12pt   -0.01±0.13pt        n/a
    ag_news       Gaussian noise, s=1.28 +0.00±0.06pt   +0.08±0.08pt        n/a

So the stage does earn its place -- 4.1pt on the ordinal task and 10.9pt on the 77-label one --
but a large part of that is simply thirty more epochs: an equal-epochs cross-entropy control
recovers 2.1 of the 4.1pt and 4.3 of the 10.9pt.  On ``ag_news`` it buys nothing and costs
calibration (+0.37±0.37pt accuracy, not resolved, but +2.28±0.83pt top-label ECE at T=1), and the
control is *bit-identical* to plain cross-entropy there because the best-dev-NLL restore lands on
the same epoch either way.  That is reason 1 below made visible: under hard labels the log-score
term is cross-entropy, so the only thing left for the stage to add is the spherical term.

The two variants the RLCD write-up describes are both indistinguishable from what ships:

* **"the reward is a strictly proper scoring rule"** -- used as the *policy-gradient* reward this
  is exactly ``pg_weight=0``, not approximately.  The proper score is a function of the reported
  distribution and the label, so it takes one value for every draw ``a``, the group-mean advantage
  is therefore identically zero, and the term contributes no gradient at all.  Measured, it lands
  within +0.08±0.42pt (laya_urgency) and -0.47±0.07pt (banking77) of the shipped verifier reward.
  The verifier ``1[a == y]`` is the only reward here that varies with the draw, which is what makes
  it the one worth using.
* **"exploration adds zero-mean Gaussian noise to the logits"** -- sampling the reported
  distribution already *is* ``argmax(logits + Gumbel)``, so this swaps the noise law rather than the
  objective.  Variance-matched to Gumbel (sigma = pi/sqrt(6) = 1.2825) the difference is
  0.00±0.14pt; at sigma = 1.0 it is -0.50±0.38pt.  No resolvable effect either way.

The shipped PG term's one small real effect is calibration on the ordinal task: switching it off
costs 1.49±0.17pt of top-label ECE at T=1, and 1.20pt after the temperature fit.  It is a mild
sharpening force -- maximising ``q_y`` pushes toward a point mass -- which helps here precisely
because this head is under-confident at T=1 (the fitted temperature lands at 0.60-0.70).  It is not
free: it costs 0.47±0.07pt accuracy on banking77 and 0.13±0.09pt on ag_news.

Two reasons the stage is bounded, both structural rather than fixable by tuning:

1. Against a one-hot label the log-score term *is* cross-entropy, so the reward has no
   calibration signal to add.  This is why the warm-up and RLCD stages look so similar.
2. Given a soft target instead, a proper rule is *faithful*, not corrective: it reproduces
   whatever distribution you hand it, miscalibration included.  Cross-fitted targets from a
   960-example fold model regress to the mean -- 0.884 confidence where the analytic
   posterior is 0.969, and 0.595 where it is 0.536 -- and the student inherits exactly that
   shape.  A temperature fit beats it because it is fitted against real labels, which are
   coarse but unbiased.

An earlier version of this block reported the opposite -- "a wash", 0.8040 vs 0.8140, on a 500-row
ticket-routing split with a Bayes ceiling of 0.851.  That split is gone (the current
``laya_urgency`` cache is 900/300/400 with a ceiling of 0.566) and the conclusion did not survive
the change, so it is replaced rather than kept beside the new one.

Most of the calibration in the headline numbers still comes from ``fit_temperature``, not from
RLCD: on laya_urgency the raw ECE is 0.0678 and the fitted 0.0379.  Laya's own 0.081 ECE is
likewise reported *after* temperature fitting.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .decision import QTYPES, clamp_temperature, temp_bucket


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring-rule reward. ``q``/``target`` are ``(B, K)``; higher is better.

    Transcribed from Laya's ``laya/common.py`` so the objective is the one they actually
    train on, not a reconstruction from a blog post.
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph

    is_score = (qtype == QTYPES["score"]).float()
    if bool(is_score.any()):
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def _mask_for(n_options: int, n_active: int, like: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(n_options, dtype=like.dtype, device=like.device)
    m[:n_active] = 1.0
    return m


def rlcd_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    qtype: int | torch.Tensor,
    target: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    group_size: int = 8,
    pg_weight: float = 0.2,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    exploration: str = "multinomial",
    explore_sigma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns ``(loss, parts)``.  ``logits`` is ``(B, K)`` and already temperature-free.

    ``qtype`` is one int for a single-task batch or a ``(B,)`` vector for a joint batch.
    ``pad_mask`` is ``(B, K)`` with **True on padding** -- the convention ``DecisionHead``
    uses, and the opposite of ``proper_reward``'s 1-for-real ``mask``, so it is inverted here
    once rather than at the call site. A joint batch always has padding, because the questions
    disagree on how many options they own; without the inversion a 2-option row is scored as if
    it owned five and its reported probabilities sum to zero.

    ``target`` is the distribution the report is scored against.  When it is omitted the
    one-hot label is used, and a strictly proper rule then has its empirical optimum at a
    point mass -- the reward is maximised by memorising each training label, so there is
    nothing to calibrate and RLCD degenerates into cross-entropy with extra steps.  Passing
    a *soft* target (cross-fitted predictions, a teacher's distribution) is what gives the
    proper rule something honest to converge to.
    """
    b, k = logits.shape
    keep = torch.ones(b, k, dtype=torch.float32) if pad_mask is None else (~pad_mask).float()
    qtype_t = (
        torch.as_tensor(qtype, dtype=torch.long)
        if torch.is_tensor(qtype)
        else torch.full((b,), int(qtype), dtype=torch.long)
    )

    probs = torch.softmax(logits / temperature, dim=-1) * keep
    if target is None:
        target = F.one_hot(labels, k).to(probs.dtype)
    else:
        target = target.to(probs.dtype) * keep
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)

    # (a) calibration: maximise the proper reward of the reported distribution.
    reward_direct = proper_reward(probs, target, qtype_t, keep, w_sph, w_rps)
    loss_proper = -reward_direct.mean()

    # (b) policy gradient: sample answers, verifier reward 1[a == y], group-mean baseline.
    #
    # ``exploration`` picks how the action is drawn.  ``multinomial`` samples the reported
    # distribution, which is exactly ``argmax(logits + Gumbel)``; ``gaussian`` takes
    # ``argmax(logits + sigma * N(0, 1))`` instead -- the probit exploration the RLCD write-up
    # describes.  Both are unbiased REINFORCE for the same expected reward, so the choice moves
    # which actions get tried and the variance of the estimate, not what is being maximised.
    # The probit policy's own score function has no closed form, so ``logp`` below stays the
    # softmax one either way; the sample is the whole content of the difference.
    with torch.no_grad():
        if exploration == "multinomial":
            draws = torch.multinomial(probs, group_size, replacement=True)
        elif exploration == "gaussian":
            noisy = (logits / temperature).unsqueeze(1) + explore_sigma * torch.randn(
                b, group_size, k, device=logits.device, dtype=logits.dtype
            )
            draws = noisy.argmax(dim=-1)
        else:
            raise SystemExit(f"unknown exploration {exploration!r}; multinomial or gaussian")
        reward = (draws == labels.unsqueeze(1)).to(logits.dtype)
        advantage = reward - reward.mean(dim=1, keepdim=True)
    logp = F.log_softmax(logits / temperature, dim=-1)
    loss_pg = -(logp.gather(1, draws) * advantage).mean()

    total = loss_proper + pg_weight * loss_pg
    return total, {
        "loss": float(total.detach()),
        "proper": float(loss_proper.detach()),
        "reward": float(reward_direct.mean().detach()),
        "pg": float(loss_pg.detach()),
        "verifier": float(reward.mean()),
    }


def cross_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    return F.cross_entropy(logits / temperature, labels)


def soft_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """``-sum(target * log_softmax(logits))`` -- cross-entropy against a distribution.

    The distillation objective, and the same quantity as :func:`cross_entropy_loss` when the target
    is one-hot.  Separate from ``proper_reward``'s log term on purpose: that one takes probabilities,
    clamps them and applies a floor, so it is a different number even before the spherical and RPS
    rules are added.
    """
    return -(target * F.log_softmax(logits / temperature, dim=-1)).sum(-1).mean()


# --------------------------------------------------------------------- temperature


@torch.no_grad()
def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    qtype: int,
    grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    min_gain_se: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Pick a temperature by dev NLL under a one-standard-error rule.

    NLL rather than ECE, because it is smooth, strictly proper and independent of bin edges
    -- this is Laya's "ECE after temperature fitting" step, one scalar per
    ``(qtype, option-count band)`` bucket.

    But the grid's argmin is *not* a usable answer on a held-out set this small.  Measured
    here: dev sets of 75 and 300 rows returned T=0.60 and T=0.70 (~1.5x sharpening), and the
    fitted value then made test ECE **worse** (category 0.0895 -> 0.1258, urgency
    0.0886 -> 0.0973).  With a 12-point grid some point always beats T=1 by luck, so
    a rescaling has to clear its own noise: among the grid points that are not significantly
    worse than the best one (paired per-example NLL difference within ``min_gain_se``
    standard errors), keep the one closest to 1.0.  Sub-noise evidence therefore yields
    T=1.0 -- "these rows cannot resolve a rescaling" -- instead of a number that only looks
    fitted because it was chosen on the rows it is scored on.
    """
    n = logits.shape[0]
    if n < 2:
        return 1.0, {"argmin_t": 1.0, "argmin_nll": float("nan"), "dev_nll": float("nan")}

    loss = {float(t): F.cross_entropy(logits / t, labels, reduction="none") for t in grid}
    means = {t: float(v.mean()) for t, v in loss.items()}
    t_star = min(means, key=means.get)
    for t in sorted(means, key=lambda x: abs(x - 1.0)):  # least rescaling first
        diff = loss[t] - loss[t_star]
        se = float(diff.std(unbiased=True)) / math.sqrt(n)
        if float(diff.mean()) <= min_gain_se * se:
            best = t
            break
    return clamp_temperature(best), {
        "argmin_t": t_star,
        "argmin_nll": means[t_star],
        "dev_nll": means[best],
    }


# ------------------------------------------------------------------------- metrics


@torch.no_grad()
def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return float((probs.argmax(dim=-1) == labels).float().mean())


@torch.no_grad()
def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Top-label ECE: how far reported confidence is from observed accuracy."""
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).to(probs.dtype)
    ece = torch.zeros(())
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not bool(in_bin.any()):
            continue
        # |mean(conf) - mean(acc)| within the bin, NOT mean(|conf - correct|).
        # The latter averages per-sample deviation and returns ~0.4 even for a
        # perfectly calibrated model, because a correct answer at 0.6 confidence
        # already contributes |0.6 - 1| = 0.4 on its own.
        gap = (conf[in_bin].mean() - correct[in_bin].mean()).abs()
        ece = ece + in_bin.to(probs.dtype).mean() * gap
    return float(ece)


@torch.no_grad()
def entropy_confidence_ece(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """ECE on Laya's headline confidence ``1 - H(p)/log k`` against accuracy.

    Reported alongside top-label ECE because Laya publishes that confidence and it is not
    monotone in ``max(p)``; the two numbers are not comparable and both are useful.
    """
    k = probs.shape[-1]
    if k < 2:
        return 0.0
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
    conf = (1.0 - ent / math.log(k)).clamp(0.0, 1.0)
    correct = (probs.argmax(-1) == labels).to(probs.dtype)
    ece = torch.zeros(())
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not bool(in_bin.any()):
            continue
        # |mean(conf) - mean(acc)| within the bin, NOT mean(|conf - correct|).
        # The latter averages per-sample deviation and returns ~0.4 even for a
        # perfectly calibrated model, because a correct answer at 0.6 confidence
        # already contributes |0.6 - 1| = 0.4 on its own.
        gap = (conf[in_bin].mean() - correct[in_bin].mean()).abs()
        ece = ece + in_bin.to(probs.dtype).mean() * gap
    return float(ece)


@torch.no_grad()
def negative_log_likelihood(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return float(
        -torch.log(probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-12)).mean()
    )


@torch.no_grad()
def reliability_curve(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).to(probs.dtype)
    edges = torch.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not bool(in_bin.any()):
            continue
        out.append((float(conf[in_bin].mean()), float(correct[in_bin].mean()), int(in_bin.sum())))
    return out
