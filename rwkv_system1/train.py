"""Two-stage training of the decision head on cached backbone features, then calibration.

Mirrors Laya's pipeline:

1. **warm-up** with cross-entropy -- good at picking the right answer, bad at saying how
   sure it is, because CE is minimised by pushing mass onto vertices;
2. **RLCD** on the strictly proper reward -- same head, same data, different objective;
3. **temperature fitting** -- one scalar per ``(qtype, option-count band)`` bucket,
   fitted on held-out data.  Laya reports its 0.081 ECE *after* this step, so skipping it
   would compare our raw head against their calibrated one.

Steps 2 and 3 do different jobs: RLCD reshapes the decision boundary, temperature rescales
the logits.  Both are reported so the split can be seen rather than assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .decision import QTYPES, DecisionHead, temp_bucket
from .features import FeatureSet, option_counts, pad_mask, qtype_vector
from .rlcd import (
    accuracy,
    cross_entropy_loss,
    entropy_confidence_ece,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    rlcd_loss,
    soft_cross_entropy,
)


@dataclass
class History:
    warmup: list[dict] = field(default_factory=list)
    rlcd: list[dict] = field(default_factory=list)
    #: one scalar per ``(qtype, option-count band)`` bucket, as Laya ships
    #: ``temperature_by_options``. Empty means every bucket stays at 1.0.
    temperatures: dict[str, float] = field(default_factory=dict)
    #: mean entropy of the cross-fitted soft targets, in nats; None when hard labels
    target_entropy: float | None = None

    def temperature_for(self, qtype: int, n_options: int) -> float:
        return float(self.temperatures.get(temp_bucket(int(qtype), int(n_options)), 1.0))


def permute_columns(
    feats: torch.Tensor, mask: torch.Tensor | None, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Hand every option to the head in a random slot, and return the map back.

    ``perm[b, j]`` is the original option now sitting in column ``j``, so gathering with
    ``perm.argsort(1)`` puts the logits back in canonical order.  Doing the round trip rather
    than leaving the loss in permuted space is what keeps ``score`` questions meaningful: RPS
    cumulative-sums over the option axis, and a shuffled axis is not a level axis.

    This is the only fix the head needs for the slot-following defect measured 2026-09-22 --
    per-option suffix passes were also tried and do not help on their own, because slot identity
    reaches the head through ``slot_emb`` as well as through the shared causal pass.
    """
    b, k = feats.shape[0], feats.shape[1]
    perm = torch.stack([torch.randperm(k, generator=gen) for _ in range(b)])
    out = feats.gather(1, perm.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
    return out, None if mask is None else mask.gather(1, perm), perm.argsort(1)


def _batches(
    fs: FeatureSet,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    soft_targets: torch.Tensor | None = None,
):
    """Batches of ``(features, labels, qtype, pad_mask[, target])``.

    ``qtype`` and ``pad_mask`` travel with the rows rather than being sliced from the whole
    set afterwards, because a joint batch mixes question types and shuffles the rows. A
    single-task set gets an all-False mask, so the training path has one shape to handle.
    """
    masks = pad_mask(fs)
    n = len(fs)
    tensors = [
        fs.features,
        fs.labels,
        qtype_vector(fs),
        torch.zeros(n, fs.features.shape[1], dtype=torch.bool) if masks is None else masks,
    ]
    if soft_targets is not None:
        tensors.append(soft_targets)
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def stratified_folds(labels: torch.Tensor, n_options: int, n_folds: int, seed: int) -> torch.Tensor:
    """Fold assignment that keeps every class spread evenly across folds."""
    folds = torch.zeros(len(labels), dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    for cls in range(n_options):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=gen)]
        folds[perm] = torch.arange(len(idx)) % n_folds
    return folds


def cross_fit_targets(
    fs: FeatureSet,
    head_kwargs: dict,
    n_folds: int = 5,
    epochs: int = 40,
    lr: float = 3e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-2,
    seed: int = 0,
    log=print,
) -> torch.Tensor:
    """Out-of-fold soft targets: what a model that never saw example *i* predicts for it.

    This is the honest way to give a strictly proper rule something to calibrate to.  With
    one-hot labels the reward is maximised by memorising each label, so RLCD has no work to
    do; with out-of-fold predictions the target carries the model's own uncertainty, and
    where the task is genuinely ambiguous the target is genuinely spread out.

    The fold predictions come out sharp -- a trained head is confident even when it is
    guessing -- so they are temperature-scaled before use.  The temperature for fold ``f``
    is fitted on fold ``f+1``, never on ``f`` itself: fitting it on the fold being predicted
    would smuggle those labels into their own target.
    """
    n, k = len(fs), fs.n_options
    folds = stratified_folds(fs.labels, k, n_folds, seed)
    logits = torch.empty(n, k)
    present = folds.unique()

    for f in present.tolist():
        hold = folds == f
        fit = fs.subset(~hold)
        torch.manual_seed(seed + f)
        head = DecisionHead(**head_kwargs)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(seed + f)
        qt = torch.full((len(fit),), fit.qtype, dtype=torch.long)
        head.train()
        for _ in range(epochs):
            for xb, yb in _batches(fit, batch_size, True, gen):
                loss = cross_entropy_loss(head(xb, qt[: len(yb)]), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        head.eval()
        q = torch.full((int(hold.sum()),), fs.qtype, dtype=torch.long)
        with torch.no_grad():
            logits[hold] = head(fs.features[hold], q)
        log(f"    fold {f + 1}/{len(present)}: fit on {len(fit)}, predicted {int(hold.sum())}")

    # Temperature for fold f is fitted on the next fold's logits/labels.
    targets = torch.empty(n, k)
    for f in present.tolist():
        ref = folds == (f + 1) % n_folds
        if not bool(ref.any()):
            ref = folds == f
        t, _ = fit_temperature(logits[ref], fs.labels[ref], fs.qtype)
        targets[folds == f] = torch.softmax(logits[folds == f] / t, dim=-1)
        log(f"    fold {f + 1}: target temperature {t:.2f} (fitted on fold {(f + 1) % n_folds + 1})")
    return targets


def _entropy(t: torch.Tensor) -> float:
    return float(-(t * torch.log(t.clamp_min(1e-12))).sum(-1).mean())


@torch.no_grad()
def head_logits(head: DecisionHead, fs: FeatureSet, batch_size: int = 512) -> torch.Tensor:
    """Raw (temperature-free) logits for a whole split."""
    head.eval()
    qtype = qtype_vector(fs)
    masks = pad_mask(fs)
    chunks = []
    for start in range(0, len(fs), batch_size):
        feats = fs.features[start : start + batch_size]
        m = None if masks is None else masks[start : start + feats.shape[0]]
        chunks.append(head(feats, qtype[start : start + feats.shape[0]], m))
    return torch.cat(chunks)


def mean_probabilities(members: list[torch.Tensor]) -> torch.Tensor:
    """Average member distributions, never their logits.

    One copy on purpose: the scored number and the live answer must come from the same estimator,
    and logit-averaging is a different (sharper) mixture that would not reproduce what was measured
    in ``experiments/harnesses/{seed_ensemble,joint_ensemble}.py``.
    """
    if not members:
        raise ValueError("an ensemble of zero members has no distribution")
    return torch.stack(members).mean(0)


@torch.no_grad()
def head_probabilities(head, fs: FeatureSet, temperature: float = 1.0) -> torch.Tensor:
    """Probabilities for a whole split, from one head or from several averaged.

    ``head`` may be a sequence of heads, which is the 3-seed ensemble: the members' probabilities
    (each already rescaled by ``temperature``) are averaged, not their logits.
    """
    if isinstance(head, (list, tuple)):
        return mean_probabilities([head_probabilities(h, fs, temperature) for h in head])
    return head.probabilities(head_logits(head, fs), temperature)


def evaluate(head, fs: FeatureSet, temperature: float = 1.0) -> dict[str, float]:
    """Metrics for one head, or for the average distribution of several.

    Accepting a sequence here rather than a separate `evaluate_ensemble` keeps one copy of the
    metric formulas: an ensemble scored by a second implementation could disagree with the single
    head on a detail nobody would notice until a table moved.
    """
    probs = head_probabilities(head, fs, temperature)
    labels = fs.labels

    out = {
        "accuracy": accuracy(probs, labels),
        "ece": expected_calibration_error(probs, labels),
        "ece_entropy": entropy_confidence_ece(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "temperature": float(temperature),
    }
    if fs.true_probs is not None:
        # Against the analytic posterior -- the real target. Top-label ECE only ever
        # sees the sampled label, so it cannot tell "correct but overcautious" from
        # "wrong", and it is blind to how the non-top mass is distributed.
        out["posterior_l1"] = float((probs - fs.true_probs).abs().sum(-1).mean() / 2)
        out["posterior_kl"] = float(
            (
                fs.true_probs
                * (
                    torch.log(fs.true_probs.clamp_min(1e-12))
                    - torch.log(probs.clamp_min(1e-12))
                )
            )
            .sum(-1)
            .mean()
        )
    return out


def fit_temperatures(
    head: DecisionHead, dev_fs: FeatureSet, log=print
) -> dict[str, float]:
    """One scalar per ``(qtype, option-count band)``, each fitted on the rows in that bucket.

    Pooling buys a *stable* number, not a per-task one, and measured 2026-09-22 it is not free:
    ``laya_category``'s own 75-row dev asks for T=0.7, but the pooled ``choice:3-5`` bucket (its
    75 rows with ag_news' 500 -- category has 5 options, ag_news 4) returns 1.2-1.5, and
    category's test ECE goes 0.0855 at T=1 -> 0.1571 fitted.  No single task here has a dev set
    big enough to resolve a rescaling (75 rows gave three different answers across seeds), so
    what a pooled bucket really says is "this is ag_news' temperature", applied to everything
    sharing its band.
    """
    logits = head_logits(head, dev_fs)
    qtypes = qtype_vector(dev_fs)
    ks = option_counts(dev_fs)
    groups: dict[str, list[int]] = {}
    for i in range(len(dev_fs)):
        groups.setdefault(temp_bucket(int(qtypes[i]), int(ks[i])), []).append(i)
    out: dict[str, float] = {}
    for bucket, idxs in sorted(groups.items()):
        sel = torch.tensor(idxs)
        t, info = fit_temperature(logits[sel], dev_fs.labels[sel], int(qtypes[sel[0]]))
        out[bucket] = t
        log(
            f"    bucket {bucket:<12} n={len(idxs):>4}  T={t:.3f}"
            f"  (grid argmin {info['argmin_t']:.2f}: dev NLL {info['dev_nll']:.4f}"
            f" vs {info['argmin_nll']:.4f})"
        )
    return out


def train(
    head: DecisionHead,
    train_fs: FeatureSet,
    dev_fs: FeatureSet,
    warmup_epochs: int = 40,
    rlcd_epochs: int = 25,
    lr: float = 3e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-2,
    group_size: int = 8,
    pg_weight: float = 0.2,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    exploration: str = "multinomial",
    explore_sigma: float = 1.0,
    soft_targets: torch.Tensor | None = None,
    warmup_soft_targets: bool = False,
    fit_temperature_on_dev: bool = True,
    permute_options: bool = False,
    seed: int = 0,
    log=print,
) -> History:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    # A separate stream so turning the augmentation on cannot shift the batch order too.
    perm_gen = torch.Generator().manual_seed(seed + 7919)
    history = History()
    history.target_entropy = _entropy(soft_targets) if soft_targets is not None else None
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    def stage(epochs: int, use_rlcd: bool, label: str) -> None:
        rows = history.rlcd if use_rlcd else history.warmup
        # Sample-level calibration is meaningless on a memorised training set: a head
        # that fits the labels to 0.02 CE reports ~0.99 everywhere and is wrong on
        # everything it did not see. Keep the best dev NLL and restore it, otherwise
        # both the accuracy and the ECE we report are just the last overfit epoch.
        #
        # But this rule has its own failure, measured 2026-09-22: on a 400-row train split all
        # three arms of a layout comparison reached 80.5% train accuracy and were then rolled
        # back to epoch 1, whose argmax is one slot for every input, so each arm reported
        # exactly the majority-class rate (37.5%) and the comparison measured the rollback
        # rather than the arms. Confidence grows from the first step, so dev NLL is optimised
        # before dev accuracy has moved -- the smaller the split, the earlier that bites. The
        # four shipped heads are unaffected (best epoch 12/6, 50/30, 7/11, 7/12, none collapsed),
        # but any run on a few hundred rows needs a second opinion: read dev *accuracy* too.
        best_nll, best_state, best_epoch = float("inf"), None, 0
        # Stage 2 always wants the soft target.  Stage 1 wants it only under
        # ``warmup_soft_targets``, which is the distillation wiring: the soft target becomes the
        # *primary* objective for the warm-up rather than a refinement stage's.  Off by default,
        # because it changes every task that ships a soft target and the shipped recipe was
        # measured without it.
        want_targets = (use_rlcd or warmup_soft_targets) and soft_targets is not None
        targets = soft_targets if want_targets else None
        for epoch in range(1, epochs + 1):
            head.train()
            agg = {"loss": 0.0, "proper": 0.0, "pg": 0.0, "verifier": 0.0, "n": 0}
            for batch in _batches(train_fs, batch_size, True, gen, targets):
                feats, labels, qtype, mask = batch[:4]
                back = None
                if permute_options and feats.shape[1] > 1:
                    feats, mask, back = permute_columns(feats, mask, perm_gen)
                logits = head(feats, qtype, mask)
                if back is not None:
                    logits = logits.gather(1, back)
                if use_rlcd:
                    loss, parts = rlcd_loss(
                        logits,
                        labels,
                        qtype,
                        target=batch[4] if targets is not None else None,
                        pad_mask=mask,
                        group_size=group_size,
                        pg_weight=pg_weight,
                        w_sph=w_sph,
                        w_rps=w_rps,
                        exploration=exploration,
                        explore_sigma=explore_sigma,
                    )
                    for k in ("proper", "pg", "verifier"):
                        agg[k] += parts[k] * len(labels)
                else:
                    loss = (soft_cross_entropy(logits, batch[4]) if targets is not None
                            else cross_entropy_loss(logits, labels))
                opt.zero_grad()
                loss.backward()
                opt.step()
                agg["loss"] += float(loss.detach()) * len(labels)
                agg["n"] += len(labels)

            dev = evaluate(head, dev_fs)
            row = {
                "epoch": epoch,
                "train_loss": agg["loss"] / agg["n"],
                "dev_acc": dev["accuracy"],
                "dev_ece": dev["ece"],
                "dev_nll": dev["nll"],
            }
            if dev["nll"] < best_nll:
                best_nll, best_epoch = dev["nll"], epoch
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                row["best"] = True
            rows.append(row)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                extra = f"  reward {agg['verifier'] / agg['n']:.3f}" if use_rlcd else ""
                log(
                    f"    {label} epoch {epoch:>3}/{epochs}  loss {row['train_loss']:.4f}"
                    f"  dev_acc {dev['accuracy']:.4f}  dev_ece {dev['ece']:.4f}"
                    f"  dev_nll {dev['nll']:.4f}{extra}"
                )
        if best_state is not None and best_epoch != epochs:
            head.load_state_dict(best_state)
            log(f"    {label} restored epoch {best_epoch} (best dev NLL {best_nll:.4f})")

    log(f"  stage 1: cross-entropy warm-up ({warmup_epochs} epochs"
        f"{', on the soft target' if warmup_soft_targets and soft_targets is not None else ''})")
    stage(warmup_epochs, use_rlcd=False, label="ce  ")
    raw = evaluate(head, dev_fs)
    log(f"    -> raw (T=1)  dev_acc {raw['accuracy']:.4f}  dev_ece {raw['ece']:.4f}")

    if rlcd_epochs > 0:
        opt = torch.optim.AdamW(head.parameters(), lr=lr * 0.3, weight_decay=weight_decay)
        has_score = bool((qtype_vector(train_fs) == QTYPES["score"]).any())
        log(
            f"  stage 2: RLCD proper reward (log + {w_sph}*spherical"
            f"{f' - {w_rps}*RPS on score rows' if has_score else ''})"
            f", group={group_size}, pg_weight={pg_weight}"
            + (f", exploration={exploration}(sigma={explore_sigma})"
               if exploration != "multinomial" else "")
        )
        stage(rlcd_epochs, use_rlcd=True, label="rlcd")

    if fit_temperature_on_dev:
        log("  stage 3: temperature fit on dev, one scalar per bucket:")
        history.temperatures = fit_temperatures(head, dev_fs, log=log)

    return history
