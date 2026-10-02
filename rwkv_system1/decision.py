"""Decision primitives, option rendering, and the scoring head.

Mirrors Laya's three primitives -- ``choice``, ``score`` and ``noul`` (a boolean) -- with
the same type indices so a checkpoint config from either project is readable.  ``noul`` is
Laya's name for a two-way true/false question and is rendered here as ``boolean``.

Two design choices, the first of which a measurement has now half-refuted:

* The option set is treated as a *set* by the head's shape, but with ``slot_embeddings`` on --
  which it no longer is by default -- it was not, and the measured size of that failure is large
  enough to be a headline caveat on every number this project reported before 2026-09-22.  Re-ordering a
  trained ``ag_news`` head's option list and mapping the answer back through the permutation:
  the answer followed the option *text* 36% of the time (chance is 25%) and the *slot* the rest,
  and accuracy fell 89.5% -> 37.4% under a rotate-by-two.  On ``laya_urgency`` the same probe
  lost 32.8 pt and landed *below* the 25% chance line.  In other words the head was reading the
  *state* and emitting a distribution over slots in a fixed order rather than evaluating the
  criteria.
  Two candidate causes, and three-seed replication says only one is real: turning
  ``slot_embeddings`` off makes the head exactly order-invariant on its own (accuracy changes by
  0.0 pt under re-ordering, the answer tracks the text 100% of the time, on both tasks), so the
  contamination from options sharing one causal suffix pass contributes nothing measurable once
  the embedding is gone.  The defect lives in the head, not in the layout, which also means a
  per-option suffix pass is not needed for invariance.  ``train.permute_columns`` is the other
  route and reaches the same invariance (+0.7 pt); it is the one to use if the embeddings have to
  stay, i.e. with ``n_layers > 0``, where they are what keeps attention from collapsing.
  What invariance costs: ~1.3 pt of accuracy and ~2 pt of top-label ECE on ``ag_news``, and +0.6
  pt accuracy on ``laya_urgency``.  Kept in perspective: scored against a fixed label list the
  slot shortcut is legitimate information -- that is how most classifiers are trained -- so the
  old numbers are not fake, but they do not license any claim about arbitrary questions or rubric
  text, which is the property Laya's typed-decisions tests.
* Temperature is bucketed by ``(qtype, option-count band)`` and clamped.  Laya ships one
  fitted temperature per bucket because a single global value cannot calibrate a 2-option
  question and a 20-option question at once.  They also clamp it to ``[0.5, 5]``: their own
  ``choice:11+`` bucket came out at ``0.1006``, which multiplies logits ~10x and publishes a
  0.24 top probability as 0.99.  A rule that turns coin flips into certainties is not
  calibration, so the same clamp is applied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import torch
import torch.nn as nn

#: Same indices as Laya's ``QTYPES`` so configs are interchangeable.
QTYPES = {"choice": 0, "score": 1, "boolean": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

TEMP_MIN = 0.5
TEMP_MAX = 5.0


class DecisionType(IntEnum):
    CHOICE = 0
    SCORE = 1
    BOOLEAN = 2

    @property
    def label(self) -> str:
        return self.name.lower()


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


def render_options(qtype: DecisionType, options: list[str]) -> list[str]:
    """Laya-style option text: the label is spelled out, score levels are numbered.

    Called **once, at construction** -- ``Question.options`` is then exactly what goes to the
    backbone. It used to be applied lazily by a ``Question.rendered`` property on top of
    options that some loaders had already rendered, which quietly shipped
    ``"level 0: level 0: ..."`` and ``"false: false: ..."`` to the model for every score and
    boolean question. Nothing errors when the prompt is not the one you meant.
    """
    if qtype is DecisionType.SCORE:
        return [f"level {i}: {o}" for i, o in enumerate(options)]
    if qtype is DecisionType.BOOLEAN:
        false_txt = options[0] or "no, the statement does not hold"
        true_txt = options[1] or "yes, the statement holds"
        return [f"false: {false_txt}", f"true: {true_txt}"]
    return list(options)


def render_criteria_options(qtype: str, criteria) -> tuple[list[str], list[str]]:
    """Render Laya's *criteria* format into ``(prompt text, bare labels)``.

    This is the text Laya's own model was trained on, kept as one function so the corpora
    loader and a live ``DecisionModel.decide`` call cannot drift apart.
    """
    if qtype == "choice":
        criteria = criteria or {}
        return (
            [f"{k}: {criteria[k]}" if criteria[k] not in (None, "") else k for k in criteria],
            list(criteria),
        )
    if qtype == "score":
        return [f"level {i}: {c}" for i, c in enumerate(criteria)], [str(i) for i in range(len(criteria))]
    crit = criteria or {}
    false_crit = crit.get("false") or "no, the statement does not hold"
    true_crit = crit.get("true") or "yes, the statement holds"
    return [f"false: {false_crit}", f"true: {true_crit}"], ["false", "true"]


@dataclass
class Question:
    name: str
    instructions: str
    type: DecisionType
    #: Exactly the strings sent to the backbone, already rendered. Callers that start from
    #: bare names build them with ``render_options`` / ``render_criteria_options``.
    options: list[str]
    #: Bare option names, when ``options`` carries rendered prompt text that also embeds a
    #: rubric description (Laya's format is ``"<name>: <description>"``). Falls back to
    #: ``options`` so callers that only pass plain names keep working.
    labels: list[str] | None = None

    def __post_init__(self):
        if len(self.options) < 2:
            raise ValueError(f"question {self.name!r} needs at least two options")
        if self.type is DecisionType.BOOLEAN and len(self.options) != 2:
            raise ValueError("boolean questions must have exactly two options")
        if self.labels is not None and len(self.labels) != len(self.options):
            raise ValueError("labels and options must be the same length")

    @property
    def answer_labels(self) -> list[str]:
        """What to show a caller as the answer -- bare names, not the rendered rubric."""
        return self.labels if self.labels is not None else self.options


@dataclass
class DecisionExample:
    state: str
    question: Question
    answer: int
    #: Analytic posterior when the corpus can state it, so calibration is measured
    #: against ground truth rather than a proxy.
    true_probs: list[float] | None = None
    meta: dict = field(default_factory=dict)


class DecisionHead(nn.Module):
    """Scores options from their marker hidden states.

    Laya's head runs two transformer layers over the *whole* token sequence and gathers the
    marker positions afterwards. Porting that shape to the marker slots alone does not work,
    and the measurements are unambiguous:

    * the marker features are ~99% a shared component (per-position norm ~22.7, spread
      across options ~0.19), so self-attention over them degenerates into mean pooling;
    * training then drives the option vectors together until their logits agree to 1e-3,
      the softmax is uniform, and cross-entropy sees no signal. Held at chance (0.29) for
      every learning rate tried (3e-3, 1e-2, 3e-2) and with or without slot embeddings.
    * a per-option scorer with no set attention reaches 0.793 on the same features.

    So ``n_layers`` defaults to 0. The backbone already does the contextualisation -- each
    marker is causal over state + question + option -- and the head's job is only to turn one
    contextual vector into one score. Set attention is kept available because it is harmless
    at initialisation and may help on tasks where options genuinely interact, but it is not
    the default and the evidence above is why.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_layers: int = 0,
        dropout: float = 0.1,
        max_options: int = 32,
        slot_embeddings: bool | None = None,
    ):
        super().__init__()
        nhead = max(1, d_model // 64)  # Laya's rule
        # The backbone is 768-1024 wide depending on the checkpoint, but the head only ever
        # sees n_options vectors. Running the attention stack at the backbone's width would
        # spend ~26M parameters fitting a few thousand examples; projecting down first is
        # both faster and better-conditioned.
        self.proj = nn.Linear(in_dim, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.type_emb = nn.Embedding(len(QTYPES), d_model)
        # Self-attention over a set with no slot information degenerates. With near-uniform
        # attention every position receives the same mean update, and a couple of layers
        # drive the option vectors together until their logits agree to within 1e-3 and the
        # softmax is uniform -- so cross-entropy sees no signal and the head sits at chance
        # (measured: 0.29 at every learning rate from 3e-3 to 3e-2, logit spread 0.0002).
        # A slot embedding keeps the positions distinguishable. Laya does not need one
        # because its head runs over the full token sequence, where position already
        # carries identity and the option texts differ.
        #
        # Only set this with ``n_layers > 0``. With the default per-option scorer it buys
        # nothing and is the entire cause of the slot-following defect measured at the top of
        # this module: the head answers from the slot instead of reading the option.
        #
        # ``None`` therefore means "whatever the layout needs": a set-attention stack collapses
        # without a position signal, and a per-option scorer is actively harmed by one.
        if slot_embeddings is None:
            slot_embeddings = n_layers > 0
        self.slot_embeddings = slot_embeddings
        if slot_embeddings:
            self.slot_emb = nn.Embedding(max_options, d_model)
        self.head = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model,
                    nhead,
                    4 * d_model,
                    dropout,
                    batch_first=True,
                    norm_first=True,
                ),
                n_layers,
                enable_nested_tensor=False,
            )
            if n_layers > 0
            else None
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(
        self, feats: torch.Tensor, qtype: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``feats`` is ``(B, n_options, in_dim)``; returns raw logits ``(B, n_options)``."""
        h = self.norm_in(self.proj(feats.float())) + self.type_emb(qtype).unsqueeze(1)
        if self.slot_embeddings:
            n = feats.shape[1]
            h = h + self.slot_emb(torch.arange(n, device=feats.device)).unsqueeze(0)
        if self.head is not None:
            h = self.head(h, src_key_padding_mask=mask)
        logits = self.scorer(h).squeeze(-1)
        return logits.masked_fill(mask, -1e4) if mask is not None else logits

    @staticmethod
    def probabilities(logits: torch.Tensor, temperature=1.0) -> torch.Tensor:
        return torch.softmax(logits / temperature, dim=-1)

    @staticmethod
    def confidence(probs: torch.Tensor) -> torch.Tensor:
        """Laya's headline confidence: ``1 - H(p) / log(k)``, not ``max(p)``.

        For a 4-way question this is 0 on a uniform answer and 1 on a certain one, so it
        reports "how decided am I" rather than "how big is the top bar".
        """
        k = probs.shape[-1]
        if k < 2:
            return torch.ones(probs.shape[:-1])
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
        return (1.0 - ent / math.log(k)).clamp(0.0, 1.0)

    @staticmethod
    def point_answer(question: Question, probs: torch.Tensor) -> tuple[str, float]:
        conf = float(DecisionHead.confidence(probs))
        if question.type is DecisionType.SCORE:
            levels = torch.arange(len(question.options), dtype=probs.dtype)
            return f"{float((probs * levels).sum()):.4f}", conf
        idx = int(probs.argmax())
        return question.answer_labels[idx], conf
