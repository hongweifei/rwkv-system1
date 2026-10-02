"""One-off feature extraction from the frozen backbone, cached to disk.

The backbone costs ~100 GFLOP per decision, so it runs exactly once per split and
everything downstream (head training, RLCD, evaluation) reads cached vectors.

Two bits of care keep this both fast and correct:

* States are bucketed by *exact* token length and batched within a bucket, so no
  padding ever enters the recurrence.  Padding a state would silently corrupt the
  carried recurrent state, and unlike attention there is no mask that can undo it.
* Queries are right-padded, which *is* safe: an RWKV hidden state at position ``t``
  depends only on tokens at or before ``t``, so padding after the last option cannot
  touch any marker we read.

Only the marker positions are cached -- ``(n_options, d_model)`` per example -- which
keeps a split at a few megabytes instead of the hundreds a full-sequence cache would
cost, and makes head training and RLCD nearly free.

:data:`GATHER_RULES` is how a marker position becomes a vector, and it is part of the
feature identity (recorded as ``marker_rule`` in ``cache_layout.feature_inputs``).  ``last``
is the shipped rule and maps to the same integer as before the alternatives existed, so
adding them did not invalidate a single cache.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .backbone import DecisionBackbone
from .decision import QTYPES
from .rwkv7 import RWKV7


#: Gather rule -> the integer recorded as ``marker_rule`` in the feature identity.  ``last`` is 1
#: because that is the rule the shipped caches were written under; the alternatives take new
#: numbers so asking for one produces new file names instead of reusing the vectors of another.
GATHER_RULES = {"last": 1, "pooled": 2, "two_layer": 3}


#: What a task runs under when neither the config nor the registry has an opinion.  ``pooled`` wins
#: by ~10 pt on every task with a large label space and is a wash with real losses below it, so the
#: registry pins it where it pays and this stays the safe default -- see ``task_registry`` for the
#: measurement.  Named once here because a default spelled twice is a default that can disagree.
SHIPPED_RULE = "last"


def gathered_dim(d_model: int, rule: str) -> int:
    """How wide one option's feature vector comes out under ``rule``.

    ``two_layer`` concatenates two ``d_model`` vectors, so the output buffer cannot be sized from
    ``d_model`` alone -- sizing it that way is what made the first run of the thick-feature screen
    fail with "expanded size (768) must match existing size (1536)".
    """
    return d_model * 2 if rule == "two_layer" else d_model


@dataclass
class FeatureSet:
    #: ``(N, n_options, d_model)`` marker hidden states, fp16
    features: torch.Tensor
    labels: torch.Tensor  # (N,)
    n_options: int
    qtype: int
    true_probs: torch.Tensor | None  # (N, n_options), analytic when available
    states: list[str]
    options: list[str]
    qtype_name: str
    #: Everything that shaped these vectors, as produced by ``cache_layout.feature_inputs``.
    #: Stored with the cache and compared wholesale on reuse, so a new input does not need to be
    #: added to the name, the sidecar and the check separately.
    inputs: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[-1]

    def subset(self, mask: torch.Tensor) -> "FeatureSet":
        m = mask.bool()
        keep = m.tolist()
        return FeatureSet(
            features=self.features[m],
            labels=self.labels[m],
            n_options=self.n_options,
            qtype=self.qtype,
            true_probs=self.true_probs[m] if self.true_probs is not None else None,
            states=[s for s, k in zip(self.states, keep) if k],
            options=self.options,
            qtype_name=self.qtype_name,
            inputs=self.inputs,
        )

    def save(self, path: str | Path) -> None:
        """Tensors to a ``.pt`` loadable with ``weights_only=True``, strings to a
        sidecar JSON, so nothing on disk ever needs unpickling.  Extensions are
        appended rather than swapped because cache names start with the backbone tag
        (``rwkv7-g1d-0.1b_...``) and ``Path.with_suffix`` would read the dot in ``0.1b``
        as the extension separator."""
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "n_options": self.n_options,
                "qtype": self.qtype,
                "true_probs": self.true_probs,
            },
            str(path) + ".pt",
        )
        Path(str(path) + ".json").write_text(
            json.dumps(
                {"states": self.states, "options": self.options,
                 "qtype_name": self.qtype_name, "inputs": self.inputs},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        tensors = torch.load(str(path) + ".pt", map_location="cpu", weights_only=True)
        strings = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
        return cls(**tensors, **strings)


@dataclass
class JointFeatureSet:
    """Several per-task caches laid side by side, so one head can train on every question.

    MEASURED TWICE on 2026-09-22, and the second run reversed the first, so read all of it.
    Same cached features, same recipe, 3 seeds, joint over 5350 rows against four per-task heads.
    Under the old head that carried slot embeddings, sharing lost on every task -- NLL worse on
    all four, accuracy worse on three.  With ``slot_embeddings`` derived off (what ships now),
    joint *wins* where one head is data starved and loses on the one score task:
    ``laya_category`` +5.0 pt accuracy (3/3 seeds, 8.7 SE) with better NLL, ``ag_news`` better by
    0.062 nats (3/3, 7.9 SE) and 1.4 pt of top-label ECE at T=1, ``laya_needs_human`` a wash, but
    ``laya_urgency`` -2.5 pt accuracy and +0.032 nats, wrong on all three seeds.  The reversal is
    the finding: the per-task advantage had been the slot shortcut, which each head could
    memorise alone, and once the head has to read the option text, pooling four corpora is what
    carries.  Two costs that no accuracy column shows: a joint head pools the temperature buckets
    (see ``train.fit_temperatures``, which records category losing 5 pt of ECE to that), and one
    readout has to serve four rubrics, so ``laya_urgency`` is where it pays.  Training wall clock
    is the same either way (~70 s/seed for the four, ~75 s for the joint) and inference costs
    nothing extra, because the scorer is per-option, so one active head is the same FLOPs.

    Re-measured 2026-10-01 on the families where the premise actually holds -- one task in several
    languages, so the option list, the instruction and the label space are identical and only the
    state differs (``experiments/results/joint_family.json``, 3 seeds, pinned-pooled caches):
    MASSIVE's ten languages gain **+9.7 pt accuracy (10/10, +7.2 to +13.7) and -0.38 nats of NLL
    (10/10)**, and that head is what ships, declared by ``RealTask.joint_group``.  XNLI gains 3.5 pt
    of accuracy but loses 4 pt of ECE and 0.02 nats of NLL, because its eight per-task fitted
    temperatures span 1.00 to 5.00 and one pooled bucket cannot serve them; amazon reviews is
    positive on all three but inside its own standard error.  So the rule the 09-22 run could not
    settle is now settled the narrow way: **share a head across languages of one task when the
    label space is large and the per-language row count is small, and check the fitted-temperature
    spread before believing the accuracy column.**  The statistical unit is the family, not the
    language -- ten positive rows here are one head read ten ways.

    The rows disagree on option count (2, 4 and 5), so they are right-padded to ``Kmax`` and
    ``mask`` records which slots are real. Padding goes at the end because a slot's meaning is
    positional -- the first ``n_options`` entries are that question's own options in order.

    Without the mask a 2-option question is scored as though it owned five: the padded slots
    join the entropy normaliser and the RPS cumulative sums, and the reported confidence is
    then a statement about slots that do not exist.
    """

    features: torch.Tensor  # (N, Kmax, D) fp16
    labels: torch.Tensor  # (N,)
    mask: torch.Tensor  # (N, Kmax) bool, True on padding
    qtypes: torch.Tensor  # (N,) long
    n_options: torch.Tensor  # (N,) long, the real option count per row
    true_probs: torch.Tensor | None  # (N, Kmax) float32, zero on padding
    task_ids: torch.Tensor  # (N,) long, index into task_names
    task_names: list[str]
    options_by_task: dict[str, list[str]]
    #: how many of the N rows carry a teacher distribution rather than a filled-in one-hot
    soft_rows: int = 0

    def __len__(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[-1]

    def subset(self, keep: torch.Tensor) -> "JointFeatureSet":
        m = keep.bool()
        return JointFeatureSet(
            features=self.features[m],
            labels=self.labels[m],
            mask=self.mask[m],
            qtypes=self.qtypes[m],
            n_options=self.n_options[m],
            true_probs=self.true_probs[m] if self.true_probs is not None else None,
            task_ids=self.task_ids[m],
            task_names=self.task_names,
            options_by_task=self.options_by_task,
        )


def concat_joint(sets: list[tuple[str, FeatureSet]]) -> JointFeatureSet:
    """Lay ``(name, FeatureSet)`` pairs side by side. All must share ``d_model``."""
    if not sets:
        raise ValueError("no feature sets to join")
    dims = {name: fs.feature_dim for name, fs in sets}
    if len(set(dims.values())) > 1:
        raise ValueError(f"cannot join caches of different widths: {dims}")
    kmax = max(fs.n_options for _, fs in sets)
    d = sets[0][1].feature_dim
    n = sum(len(fs) for _, fs in sets)
    feats = torch.zeros(n, kmax, d, dtype=torch.float16)
    mask = torch.ones(n, kmax, dtype=torch.bool)
    labels = torch.zeros(n, dtype=torch.long)
    qtypes = torch.zeros(n, dtype=torch.long)
    n_opts = torch.zeros(n, dtype=torch.long)
    tasks = torch.zeros(n, dtype=torch.long)
    probs = torch.zeros(n, kmax, dtype=torch.float32)

    pos, n_soft = 0, 0
    for t, (name, fs) in enumerate(sets):
        m = len(fs)
        k = fs.n_options
        feats[pos : pos + m, :k] = fs.features
        mask[pos : pos + m, :k] = False
        labels[pos : pos + m] = fs.labels
        qtypes[pos : pos + m] = fs.qtype
        n_opts[pos : pos + m] = k
        tasks[pos : pos + m] = t
        if fs.true_probs is not None:
            probs[pos : pos + m, :k] = fs.true_probs
            n_soft += m
        else:
            # A hard-label task (ag_news) contributes one-hot rows instead of being left out of
            # the joint set. The proper reward is computed row by row, and against a one-hot
            # target its log-score term *is* cross-entropy, so such a row trains exactly as it
            # would alone; dropping the task instead would leave the shipped head unable to
            # answer it.
            probs[pos : pos + m].scatter_(1, fs.labels[:, None], 1.0)
        pos += m
    return JointFeatureSet(
        features=feats[:pos],
        labels=labels[:pos],
        mask=mask[:pos],
        qtypes=qtypes[:pos],
        n_options=n_opts[:pos],
        true_probs=probs[:pos],
        task_ids=tasks[:pos],
        task_names=[name for name, _ in sets],
        options_by_task={name: fs.options for name, fs in sets},
        soft_rows=n_soft,
    )


def qtype_vector(fs, n: int | None = None) -> torch.Tensor:
    """Per-example question type, for either a single-task or a joint cache."""
    total = len(fs) if n is None else n
    if hasattr(fs, "qtypes"):
        return fs.qtypes[:total]
    return torch.full((total,), int(fs.qtype), dtype=torch.long)


def pad_mask(fs, n: int | None = None):
    """The (n, K) padding mask, or None when every slot is real."""
    m = getattr(fs, "mask", None)
    return None if m is None else m[: len(fs) if n is None else n]


def option_counts(fs) -> torch.Tensor:
    if hasattr(fs, "n_options") and torch.is_tensor(fs.n_options):
        return fs.n_options
    return torch.full((len(fs),), int(fs.n_options), dtype=torch.long)


def read_sidecar(path: str | Path) -> dict | None:
    """The sidecar JSON beside a cached artifact, or None when there is not one."""
    sidecar = Path(str(path) + ".json")
    return json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None


def stored_mismatch(stored: dict, inputs: dict, options: list[str], states: list[str]) -> str:
    """The check every cached artifact shares: the inputs record, then the row identity.

    One copy on purpose.  This comparison is what caught five separate staleness bugs -- a
    missing fingerprint component, a name claiming sizes the file did not hold, no
    data-version lever, a renderer that changed the text under a stable name, a seed that
    reached the head's batches but not its weights -- and two copies of it would eventually
    disagree about which fields matter, which is the failure it exists to prevent.

    The strings are already in the sidecar, so the text half costs a comparison, not a hash.
    """
    was = stored.get("inputs")
    if was != inputs:
        # The whole record, in one comparison: this is what makes a new input a one-line change
        # in `feature_inputs` rather than something to remember in three places.
        if not isinstance(was, dict):
            return "the cache predates the inputs record"
        differing = sorted(k for k in set(was) | set(inputs) if was.get(k) != inputs.get(k))
        return f"inputs changed: {', '.join(differing)}"
    if stored.get("options") != list(options):
        return "option text changed"
    old_states = stored.get("states", [])
    if len(old_states) != len(states):
        return f"row count {len(old_states)} -> {len(states)}"
    for i, (was_text, now) in enumerate(zip(old_states, states)):
        if was_text != now:
            return f"state text changed at row {i}: {was_text[:40]!r} -> {now[:40]!r}"
    return ""


def cache_mismatch(
    path: str | Path,
    inputs: dict,
    queries: list[tuple[str, str, list[str]]],
    states: list[str],
    labels: list[int],
) -> str:
    """Why a cached split no longer matches what this code would extract; '' if it matches.

    The cache name covers the token budgets, the template and the marker rule, but not the
    *content* -- which is the one thing a caller can change without touching any of them, because
    a loader or a renderer builds it.  All three pieces are compared here:

    * the option text, after a lazily-applied renderer shipped ``"level 0: level 0: ..."`` to the
      backbone for every score question while the cache kept its name and got reused forever;
    * the state text, because a loader's own knobs shape it too -- ``load_ag_news`` cuts at
      ``max_chars=1200``, and that default is in neither the fingerprint nor the cache name, so
      changing it would otherwise have gone unnoticed;
    * the labels, because a loader can remap gold without touching either of the other two --
      reorder a class list, change ``sorted()`` of a label set, fix an off-by-one -- and the
      cached features would then be trained against the *old* answers.  The labels live in the
      ``.pt`` rather than the sidecar, so this costs one tensor load per split.
    """
    stored = read_sidecar(path)
    if stored is None:
        return "no sidecar"
    qtype_name, _, options = queries[0]
    # Before the shared check: when a task's question schema changes, the options change with it,
    # and "question type" names the actual cause where "option text changed" would only name a
    # symptom.
    if stored.get("qtype_name") != qtype_name:
        return f"question type {stored.get('qtype_name')!r} -> {qtype_name!r}"
    why = stored_mismatch(stored, inputs, options, states)
    if why:
        return why
    cached = torch.load(str(path) + ".pt", map_location="cpu", weights_only=True)
    cached_labels = cached["labels"].tolist()
    if cached_labels != list(labels):
        for i, (was, now) in enumerate(zip(cached_labels, labels)):
            if was != now:
                return f"label changed at row {i}: {was} -> {now}"
        return f"label count {len(cached_labels)} -> {len(labels)}"
    return ""


@torch.inference_mode()
def markers_for_state(
    backbone: DecisionBackbone,
    state: str,
    queries: list[tuple[str, str, list[str]]],
    rule: str = SHIPPED_RULE,
) -> list[torch.Tensor]:
    """Marker features for every query asked about one state, on a single prefill.

    This is the payoff of choosing a recurrent backbone: the state is encoded once and
    each question only pays for its own question-and-options block.  With a Transformer
    each question would re-run the whole state, or else hold a KV cache per question.

    ``rule`` has to be the rule the head was *trained* on, which is per task
    (``cache_layout.resolve_rule``): it used to be hard-coded to ``last``, so asking a
    ``pooled``-pinned task live would have fed the head terminator activations where it expects
    span means -- a wrong vector with no error anywhere, and the ten MASSIVE heads ship on
    ``pooled``.
    """
    if rule not in GATHER_RULES:
        raise ValueError(f"unknown gather rule {rule!r}; use {sorted(GATHER_RULES)}")
    if rule == "two_layer":
        raise ValueError("markers_for_state serves the rules that ship; two_layer is a "
                         "cache-side experiment and needs forward_layers")
    pre = backbone.encode_state(state)
    _, _, carried = backbone.model.forward(pre, return_hidden=False)
    out = []
    for q in queries:
        suffix, markers = backbone.encode_query(*q)
        backbone.check_room(int(pre.shape[1]), int(suffix.shape[1]))
        hidden, _, _ = backbone.model.forward(
            suffix, RWKV7.clone_state(carried), return_hidden=True
        )
        out.append(_gather_markers(
            hidden[0], ids=suffix[0].tolist(), markers=markers, rule=rule, mid=None,
            terminator_id=backbone.terminator_id,
        ).to(torch.float16))
    return out


@torch.inference_mode()
def _gather_markers(
    hidden: torch.Tensor,
    ids: list[int],
    markers: list[int],
    rule: str,
    mid: torch.Tensor | None,
    terminator_id: int,
) -> torch.Tensor:
    """``(K, C)`` for one example: each option's marker position, read the way ``rule`` says.

    The three rules expose different amounts of the *same* forward pass, which is the point -- none
    of them costs a second pass over the sequence:

    * ``last`` -- the terminator activation alone, one 768-d vector per option. What ships.
    * ``pooled`` -- the mean of that activation over the option's own token span. Option ``k``
      occupies the tokens after the terminator that closed option ``k-1`` (for ``k = 0``, the one
      that closed the question header) up to and including its own terminator, so the spans are
      recoverable from the terminator positions.
    * ``two_layer`` -- the terminator activation from the last block concatenated with the one from
      block ``n_layer // 2``, both post-``ln_out`` so they share a space. 1536-d.
    """
    if rule == "last":
        return hidden.index_select(0, torch.tensor(markers))
    if rule == "pooled":
        stops = [i for i, t in enumerate(ids) if t == terminator_id]
        return torch.stack([hidden[stops[k] + 1 : stops[k + 1] + 1].mean(0)
                            for k in range(len(markers))])
    if rule == "two_layer":
        if mid is None:
            raise ValueError("two_layer needs the intermediate block's hidden states")
        return torch.cat([hidden.index_select(0, torch.tensor(markers)),
                          mid.index_select(0, torch.tensor(markers))], dim=-1)
    raise ValueError(f"unknown gather rule {rule!r}; use {sorted(GATHER_RULES)}")


def extract(
    backbone: DecisionBackbone,
    states: list[str],
    queries: list[tuple[str, str, list[str]]],
    labels: list[int],
    true_probs: list[list[float]] | None = None,
    batch_size: int = 8,
    inputs: dict | None = None,
    rule: str = "last",
) -> FeatureSet:
    """``queries`` items are ``(qtype_name, instructions, options)``.

    ``inputs`` is the identity record (``cache_layout.feature_inputs``) and is stored with the
    features so a later reuse can be checked against the whole thing at once.  ``rule`` picks how a
    marker becomes a feature vector (see :func:`_gather_markers`).
    """
    if rule not in GATHER_RULES:
        raise ValueError(f"unknown gather rule {rule!r}; use {sorted(GATHER_RULES)}")
    if not (len(states) == len(queries) == len(labels)):
        raise ValueError("states, queries and labels must be the same length")
    if not states:
        raise ValueError("empty extraction set")

    n_options = len(queries[0][2])
    for _, _, opts in queries:
        if len(opts) != n_options:
            raise ValueError("all queries in one extraction must share an option count")

    state_ids = [backbone.encode_state(s) for s in states]
    encoded = [backbone.encode_query(*q) for q in queries]
    suffix_ids = [e[0] for e in encoded]
    markers = [e[1] for e in encoded]
    d_model = backbone.model.n_embd
    # Nothing downstream truncates, so the whole prompt has to fit: check it here, where both
    # halves are known, instead of letting either half be cut silently.
    backbone.check_room(max(int(t.shape[1]) for t in state_ids),
                        max(int(t.shape[1]) for t in suffix_ids))

    out = torch.empty(len(states), n_options, gathered_dim(d_model, rule), dtype=torch.float16)

    buckets: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(state_ids):
        buckets[p.shape[1]].append(i)

    for length, idxs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start : start + batch_size]
            b = len(chunk)

            pre = torch.cat([state_ids[i] for i in chunk])  # equal length by construction
            _, _, state = backbone.model.forward(pre, return_hidden=False)

            l_suf = max(suffix_ids[i].shape[1] for i in chunk)
            suffix = torch.zeros(b, l_suf, dtype=torch.long)
            for j, i in enumerate(chunk):
                s = suffix_ids[i]
                suffix[j, : s.shape[1]] = s
            if rule == "two_layer":
                mid_layer = backbone.model.n_layer // 2
                layers, _ = backbone.model.forward_layers(
                    suffix, state, collect=(mid_layer, backbone.model.n_layer - 1)
                )
                hidden, mid = layers[backbone.model.n_layer - 1], layers[mid_layer]
            else:
                hidden, _, _ = backbone.model.forward(suffix, state, return_hidden=True)
                mid = None

            for j, i in enumerate(chunk):
                mid_j = None if mid is None else mid[j]
                out[i] = _gather_markers(
                    hidden[j], ids=suffix_ids[i][0].tolist(), markers=markers[i], rule=rule,
                    mid=mid_j, terminator_id=backbone.terminator_id,
                ).to(torch.float16)

    # A hard-label corpus (ag_news) carries None per example, so the list itself is not the
    # right test for "are there soft targets".
    has_probs = true_probs is not None and true_probs[0] is not None
    probs_t = torch.tensor(true_probs, dtype=torch.float32) if has_probs else None
    return FeatureSet(
        features=out,
        labels=torch.tensor(labels, dtype=torch.long),
        n_options=n_options,
        qtype=QTYPES[queries[0][0]],
        true_probs=probs_t,
        states=list(states),
        options=list(queries[0][2]),
        qtype_name=queries[0][0],
        inputs=dict(inputs or {}),
    )
