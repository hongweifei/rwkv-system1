"""The run configuration: sizes, token budgets, head shape, recipe hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    n_train: int = 1200
    n_dev: int = 300
    n_test: int = 500
    n_cues: int = 2
    p_neutral: float = 0.35
    n_filler: int = 3
    batch_extract: int = 8
    #: The context the checkpoint was trained at.  The option block is never truncated; this
    #: bounds state + block only, and the feature fingerprint tracks it because changing it
    #: changes which texts survive.
    ctx_tokens: int = 8192
    #: How a marker position becomes a feature vector -- ``features.GATHER_RULES``.  ``last`` is the
    #: terminator's own activation; ``pooled`` averages it over the option's token span;
    #: ``two_layer`` concatenates it with the same position from a middle block.  All three read one
    #: forward pass, so the alternatives cost cache width, not compute.  It is a *feature* knob and
    #: lands in ``feature_inputs``, never in ``head_inputs`` -- the head records the resulting
    #: identity instead (``head_inputs`` hashes ``fs.inputs``), which is what keeps a pooled head
    #: from overwriting the ``last`` head of the same task and size.
    #:
    #: ``None`` means **no opinion**: the task's own pin then decides, else
    #: ``features.SHIPPED_RULE``.  The three levels matter because a pin that no CLI flag could
    #: override would make an arm extraction unreproducible, so an explicit value here always wins
    #: (see ``cache_layout.resolve_rule``).  Setting it is how the diagnostic arms in
    #: ``experiments/chains`` are run; the standing recipe leaves it unset.
    feature_rule: str | None = None
    head_d_model: int = 256
    head_layers: int = 0
    head_dropout: float = 0.1
    #: Option-order identity. ``None`` = derive it from ``head_layers``: a per-option scorer
    #: must not get one (with it on, only 36% of answers follow the option text and re-ordering
    #: the options costs 51 pt of accuracy), while a set-attention stack collapses without one.
    #: Set it explicitly only to run the ablation.
    head_slot_embeddings: bool | None = None
    #: Randomise which slot each option is handed to during training. See
    #: :func:`rwkv_system1.train.permute_columns`, which is why this does not break RPS.
    permute_options: bool = False
    targets: str = "gold"
    #: Distillation wiring: train the *warm-up* against the soft target instead of the one-hot
    #: label, so the distribution is the primary objective rather than a refinement stage's.  Off
    #: by default -- it changes every task that ships a soft target, and the shipped recipe was
    #: measured without it.  Turning it on also needs ``targets`` to be something that produces
    #: one.  Measured on sst5 it is 2.67±0.33pt *worse* than the labels it replaces; see
    #: :mod:`rwkv_system1.teacher` for the three-seed decomposition behind that.
    warmup_soft_targets: bool = False
    #: The online teacher, used when ``targets == "teacher"``.  It changes the head, so it is in
    #: the head's identity record; the worker count is not, because it only changes how long the
    #: asking takes.  See :mod:`rwkv_system1.teacher` for the measured per-task accuracy gap that
    #: decides whether this is worth turning on for a given task.
    teacher_model: str = "qwen3.8-flash"
    teacher_workers: int = 6
    crossfit: bool = True
    crossfit_folds: int = 5
    crossfit_epochs: int = 40
    warmup_epochs: int = 60
    rlcd_epochs: int = 30
    lr: float = 3e-3
    batch_size: int = 64
    weight_decay: float = 1e-2
    group_size: int = 8
    pg_weight: float = 0.2
    #: How the policy-gradient stage draws its actions. ``multinomial`` samples the reported
    #: distribution (identical to ``argmax(logits + Gumbel)``); ``gaussian`` perturbs the logits
    #: with ``explore_sigma * N(0, 1)`` and takes the argmax, which is the probit exploration
    #: Laya's write-up describes. Same expected gradient either way -- an ablation knob.
    exploration: str = "multinomial"
    explore_sigma: float = 1.0
    w_sph: float = 0.5
    w_rps: float = 1.0
    #: Seeds both the head's weight init and the batch order. ``train()`` seeds the latter
    #: itself, but it is handed an already-constructed head, and a fresh process's torch RNG is
    #: seeded from the OS -- so without seeding here the saved ``.pt`` is not reproducible from
    #: this config at all.
    seed: int = 0
