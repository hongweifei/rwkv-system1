"""Where a feature cache lives, what it is called, and what invalidates it.

Identity is defined **once**: :func:`feature_inputs` for a cache, :func:`head_inputs` for a head.
That same dict is hashed into the file name, written into the artifact, and compared wholesale on
reuse, so adding an input is one line and the name, the stored record and the check all follow.

That is the fix for a *class* of bug rather than an instance.  Six of them so far, every one an
input recorded in one place but not the others: a missing fingerprint component; a name claiming
split sizes the file did not hold; no data-version lever at all; a renderer that changed the text
under a stable name; a seed that reached the head's batches but not its weights; and a state cap
that cut the text while the name said nothing about it.

Layer: identity only.  It never trains and never extracts; it decides names.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from . import corpus
from .task_registry import REAL_TASKS
from .decision import DecisionHead, temp_bucket
from .features import GATHER_RULES, SHIPPED_RULE, FeatureSet, read_sidecar


ROOT = Path(__file__).resolve().parent.parent


CACHE = ROOT / "cache"


RUNS = ROOT / "runs"


BACKBONES = {
    # RWKV7-G1 "GooseOne" (ModelScope `RWKV/rwkv7-g1`). Architecturally identical to the World
    # v2.9 x070 checkpoints -- the tensor-name sets match key for key, so the loader needs no
    # branch; only the widths differ (rwkv7-g1d-0.1b: 12 layers, d_model 768, 12 heads x 64).
    "rwkv7-g1d-0.1b": "rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "rwkv7-g1d-0.4b": "rwkv7-g1d-0.4b-20260210-ctx8192.pth",
}


BACKBONE_REPO = "RWKV/rwkv7-g1"


DEFAULT_BACKBONE = "rwkv7-g1d-0.1b"


def checkpoint(name: str) -> Path:
    p = ROOT / "weights" / BACKBONES[name]
    if not p.exists():
        raise SystemExit(
            f"missing {p}\nrun: uv run modelscope download --model {BACKBONE_REPO} "
            f'--include "{BACKBONES[name]}" --local-dir weights'
        )
    return p


def _file_identity(path: Path) -> dict:
    """Name, size and a head+tail digest: enough to notice a swapped file without hashing 400 MB."""
    size = path.stat().st_size
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        digest.update(fh.read(65536))
        if size > 131072:
            fh.seek(-65536, 2)
            digest.update(fh.read(65536))
    return {"name": path.name, "size": size, "digest": digest.hexdigest()[:12]}


def resolve_rule(rule: str | None, key: str | None) -> str:
    """Which gather rule a task actually runs under.  Explicit beats pinned beats shipped.

    The middle level exists because the rule pays ~10 pt on a large label space and is a wash below
    it, so it is pinned per task in ``REAL_TASKS`` (see :class:`task_registry.RealTask`).  The top
    level has to exist too: without it a CLI ``--feature-rule`` could not override a pin, and the
    arms that measure a rule against its alternative would stop being reproducible.

    ``key=None`` means "not about a particular task" -- the invariants and the harnesses that vary
    the rule themselves call it that way.
    """
    if rule is not None:
        return rule
    task = REAL_TASKS.get(key) if key is not None else None
    return (task.feature_rule if task is not None else None) or SHIPPED_RULE


def feature_inputs(
    cfg: Config, backbone_name: str, data_version_of: int | None = None, key: str | None = None
) -> dict:
    """Everything that shapes a cached vector, in one dict -- the definition of feature identity.

    Hashed into the cache name by :func:`feature_fingerprint`, stored inside the cache by
    ``extract``, and compared wholesale by ``cache_mismatch``.  The two *file* identities are as
    load-bearing as the config: a re-downloaded checkpoint or a swapped tokeniser table changes
    every vector while leaving the backbone's name -- which is also the filename prefix -- alone.
    """
    from .backbone import PromptTemplate, find_vocab

    t = PromptTemplate()
    inputs = {
        "template": [t.state_suffix, t.question_header, t.terminator],
        # No state cap any more: the policy is recorded so that re-introducing one changes this
        # hash rather than silently re-defining what a cache holds.
        "state_cap": None,
        "ctx_tokens": cfg.ctx_tokens,
        # How a marker position becomes a feature vector (``features.GATHER_RULES``).  The shipped
        # rule maps to the same integer as before the alternatives existed, so adding them did not
        # move a single cache name.
        "marker_rule": GATHER_RULES[resolve_rule(cfg.feature_rule, key)],
        "tokenizer": _file_identity(find_vocab()),
        "checkpoint": _file_identity(ROOT / "weights" / BACKBONES[backbone_name]),
    }
    if data_version_of is not None:
        inputs["data_version"] = int(data_version_of)
    return inputs


def fingerprint_of(inputs: dict) -> str:
    """Short hash of an identity dict.  One function so a name and a record cannot disagree."""
    return hashlib.sha1(
        json.dumps(inputs, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]


def feature_fingerprint(cfg: Config, backbone_name: str, key: str | None = None) -> str:
    """Short hash of :func:`feature_inputs` -- the identity component of a cache's name."""
    return fingerprint_of(feature_inputs(cfg, backbone_name, key=key))


def joint_group(key: str) -> str | None:
    """The name of the group whose head serves ``key``, or None for a head of its own."""
    task = REAL_TASKS.get(key)
    return task.joint_group if task is not None else None


def group_members(key: str) -> list[str] | None:
    """Every task that shares one head with ``key``, in registry order, or None.

    Read from the registry, never inferred from a name prefix: a prefix rule would quietly fold a
    task into a group the day someone names a task after the same corpus, and a head trained over
    tasks that do not share an option list is the failure that was measured and lost.
    """
    group = joint_group(key)
    if group is None:
        return None
    return [k for k, t in REAL_TASKS.items() if t.joint_group == group]


def group_sidecars(members: list[str], backbone_name: str, cfg: Config,
                   split: str) -> dict[str, dict]:
    """The recorded sidecar of every member's cache for one split.

    Read from disk (one small JSON per member, no tensors) rather than hashed off the in-memory
    ``JointFeatureSet``, because the two callers of :func:`head_inputs` hold different things:
    ``cmd_train`` has the joined set, ``load_head`` has one member's *test* cache.  Both have to
    derive the same number or the head they name is not the same head.
    """
    out = {}
    for m in members:
        side = read_sidecar(CACHE / cache_name(backbone_name, m, split, cfg))
        if side is None or "inputs" not in side or "states" not in side:
            raise SystemExit(
                f"{m}: no cache sidecar for split {split!r} to derive the group's identity from; "
                f"extract the whole group before training its head")
        out[m] = side
    return out


def group_feature_inputs(members: list[str], backbone_name: str, cfg: Config,
                         split: str = "train") -> dict:
    return {m: side["inputs"] for m, side in
            group_sidecars(members, backbone_name, cfg, split).items()}


def group_rows(members: list[str], backbone_name: str, cfg: Config, split: str = "train") -> int:
    """How many rows the group's head actually sees in ``split`` -- not what was requested.

    MASSIVE asks for 1,100 train rows per language and the pools hand over 1,074 to 1,099, so a tag
    built from the registry would print 11000 for a head trained on 10930.  That is the same defect
    this pipeline has now met five times over: an identity string that describes a request rather
    than the artifact, which is exactly what makes a stale read invisible.
    """
    return sum(len(side["states"]) for side in
               group_sidecars(members, backbone_name, cfg, split).values())


def effective_sizes(key: str, cfg: Config) -> tuple[int, int, int]:
    """The split sizes a task will actually use.

    A real corpus may pin its own sizes in the registry -- ``laya_category`` has only 300
    labelled rows, so it cannot honour a global 1200 -- and the cache name has to reflect what
    was *actually* loaded, not what the CLI said. Otherwise the filename claims 1200 rows for a
    file holding 900, and a later run with matching flags can silently pick up the wrong cache.
    """
    task = REAL_TASKS.get(key)
    extra = task.kwargs if task else {}
    return (
        int(extra.get("n_train", cfg.n_train)),
        int(extra.get("n_dev", cfg.n_dev)),
        int(extra.get("n_test", cfg.n_test)),
    )


def size_tag(key: str, cfg: Config) -> str:
    a, b, c = effective_sizes(key, cfg)
    return f"n{a}-{b}-{c}"


def data_version(key: str) -> int:
    """Version of the *text* a task emits.

    The feature fingerprint covers how features are computed (template, token budgets, marker
    rule) but not what they are computed from. Without this, editing a generator or a loader
    leaves the cache name identical and the old vectors are silently reused for new text --
    a mismatch with no visible symptom. Bump it whenever the emitted text changes.
    """
    task = REAL_TASKS.get(key)
    if task is not None:
        return task.data_version
    return corpus.DATA_VERSION


def cache_name(backbone_name: str, key: str, split_name: str, cfg: Config) -> str:
    return (
        f"{backbone_name}_{key}_{split_name}_{size_tag(key, cfg)}"
        f"_d{data_version(key)}_{feature_fingerprint(cfg, backbone_name, key)}"
    )


def head_inputs(cfg: Config, fs: FeatureSet, key: str | None = None,
                backbone_name: str = DEFAULT_BACKBONE) -> dict:
    """What shapes a *trained head*: its own shape, and the recipe it was fitted with.

    Separate from the cache identity, because a head can be retrained without touching a feature,
    and put in the run name because `runs/` otherwise holds one file per (backbone, task, sizes):
    a 64-wide head would silently overwrite a 256-wide one, and a run with permuted options would
    overwrite one without.

    ``key`` is what makes the identity say when a head is *not* fitted on one cache.  For a task in
    a ``joint_group`` the vectors are the union of the group's caches, so ``feature_inputs`` becomes
    a hash over every member's recorded inputs, read from their sidecars -- see
    :func:`group_feature_inputs` for why it is derived from disk rather than from ``fs``.
    """
    members = group_members(key) if key is not None else None
    return {
        # The identity of the vectors this head is fitted on.  ``in_dim`` is not enough: ``last``
        # and ``pooled`` both gather d_model numbers, so without this a head trained on pooled
        # features would land on the shipped ``last`` head's filename and overwrite it silently.
        "feature_inputs": (fingerprint_of(group_feature_inputs(members, backbone_name, cfg))
                           if members else fingerprint_of(fs.inputs)),
        "in_dim": fs.feature_dim,
        "d_model": cfg.head_d_model,
        "n_layers": cfg.head_layers,
        "dropout": cfg.head_dropout,
        "slot_embeddings": cfg.head_slot_embeddings,
        "permute_options": cfg.permute_options,
        "targets": cfg.targets,
        "warmup_soft_targets": cfg.warmup_soft_targets,
        # Which teacher produced the targets changes the head, so it belongs in the identity even
        # though `train()` never sees it -- `get_soft_targets` does.
        "teacher_model": cfg.teacher_model,
        "warmup_epochs": cfg.warmup_epochs,
        "rlcd_epochs": cfg.rlcd_epochs,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "weight_decay": cfg.weight_decay,
        "group_size": cfg.group_size,
        "pg_weight": cfg.pg_weight,
        "w_sph": cfg.w_sph,
        "w_rps": cfg.w_rps,
        "seed": cfg.seed,
        "exploration": cfg.exploration,
        "explore_sigma": cfg.explore_sigma,
        # Set only for a shared head, and part of the hash, so adding a language to a group
        # renames the head rather than reusing the old one under the same name.
        **({"joint_group": joint_group(key), "joint_members": members} if members else {}),
    }


def head_fingerprint(cfg: Config, fs: FeatureSet, key: str | None = None,
                     backbone_name: str = DEFAULT_BACKBONE) -> str:
    return hashlib.sha1(
        json.dumps(head_inputs(cfg, fs, key, backbone_name), sort_keys=True,
                   default=str).encode("utf-8")
    ).hexdigest()[:8]


def head_key(key: str) -> str:
    """The name component for the head that serves ``key``.

    A grouped task's head is named for the group, not for the task: a file called
    ``rwkv7-g1d-0.1b_massive_zh_...`` that holds a head fitted on ten languages is a name that
    lies about its contents, and the next person to read it will believe it was trained on zh.
    """
    group = joint_group(key)
    return f"joint_{group}" if group is not None else key


def group_size_tag(members: list[str], cfg: Config, backbone_name: str = DEFAULT_BACKBONE) -> str:
    """``n<train rows>-<dev rows>`` actually present in the group's caches.

    No test count: a shared head has one training set and many test sets, and putting any one of
    them in the name would claim the head was fitted on it.
    """
    return (f"n{group_rows(members, backbone_name, cfg, 'train')}"
            f"-{group_rows(members, backbone_name, cfg, 'dev')}")


def run_name(backbone_name: str, key: str, cfg: Config, fs: FeatureSet) -> str:
    members = group_members(key)
    if members:
        # Each member's ``data_version`` is already inside the hashed identity, through its cache
        # sidecar, so the name needs no ``d`` component of its own here.
        return (f"{backbone_name}_{head_key(key)}_{group_size_tag(members, cfg, backbone_name)}"
                f"_h{head_fingerprint(cfg, fs, key, backbone_name)}")
    return (f"{backbone_name}_{key}_{size_tag(key, cfg)}_d{data_version(key)}"
            f"_h{head_fingerprint(cfg, fs, key, backbone_name)}")


def load_features(backbone_name: str, key: str, split_name: str, cfg: Config) -> FeatureSet:
    return FeatureSet.load(CACHE / cache_name(backbone_name, key, split_name, cfg))


def head_kwargs(cfg: Config, fs: FeatureSet) -> dict:
    return dict(in_dim=fs.feature_dim, d_model=cfg.head_d_model,
                n_layers=cfg.head_layers, dropout=cfg.head_dropout,
                slot_embeddings=cfg.head_slot_embeddings)


def load_heads(
    backbone_name: str, key: str, cfg: Config, n: int = 1
) -> tuple[list[DecisionHead], dict[str, float]]:
    """The heads that answer for ``key``: ``n`` seeds of one recipe, averaged at read time.

    The ensemble is *not* in a head's identity -- each member is an ordinary single-seed head, named
    by its own ``seed``, so training three of them changes no artifact's name and breaks no existing
    table.  What it does change is what a reported number means, which is why the reader that asks
    for ``n > 1`` gets a loud failure instead of a silently smaller ensemble.

    The temperature table comes from seed 0 and is applied to every member before their
    distributions are averaged.  That is a choice, not a measurement: the ensembles in
    ``experiments/results/{seed_ensemble,joint_ensemble}.json`` were scored at T=1, so the headline
    column is measured and the fitted columns of an ensemble are not.
    """
    if n <= 1:
        head, temps = load_head(backbone_name, key, cfg)
        return [head], temps
    heads = []
    temps: dict[str, float] = {}
    for s in range(n):
        try:
            head, t = load_head(backbone_name, key, replace(cfg, seed=s))
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{key}: no head trained at seed {s} of the {n}-seed ensemble ({exc}). "
                f"Run `train --ensemble-seeds {n}` before reading an ensemble number."
            ) from exc
        heads.append(head)
        if s == 0:
            temps = t
    return heads, temps


def load_head(
    backbone_name: str, key: str, cfg: Config
) -> tuple[DecisionHead, dict[str, float]]:
    """The trained head plus its ``bucket -> temperature`` table (empty = never rescale)."""
    fs = load_features(backbone_name, key, "test", cfg)
    name = run_name(backbone_name, key, cfg, fs)
    state_dict = torch.load(RUNS / f"{name}.pt", map_location="cpu", weights_only=True)
    meta = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    # The name already carries this hash, so a mismatch means the file was hand-copied or the
    # hashing changed -- either way the caller is not getting the head it asked for.
    inputs = head_inputs(cfg, fs, key, backbone_name)
    recorded = meta.get("head_inputs")
    if recorded is not None and recorded != inputs:
        differing = [k for k in inputs if recorded.get(k) != inputs[k]]
        print(f"  note: {name}.json records a different recipe ({', '.join(sorted(differing))})")
    head = DecisionHead(**head_kwargs(cfg, fs))
    head.load_state_dict(state_dict)
    head.eval()
    return head, dict(meta["history"]["temperatures"])


def head_temperature(temps: dict[str, float], qtype: int, n_options: int) -> float:
    return float(temps.get(temp_bucket(int(qtype), int(n_options)), 1.0))


