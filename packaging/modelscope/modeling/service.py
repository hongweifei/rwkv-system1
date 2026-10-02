"""The published model: one ``model.pth`` holding a frozen RWKV-7 base and the trained decision heads.

Two ways to use it:

    # 1. from the downloaded folder
    python examples/selfcheck.py                     # must end with "OK"

    from modeling.service import DecisionModel
    model = DecisionModel(".")                       # or just DecisionModel()
    model.answer("laya_needs_human", '{"thread": [...]}')

    # 2. by repo id -- the data is fetched into the SDK cache first, then the same call works
    model = DecisionModel("陆白tuber/rwkv7-system1")
    model.score("banking77", "how do I reset my password")

``model.pth`` is one archive with two sections: ``base`` (the upstream RWKV-7 tensors, exactly as
that file has them) and ``heads`` (one entry per trained head).  The base is normalised by the same
``rwkv7.normalise_native`` the training pipeline uses, and the build verifies that the packed path
and the loose-file path give **bit-identical** tensors and a **zero-drift** forward -- so bundling
is a container change, not a reimplementation of the loader.

Nothing here needs the training repository: ``rwkv7.py``, ``decision.py``, ``backbone.py`` and
``features.py`` are copied into this folder at build time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from .backbone import DecisionBackbone
from .containers import load as load_container
from .decision import QTYPES, DecisionHead, DecisionType, Question, temp_bucket
from .features import markers_for_state
from .rwkv7 import RWKV7

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "config.json"
#: How a marker position becomes the vector a head reads.  A head only works on the layout it was
#: trained against, and the two differ by ~1e2 in vector value -- see ``examples/selfcheck.py``.
RULES = ("last", "pooled")


def head_tensor_identity(state: dict[str, torch.Tensor]) -> str:
    """A short hash over a head's tensors, so a mis-keyed section is caught, not silently loaded."""
    h = hashlib.sha1()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def resolve(source: str | Path) -> Path:
    """A local folder or file, or a ModelScope repo id fetched into the SDK cache.

    Only the data comes from the hub: the runtime modules are the ones this file sits next to, so a
    download cannot quietly replace the code that interprets the weights.
    """
    p = Path(source)
    if p.exists():
        return p.parent if p.is_file() else p
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"{source} is not a local path and modelscope is not installed to fetch it") from exc
    print(f"fetching {source} ...")
    return Path(snapshot_download(str(source)))


class DecisionModel:
    def __init__(self, source: str | Path = ".", ctx_tokens: int = 8192,
                 vocab: str | Path | None = None, ensemble_seeds: int | None = None,
                 verify: bool = True, fmt: str | None = None):
        root = resolve(source)
        self.config_path = root / "config.json"
        if not self.config_path.exists():
            raise FileNotFoundError(f"{self.config_path}: not a decision-model folder "
                                    f"(expected config.json + model.pth)")
        self.manifest = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.tasks_map = self.manifest["tasks"]
        # Both containers hold the same model -- the build compares them tensor by tensor and
        # refuses to ship otherwise -- so the choice is about tooling, not about weights.  Default
        # to safetensors: it cannot carry executable Python at all, which removes the one class of
        # risk that `torch.load` has even with weights_only=True.
        containers = self.manifest.get("containers") or {}
        if not containers:
            raise SystemExit(
                f"{self.config_path} describes the unpacked layout (per-head files); rebuild the "
                f"package with scripts/build_modelscope_pkg.py")
        fmt = fmt or ("safetensors" if "safetensors" in containers else "pth")
        if fmt not in containers:
            raise SystemExit(f"this package has containers {sorted(containers)}, not {fmt!r}")
        bundle_name = containers[fmt]["name"]
        shipped = list(self.manifest["seeds"])
        self.ensemble_seeds = len(shipped) if ensemble_seeds is None else int(ensemble_seeds)
        if not 1 <= self.ensemble_seeds <= len(shipped):
            raise ValueError(f"this model ships {len(shipped)} seed(s) {shipped}; "
                             f"asked for {self.ensemble_seeds}")
        self.seeds = shipped[: self.ensemble_seeds]

        bundle_path = root / bundle_name
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"{bundle_path} is missing. Fetch it with:\n"
                f"  modelscope download --model {self.manifest['repo']} --include {bundle_name} "
                f"--local-dir {root}")
        want = containers[fmt]
        if verify and want.get("sha256"):
            got = _sha256(bundle_path)
            if got != want["sha256"]:
                # Both digests printed in full: truncated to a prefix, the two look identical and the
                # message reads as an accusation against itself.  A file hash is not a secret.
                raise ValueError(
                    f"{bundle_name} is not the file this config was built with.\n"
                    f"  on disk: {got}\n"
                    f"  config:  {want['sha256']}\n"
                    f"Re-download it (or rebuild the package); the heads are keyed to these bytes.")
        # Both containers come back through one function, so the flat-key scheme of safetensors and
        # the nested dict of the .pth cannot each grow their own idea of what a head section is.
        base, heads = load_container(bundle_path)
        self._heads_state = heads
        self.fmt = fmt
        vocab_path = vocab or self._vocab_path(root)
        self.backbone = DecisionBackbone.from_model(
            RWKV7.from_bundle(dict(base), f"{bundle_name}:{self.manifest['base_name']}"),
            vocab=str(vocab_path), ctx_tokens=ctx_tokens)
        self._heads: dict[str, list[DecisionHead]] = {}

    def _vocab_path(self, root: Path) -> Path:
        """The tokenizer table, which ships next to the code rather than inside the weight file.

        The name is read from config: the table is model-independent but not optional, and a
        hard-coded filename here is how a renamed table becomes a FileNotFoundError at whatever the
        least expected moment is.
        """
        return root / "modeling" / self.manifest["vocab_file"]

    def tasks(self) -> list[str]:
        return sorted(self.tasks_map)

    def spec(self, task: str) -> dict:
        try:
            return self.tasks_map[task]
        except KeyError:
            raise SystemExit(f"unknown task {task!r}; this model serves {self.tasks()}") from None

    def question(self, task: str) -> Question:
        s = self.spec(task)
        return Question(name=s["name"], instructions=s["instructions"],
                        type=DecisionType[s["qtype"].upper()], options=list(s["options"]),
                        labels=list(s["labels"]) if s.get("labels") else None)

    def heads(self, task: str) -> list[DecisionHead]:
        """This task's heads, one per served seed, taken from the bundle by key.

        The members are ordinary single-seed heads -- the ensemble exists only at read time -- so
        serving one seed instead of three changes no head's stored identity.
        """
        if task not in self._heads:
            s = self.spec(task)
            keys = s["head_keys"][: len(self.seeds)]
            if len(keys) < len(self.seeds):
                raise SystemExit(f"{task}: config lists {len(s['head_keys'])} head(s) but "
                                 f"{len(self.seeds)} seeds are being served")
            missing = [k for k in keys if k not in self._heads_state]
            if missing:
                raise SystemExit(f"{task}: {missing[0]} is not in the weight file")
            built = []
            for k in keys:
                state = self._heads_state[k]
                record = s.get("head_digests", {}).get(k)
                if record and head_tensor_identity(state) != record:
                    raise SystemExit(f"{task}: head {k} does not hash to the value config recorded "
                                     f"({record}); the weight file and config disagree")
                head = DecisionHead(in_dim=s["in_dim"], d_model=s["d_model"],
                                    n_layers=s["n_layers"], dropout=s["dropout"],
                                    slot_embeddings=s["slot_embeddings"]).eval()
                head.load_state_dict(state)
                built.append(head)
            self._heads[task] = built
        return self._heads[task]

    def head(self, task: str) -> DecisionHead:
        """The seed-0 member alone -- the single-head model."""
        return self.heads(task)[0]

    def features_for(self, state: str, tasks: Iterable[str]) -> dict[str, torch.Tensor]:
        """Marker features for several questions about one state, one prefill per gather rule.

        Questions sharing a rule share the prefill -- that is the reason a recurrent backbone was
        worth choosing over re-running the state per question.
        """
        out: dict[str, torch.Tensor] = {}
        by_rule: dict[str, list[str]] = {}
        for t in tasks:
            by_rule.setdefault(self.spec(t)["gather_rule"], []).append(t)
        for rule, keys in by_rule.items():
            if rule not in RULES:
                raise ValueError(f"{keys}: gather rule {rule!r} is not one this model serves")
            queries = [(self.spec(k)["qtype"], self.spec(k)["instructions"], self.spec(k)["options"])
                       for k in keys]
            for key, feat in zip(keys, markers_for_state(self.backbone, state, queries, rule=rule)):
                out[key] = feat
        return out

    def temperature_for(self, task: str) -> float:
        s = self.spec(task)
        bucket = temp_bucket(QTYPES[s["qtype"]], len(s["options"]))
        return float(s["temperatures"].get(bucket, 1.0))

    def probabilities(self, task: str, state: str, temperature: float | None = None) -> torch.Tensor:
        """The served distribution: the mean of the members' softmaxes, never a mean of logits.

        Logit-averaging keeps the mixture sharp wherever a member is confident, so it is a different
        estimator from the one the ensemble was measured with and would not reproduce this card.
        """
        feats = self.features_for(state, [task])[task]
        qtype = torch.tensor([QTYPES[self.spec(task)["qtype"]]])
        t = self.temperature_for(task) if temperature is None else temperature
        with torch.no_grad():
            members = [DecisionHead.probabilities(h(feats.unsqueeze(0), qtype), t)
                       for h in self.heads(task)]
        return torch.stack(members).mean(0)[0]

    def answer(self, task: str, state: str) -> dict:
        probs = self.probabilities(task, state)
        q = self.question(task)
        label, conf = DecisionHead.point_answer(q, probs)
        return {"task": task, "answer": label, "confidence": conf, "seeds": self.seeds,
                "temperature": self.temperature_for(task),
                "gather_rule": self.spec(task)["gather_rule"],
                "probs": [round(float(p), 6) for p in probs], "labels": q.answer_labels}

    def score(self, task: str, state: str) -> dict:
        """The same decision plus the full distribution, ranked."""
        a = self.answer(task, state)
        a["ranked"] = [(str(l), p) for l, p in
                       sorted(zip(a["labels"], a["probs"]), key=lambda x: -x[1])]
        return a


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()
