"""The online teacher: a distribution over a task's option set, cached and verified like a feature.

Why a teacher at all, given the ablation: on a task that ships hard labels the proper-score stage is
a wash, because against a one-hot target the log term *is* cross-entropy and the stage has nothing
to calibrate to (measured 2026-09-23: on ``ag_news`` the shipped recipe is +0.37±0.37pt accuracy and
+2.28±0.83pt worse top-label ECE than plain cross-entropy).  A teacher's distribution is the only
way to give those tasks a soft target, and the plumbing already existed -- ``FeatureSet.true_probs``
is documented as the teacher channel, ``proper_reward`` is faithful to whatever distribution it is
handed, and ``get_soft_targets`` already had a cached-targets mode.

MEASURED on 20 test rows per task, so one row of this table carries ~11pt of standard error; read
the pattern, not the point:

    task              k   teacher   student   teacher's normalised entropy
    massive_en       60     84.2%     15.0%    0.17
    xnli_en           3     65.0%     40.0%    0.32
    sst5              5     60.0%     25.0%    0.44
    emotion           6     70.0%     55.0%    0.43
    laya_urgency      4     75.0%     65.0%    0.65
    go_emotions      28     50.0%     40.0%    0.64
    ag_news           4     90.0%     95.0%    0.33

The teacher wins where the student is weak and *loses* where the student is already at 89-95%, so
this is a per-task decision, not a global improvement.  It spreads mass on the narrow questions and
collapses toward a point mass on the wide ones -- banking77's 77 labels came back as one option at
100 and the rest at 0 -- which is why the wide tasks gain from it as a labeller rather than as a
source of dark knowledge.  Read this table as a *screen* for which tasks the teacher is stronger on,
not as a prediction that it helps: the decomposition below shows the strength does not transfer.

**MEASURED 2026-09-24, and the verdict is negative: the teacher's two halves are each worth less
than the loss it causes.**  Three seeds on sst5 (600-row test), the true labels as the referee
throughout, and dev/test never pseudo-labelled:

    arm                                       accuracy   top-label ECE
    the shipped recipe on the true labels      35.50%     0.109
    the teacher's argmax as the labels         32.83%     0.171
    the teacher's distribution, stage 2 only   32.17%     0.049
    the teacher's distribution, stages 1+2     33.67%     0.047

Split into the two things a teacher can contribute:

* **its answer** costs 2.67±0.33pt, all three seeds worse.  It agrees with the labels on 54.6% of
  the very rows it would teach from, so fitting its argmax means fitting a noisier target than the
  label.  It also makes the head *over*-confident (ECE 0.109 -> 0.171), which is what happens when a
  54.6%-accurate target is fitted to 100% confidence.
* **its hesitation** is worth 0.84±0.83pt on top of that (33.67% against 32.83%).  The error bar is
  larger than the effect, so the distribution's shape carries nothing measurable.  It does buy
  calibration -- 0.109 -> 0.047 at T=1, i.e. it acts as label smoothing -- but most of that goes
  away once the temperature is fitted (0.053 against 0.048), so it is not a stable win.

An earlier single-seed run reported the same regression against 37.17%; that was the luckiest of the
three baseline seeds (37.17 / 34.00 / 35.33), which is why the first answer had to be "suggestive"
and this one does not.

So on a task with clean labels and a frozen backbone this direction is closed, and structurally so
rather than for want of tuning: the teacher's advantage lives in the backbone (84.2% against the
student's 15.0% on ``massive_en``) and distillation changes the target, not the features.  Worth
revisiting where the *labels* are noisy or incomplete -- there the teacher is a source of
information rather than a worse copy of it -- and nowhere else short of training the backbone, which
needs a GPU this box does not have.  The code is kept because that case is not this project's, and
because everything here is opt-in and off by default.

**The unfavourable prior, and why the targets are still temperature-fitted before use.**  A proper
rule is faithful, not corrective: it reproduces the distribution it is handed, miscalibration
included.  That was measured on ``crossfit`` -- targets from a same-architecture fold model came
back at 0.884 confidence where the analytic posterior is 0.969, and the student inherited exactly
that shape.  So the raw scores are never used directly; :func:`fit_teacher_temperature` rescales
them against held-out labels first, with the same one-standard-error guard the head's own fit gets.

``enable_thinking: false`` matters: with the model's reasoning on, one 4-option call took 26.6 s and
1388 reasoning tokens; off, the same call is 1.5 s and 36 tokens and the answer is graded rather
than a bare argmax.  The prompt asks for 0-100 per option rather than probabilities summing to 1,
because then the model does no arithmetic, which is what makes it hold up at 77 options.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .cache_layout import CACHE, data_version, load_features, size_tag
from .features import read_sidecar, stored_mismatch
from .rlcd import fit_temperature

if TYPE_CHECKING:
    from .pipeline.config import Config


ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-flash"
#: Bump when ``PROMPT`` changes: the cache name carries it, so a reworded prompt cannot silently
#: reuse distributions the old wording produced.
PROMPT_VERSION = 1
ENV_KEY = "DASHSCOPE_API_KEY"

#: Sent verbatim and hashed into the cache name from this same dict, so the record of what was
#: asked cannot drift from what is sent.  ``enable_thinking`` off is worth 15x on latency and is
#: what makes the answer graded rather than a bare argmax.
REQUEST_OPTIONS = {
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
    "enable_thinking": False,
}

TIMEOUT_SECONDS = 180
ATTEMPTS = 4
#: Linear backoff, multiplied by the attempt number.
BACKOFF_SECONDS = 2.0

PROMPT = """You are answering one multiple-choice question. Read the situation, then score every \
option for how likely it is to be the correct answer.

Situation:
{state}

Question: {instructions}

Options, numbered from 0:
{options}

Reply with a single JSON object and nothing else: the keys are option numbers as strings and the \
values are integers from 0 to 100, where 100 means certainly correct and 0 means impossible. \
Score all {k} options, and do not make the scores sum to anything in particular."""


@dataclass
class TeacherTargets:
    """``(N, K)`` distributions plus the row identity they belong to.

    Tensors to a ``.pt`` and strings to a sidecar JSON, exactly as ``FeatureSet`` does, so nothing
    on disk ever needs unpickling.
    """

    probs: torch.Tensor
    states: list[str]
    options: list[str]
    inputs: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.probs.shape[0]

    def save(self, path: str | Path) -> None:
        torch.save({"probs": self.probs}, str(path) + ".pt")
        Path(str(path) + ".json").write_text(
            json.dumps({"states": self.states, "options": self.options,
                        "inputs": self.inputs, "stats": self.stats}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeacherTargets":
        tensors = torch.load(str(path) + ".pt", map_location="cpu", weights_only=True)
        strings = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
        return cls(probs=tensors["probs"], **strings)


def api_key() -> str:
    key = os.environ.get(ENV_KEY)
    if not key:
        raise SystemExit(
            f"{ENV_KEY} is not set.  It lives in the gitignored .env; load it with\n"
            f"  set -a; . ./.env; set +a"
        )
    return key


def teacher_inputs(model: str) -> dict:
    """The *how* of a cached distribution, in one dict to compare wholesale.

    Deliberately excludes the task and its sizes: those are in the file name, exactly as
    ``feature_fingerprint`` covers the template and the token budgets while ``cache_name`` carries
    the task and the sizes.  A changed corpus therefore lands on a different file rather than on a
    reused one.
    """
    return {
        "endpoint": ENDPOINT,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt": PROMPT,
        "request": REQUEST_OPTIONS,
    }


def teacher_path(model: str, key: str, split: str, cfg: Config) -> Path:
    """Where one split's distributions live.  Extensionless, like a feature cache: append the
    suffix.  (``with_suffix`` cannot be used here either -- the model name contains a dot, the same
    trap ``FeatureSet.save`` documents for ``rwkv7-g1d-0.1b``.)"""
    digest = hashlib.sha1(
        json.dumps(teacher_inputs(model), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return CACHE / (f"teacher_{model}_{key}_{split}_{size_tag(key, cfg)}"
                    f"_d{data_version(key)}_{digest}")


def teacher_mismatch(path: str | Path, inputs: dict, states: list[str],
                     options: list[str]) -> str:
    """Why a cached distribution no longer matches this task; '' if it does."""
    stored = read_sidecar(path)
    if stored is None:
        return "no sidecar"
    return stored_mismatch(stored, inputs, options, states)


def _ask(state: str, instructions: str, options: list[str], model: str,
          key: str) -> tuple[list[float], dict] | None:
    """One distribution as 0-100 scores normalised to sum to 1.  None if it never parses."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(
            state=state, instructions=instructions, k=len(options),
            options="\n".join(f"{i}. {o}" for i, o in enumerate(options)))}],
        **REQUEST_OPTIONS,
    }
    payload = json.dumps(body).encode("utf-8")
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    for attempt in range(ATTEMPTS):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(
                urllib.request.Request(ENDPOINT, data=payload, headers=headers),
                timeout=TIMEOUT_SECONDS,
            ) as r:
                d = json.load(r)
            elapsed = time.perf_counter() - t0
            scores = json.loads(d["choices"][0]["message"]["content"])
            v = [max(0.0, float(scores.get(str(i), 0.0))) for i in range(len(options))]
            total = sum(v)
            if total <= 0:
                raise ValueError("all scores zero")
            u = d["usage"]
            return [x / total for x in v], {
                "seconds": elapsed, "prompt_tokens": u["prompt_tokens"],
                "completion_tokens": u["completion_tokens"],
            }
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError,
                json.JSONDecodeError, TimeoutError) as exc:
            if attempt == ATTEMPTS - 1:
                print(f"    teacher gave up on a row after {ATTEMPTS} tries: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                return None
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
    return None


def fetch(
    states: list[str],
    instructions: str,
    options: list[str],
    model: str,
    workers: int,
    partial: Path | None = None,
    log=print,
) -> tuple[torch.Tensor, dict]:
    """Distributions for ``states``, resuming from ``partial`` if it exists.

    The partial file is JSONL of ``{"row": i, "probs": [...]}``.  A run of this shape is hours of
    paid calls on a box that has already taken the whole machine down once, so losing finished rows
    to a crash is not acceptable.

    ``model`` and ``workers`` are required rather than defaulted: the model is part of the cache
    identity and the worker count has exactly one default, in ``Config``.
    """
    key = api_key()  # once, before any thread starts, so a missing key fails immediately
    done: dict[int, list[float]] = {}
    if partial is not None and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["row"]] = rec["probs"]
        log(f"  teacher: resuming, {len(done)}/{len(states)} rows already on disk")

    todo = [i for i in range(len(states)) if i not in done]
    stats: dict = {"seconds": [], "prompt_tokens": 0, "completion_tokens": 0, "failed": 0}
    handle = partial.open("a", encoding="utf-8") if partial is not None else None
    try:
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(_ask, states[i], instructions, options, model, key): i
                       for i in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                i = futures[fut]
                got = fut.result()
                if got is None:
                    stats["failed"] += 1
                    continue
                probs, use = got
                done[i] = probs
                stats["seconds"].append(use["seconds"])
                stats["prompt_tokens"] += use["prompt_tokens"]
                stats["completion_tokens"] += use["completion_tokens"]
                if handle is not None:
                    handle.write(json.dumps({"row": i, "probs": probs}) + "\n")
                    handle.flush()
                if n % 50 == 0 or n == len(todo):
                    log(f"  teacher: {n}/{len(todo)} asked, {stats['failed']} failed, "
                        f"{sum(stats['seconds']) / max(1, len(stats['seconds'])):.1f}s/call")
    finally:
        if handle is not None:
            handle.close()

    missing = [i for i in range(len(states)) if i not in done]
    if missing:
        raise SystemExit(
            f"teacher: {len(missing)} of {len(states)} rows have no distribution "
            f"(first: {missing[:5]}); the partial file is kept, re-run to retry only those"
        )
    probs = torch.tensor([done[i] for i in range(len(states))], dtype=torch.float32)
    if probs.shape != (len(states), len(options)):
        raise RuntimeError(f"teacher returned {tuple(probs.shape)} for "
                           f"{len(states)} rows x {len(options)} options")
    stats["mean_seconds"] = (sum(stats["seconds"]) / len(stats["seconds"])
                             if stats["seconds"] else float("nan"))
    if partial is not None and partial.exists():
        partial.unlink()
    return probs, stats


def fit_teacher_temperature(dev_probs: torch.Tensor, dev_labels: torch.Tensor,
                            qtype: int) -> tuple[float, dict[str, float]]:
    """One scalar that makes the teacher's distribution match the labels it is scored against.

    Needed because a proper rule is faithful: handed a teacher that says 0.9 when it is right 0.7 of
    the time, the student learns to be wrong the same way.  Delegates to the head's own
    :func:`rwkv_system1.rlcd.fit_temperature`, which buys two things -- one grid instead of two, and
    its one-standard-error guard.

    MEASURED on sst5, so the guard's effect is not assumed: the old unguarded coarse grid picked its
    own edge (T=3.0, the largest value it held), the shared fine grid's argmin is 2.5, and the guard
    then keeps T=1.5.  That change did **not** move the task's accuracy (31.67% either way), so the
    guard is a correctness fix and *not* the explanation for that task's regression -- a flattened
    target is simply a weaker target in a stage whose log term exists to sharpen.
    """
    return fit_temperature(torch.log(dev_probs.clamp_min(1e-12)), dev_labels, int(qtype))


def targets_for_split(
    key: str,
    split: str,
    examples,
    cfg: Config,
    model: str,
    workers: int,
    force: bool = False,
    log=print,
) -> TeacherTargets:
    """Cached distribution for one split, fetching only what is missing.

    ``examples`` must be that split's rendered examples in cache order -- they are read only on a
    miss, so a task whose distributions are already on disk costs no corpus load.
    """
    states = [e.state for e in examples]
    question = examples[0].question
    options = list(question.options)
    inputs = teacher_inputs(model)
    dest = teacher_path(model, key, split, cfg)
    # `dest` carries no extension, so the existence check has to be on the `.pt`: checking
    # `dest.exists()` here was a real bug -- always False, so the distributions were written and
    # never read back, and every re-run re-paid for every call.
    if Path(str(dest) + ".pt").exists() and not force:
        why = teacher_mismatch(dest, inputs, states, options)
        if not why:
            got = TeacherTargets.load(dest)
            log(f"  teacher: {dest.name} cached, skipping")
            return got
        log(f"  teacher: {dest.name} no longer matches ({why}), re-asking")
    log(f"  teacher: {model}, {len(states)} rows, {len(options)} options")
    probs, stats = fetch(states, question.instructions, options, model=model, workers=workers,
                         partial=Path(str(dest) + ".partial"), log=log)
    out = TeacherTargets(probs=probs, states=states, options=options, inputs=inputs, stats=stats)
    out.save(dest)
    log(f"  teacher: -> {tuple(out.probs.shape)}, {stats['prompt_tokens']} prompt + "
        f"{stats['completion_tokens']} completion tokens, {stats['mean_seconds']:.1f}s/call")
    return out


def soft_targets(
    key: str,
    splits: dict,
    cfg: Config,
    backbone_name: str,
    force: bool = False,
    log=print,
) -> torch.Tensor:
    """The teacher's distribution for ``train``, temperature-fitted on ``dev``.

    ``splits`` is that task's rendered examples -- ``{"train": [...], "dev": [...]}`` -- which is
    what carries the question instructions the prompt needs.  The temperature is fitted on dev,
    never on the rows being trained, so it cannot smuggle the labels into their own target.
    """
    ask = dict(cfg=cfg, model=cfg.teacher_model, workers=cfg.teacher_workers, force=force, log=log)
    tr = targets_for_split(key, "train", splits["train"], **ask)
    dev = targets_for_split(key, "dev", splits["dev"], **ask)
    dev_fs = load_features(backbone_name, key, "dev", cfg)
    t, info = fit_teacher_temperature(dev.probs, dev_fs.labels, int(dev_fs.qtype))
    log(f"  soft targets: {key} teacher {cfg.teacher_model}, mean max prob "
        f"{float(tr.probs.max(-1).values.mean()):.3f}, temperature {t} "
        f"(dev NLL {info['dev_nll']:.4f}, grid argmin {info['argmin_t']:.2f} "
        f"at {info['argmin_nll']:.4f})")
    return torch.softmax(torch.log(tr.probs.clamp_min(1e-12)) / t, dim=-1)
