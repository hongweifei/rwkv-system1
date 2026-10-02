"""Live decisions: one prefill per state, every question answered off it."""

from __future__ import annotations

import time

import torch

from ...backbone import DecisionBackbone
from ...cache_layout import checkpoint, head_temperature, load_heads, resolve_rule
from ...decision import QTYPES, DecisionHead
from ...features import markers_for_state
from ...examples import build_examples
from ...train import mean_probabilities
from ..config import Config

#: The three corpora a live decision is demonstrated on: real dialogue states, one shared
#: prefill, every question answered off it.
LIVE_QUESTIONS = ["laya_urgency", "laya_category", "laya_needs_human"]


def cmd_predict(cfg: Config, args) -> None:
    ckpt = checkpoint(args.backbone)
    backbone = DecisionBackbone(ckpt, ctx_tokens=cfg.ctx_tokens)
    n_seeds = getattr(args, "ensemble_seeds", 1)
    if n_seeds > 1:
        print(f"predicting with the average of {n_seeds} seeds per question "
              f"(seeds 0..{n_seeds - 1})")
    heads = {k: load_heads(args.backbone, k, cfg, n_seeds) for k in LIVE_QUESTIONS}
    # Only the corpora we are about to ask about: loading all eight in one process is the path
    # that took a SIGSEGV partway through, and the states shown below come from a real dialogue
    # either way.
    data = build_examples(cfg, LIVE_QUESTIONS)
    examples = data[LIVE_QUESTIONS[0]]["test"][: args.n]
    # Each task has exactly one question schema, so any example carries it.
    schemas = {k: data[k]["test"][0].question for k in LIVE_QUESTIONS}
    qs = {k: (q.type.label, q.instructions, q.options) for k, q in schemas.items()}
    # A head only reads the feature layout it was trained on, so the gather rule is per question
    # here exactly as it is per task in `extract`.  Questions sharing a rule share one prefill; the
    # three demo questions are all `last`, so today that is one pass.
    rules = {k: resolve_rule(cfg.feature_rule, k) for k in LIVE_QUESTIONS}
    by_rule: dict[str, list[str]] = {}
    for k, rule in rules.items():
        by_rule.setdefault(rule, []).append(k)

    print("live decisions: one prefill per state, every question answered off it\n")
    for e in examples:
        t0 = time.perf_counter()
        feats: dict[str, torch.Tensor] = {}
        for rule, keys in by_rule.items():
            for key, f in zip(keys, markers_for_state(backbone, e.state, [qs[k] for k in keys],
                                                     rule=rule)):
                feats[key] = f
        elapsed = (time.perf_counter() - t0) * 1000

        print("=" * 78)
        print(f"state : {e.state}")
        print(f"        1 prefill + {len(qs)} query blocks in {elapsed:.0f} ms")
        for key, q in schemas.items():
            f = feats[key]
            heads_for_q, temps = heads[key]
            temp = head_temperature(temps, QTYPES[q.type.label], len(q.options))
            qtype_t = torch.tensor([QTYPES[q.type.label]])
            with torch.no_grad():
                probs = mean_probabilities([
                    DecisionHead.probabilities(h(f.unsqueeze(0), qtype_t), temp)
                    for h in heads_for_q])[0]
            answer, conf = DecisionHead.point_answer(q, probs)
            print(f"  {key:<15} -> {str(answer):<12} confidence {conf:.3f}  (T={temp:.2f})")
            print(f"      probabilities  {[round(float(p), 3) for p in probs]}")
