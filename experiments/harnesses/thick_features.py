"""Does exposing more of the frozen backbone help?  Three gathers over the same forward pass.

Head capacity and more rows are both measured dead ends exactly where this suite is weakest.  The
head is over-parameterised at every width tried -- 53,889 to 660,481 parameters against 1,125 to
92,400 training signals, and width 512 no better than 256, which is no better than 64 -- and the
data-scaling curves are flat where the accuracy is bad (``xnli_en`` 33.33 -> 33.56% from 50 to 400
rows, ``boolq`` 60.56 -> 59.67% from 125 to 1000, ``go_emotions`` 31.20 -> 34.00% across an 8x
range).  Both of those add capacity *on top of* a fixed input.  This asks whether the input is the
constraint, by gathering more of the computation the frozen backbone already performs:

    last        the post-``ln_out`` activation at each option's terminator, 768-d -- what ships
    pooled      that activation averaged over the option's own token span, 768-d
    two_layer   the terminator activation from the last block concatenated with the one from block
                ``n_layer // 2``, both post-``ln_out``, 1536-d

All three read **one** forward pass, so the alternatives cost cache width rather than compute.  That
is the reason to try this before the 0.4B backbone, whose 1.8 s/example is four times this box's
0.1B rate.

The ``last`` arm is the shipped feature, so on a task whose cache already exists it is read from
disk rather than re-derived -- same rows, same vectors, and the arm is free.  Each variant gets its
own cache name through ``cfg.feature_rule``, which is part of the feature identity, so nothing here
can be mistaken for the shipped vectors.

One task is a screen, not a verdict: a variant has to clear this project's seed sd (0.6-1.9pt on
accuracy) before it is worth an 8-hour re-extraction of all 34 tasks.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_sweep import report  # noqa: E402

from rwkv_system1.backbone import DecisionBackbone  # noqa: E402
from rwkv_system1.cache_layout import (CACHE, DEFAULT_BACKBONE, ROOT, cache_name, checkpoint,
                                       data_version, feature_inputs, head_kwargs)  # noqa: E402
from rwkv_system1.decision import DecisionHead  # noqa: E402
from rwkv_system1.examples import build_examples  # noqa: E402
from rwkv_system1.features import GATHER_RULES, FeatureSet, extract  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.targets import get_soft_targets  # noqa: E402
from rwkv_system1.train import evaluate, train  # noqa: E402


SEEDS = (0, 1, 2)
#: A 60-option task in a well-supported language: the option block is the discriminator, the
#: accuracy has room (32.5%), and it is the cheap end of the MASSIVE family (800 rows).
TASKS = ["massive_zh"]
REFERENCE = "last"


def splits_for(rule: str, key: str, cfg: Config, examples: dict, backbone,
               data_version_of: int | None = None) -> dict:
    """The three splits under one gather rule, from cache when it is there.

    ``cfg.feature_rule`` carries the rule, so the cache name and the stored identity both move with
    it -- which is why this needs no private caching of its own.

    ``data_version_of`` overrides the registry lookup for the *identity* record only.  A diagnostic
    arm built by a loader kwarg holds a split whose content differs from the registered task's at
    the same sizes, which is exactly the case the registry's ``data_version`` exists to name -- but
    it is keyed by task name and a diagnostic arm has none, so the caller has to say.
    """
    cfg_r = replace(cfg, feature_rule=rule)
    version = data_version(key) if data_version_of is None else data_version_of

    def dest(split: str) -> Path:
        return CACHE / cache_name(DEFAULT_BACKBONE, key, split, cfg_r)

    out = {}
    for split in ("train", "dev", "test"):
        path = dest(split)
        if Path(str(path) + ".pt").exists():
            out[split] = FeatureSet.load(path)
            continue
        fs = extract(
            backbone,
            [e.state for e in examples[split]],
            [(e.question.type.label, e.question.instructions, e.question.options)
             for e in examples[split]],
            [e.answer for e in examples[split]],
            [e.true_probs for e in examples[split]],
            inputs=feature_inputs(cfg_r, DEFAULT_BACKBONE, version),
            rule=rule,
        )
        fs.save(path)
        out[split] = fs
    return out


def main(argv: list[str]) -> int:
    tasks = [t for t in (argv[0].split(",") if argv else []) if t] or TASKS
    base = Config()
    dest = ROOT / "experiments" / "results" / "thick_features.json"
    print(f"backbone {DEFAULT_BACKBONE}   tasks {tasks}   seeds {SEEDS}   "
          f"rules {sorted(GATHER_RULES)}   reference {REFERENCE!r}\n")

    backbone = None
    rows: list[dict] = []
    for key in tasks:
        examples = build_examples(base, [key])[key]
        per_rule = {}
        for rule in sorted(GATHER_RULES):
            cfg_r = replace(base, feature_rule=rule)
            path = CACHE / cache_name(DEFAULT_BACKBONE, key, "test", cfg_r)
            if not Path(str(path) + ".pt").exists() and backbone is None:
                print(f"loading backbone {checkpoint(DEFAULT_BACKBONE).name} for the missing arms")
                backbone = DecisionBackbone(checkpoint(DEFAULT_BACKBONE),
                                            ctx_tokens=base.ctx_tokens)
            per_rule[rule] = splits_for(rule, key, base, examples, backbone)

        for rule, splits in per_rule.items():
            soft = get_soft_targets(DEFAULT_BACKBONE, key, replace(base, feature_rule=rule),
                                    splits["train"])
            print(f"\n=== {key} / {rule} ===  in_dim {splits['train'].feature_dim}, "
                  f"{len(splits['train'])} train / {len(splits['test'])} test")
            for seed in SEEDS:
                torch.manual_seed(seed)
                head = DecisionHead(**head_kwargs(base, splits["train"]))
                history = train(
                    head, splits["train"], splits["dev"], soft_targets=soft,
                    warmup_epochs=base.warmup_epochs, rlcd_epochs=base.rlcd_epochs, lr=base.lr,
                    batch_size=base.batch_size, weight_decay=base.weight_decay,
                    group_size=base.group_size, pg_weight=base.pg_weight, w_sph=base.w_sph,
                    w_rps=base.w_rps, permute_options=base.permute_options, seed=seed,
                    log=lambda *a: None,
                )
                te = splits["test"]
                raw = evaluate(head, te, temperature=1.0)
                temp = history.temperature_for(int(te.qtype), int(te.n_options))
                rows.append({
                    "task": key, "arm": rule, "seed": seed, "n_test": len(te),
                    "in_dim": te.feature_dim,
                    "t1": {m: raw[m] for m in ("accuracy", "ece", "nll")},
                    "fit": {m: evaluate(head, te, temperature=temp)[m]
                            for m in ("accuracy", "ece", "nll")},
                })
                print(f"  {rule:<10}seed {seed}  acc {raw['accuracy']:.4f}  ece {raw['ece']:.4f}  "
                      f"nll {raw['nll']:.4f}  T {temp:.2f}", flush=True)
        del per_rule, examples

    report(rows, tasks, sorted(GATHER_RULES), reference=REFERENCE)
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
