"""Does `pooled`'s gain depend on how many options there are?  One task, several label spaces.

The rollout read `pooled` as +2.04 pt over 19 paired tasks (11/19, t = +1.66) -- a wash -- but the
split by option count is exhaustive rather than cherry-picked: **every** task with a large label
space gained (`banking77` 77 options +14.60, `massive_en` 60 +9.60, `massive_zh` 60 +12.44,
`go_emotions` 28 +4.00, 4/4, mean +10.2) while the small-option tasks came out flat (+0.68 over 15,
8/15) with a few real losses (`prompt_injections` -3.45 and its ECE 0.0277 -> 0.0951, `laya_urgency`
-4.25).  No continuous covariate explains it: mean option *span* correlates r = +0.238 and flips sign
inside one band (the three `laya_*` tasks all span 13-14 tokens and read +6.00 / +10.00 / -4.25).

Cross-task grouping cannot separate "large label space" from "this task happens to like averaging",
so this shrinks the label space **within one task**: `massive_en` with `n_intents` of its 60 intents,
same corpus, same texts, same recipe, rows whose label fell outside the subset dropped.  Same two
rules, three seeds each, and the statistic is the paired `pooled - last` delta next to the +9.60 the
rollout measured at 60 options.  If the gain tracks the label space it should collapse toward 0; if
it stays, the 4/4 was luck and `pooled` closes for good.

Rows are 800/200/300 rather than the registry's 1000/400/500 so that the cache name cannot collide
with the real `massive_en` cache -- the sizes are in the name, and a diagnostic arm must never be
able to overwrite a benchmark artifact.  Both rules of an arm read the identical rows, so the delta
is paired; only the cross-arm comparison carries the row-count difference.  `data_version` is passed
explicitly because a loader kwarg changes the split's *content* at the same sizes, which is what that
field exists to name, and an unregistered key would otherwise fall back to the synthetic corpora's.

Usage::

    python experiments/harnesses/option_count.py [10,30]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_sweep import run_one  # noqa: E402
from thick_features import splits_for  # noqa: E402

from rwkv_system1.backbone import DecisionBackbone  # noqa: E402
from rwkv_system1.cache_layout import CACHE, DEFAULT_BACKBONE, ROOT, cache_name, checkpoint  # noqa: E402
from rwkv_system1.datasets.massive import load_massive  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.targets import get_soft_targets  # noqa: E402
from rwkv_system1.task_registry import MASSIVE_DATA_VERSION  # noqa: E402

SEEDS = (0, 1, 2)
RULES = ("last", "pooled")
#: The rollout's measured `pooled - last` at the full 60 intents, for the side-by-side.
REFERENCE_60 = 9.60
N_TRAIN, N_DEV, N_TEST = 800, 200, 300


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    counts = [int(c) for c in (argv[0].split(",") if argv else ["10"]) if c]
    base = Config()
    base.n_train, base.n_dev, base.n_test = N_TRAIN, N_DEV, N_TEST
    dest = ROOT / "experiments" / "results" / "option_count.json"
    print(f"backbone {DEFAULT_BACKBONE}   lang en   intents {counts}   {N_TRAIN}/{N_DEV}/{N_TEST} "
          f"rows   seeds {SEEDS}\n")

    backbone = None
    rows: list[dict] = []
    widths: dict[str, int] = {}
    for count in counts:
        # Not a registered key: the sizes above then come from the config, which is what keeps this
        # clear of the real task's cache.
        key = f"massive_en_i{count}"
        examples = load_massive(lang="en", n_train=N_TRAIN, n_dev=N_DEV, n_test=N_TEST,
                                n_intents=count)
        for rule in RULES:
            cfg_r = replace(base, feature_rule=rule)
            missing = [s for s in ("train", "dev", "test")
                       if not Path(str(CACHE / cache_name(DEFAULT_BACKBONE, key, s, cfg_r))
                                   + ".pt").exists()]
            if missing and backbone is None:
                print(f"loading backbone {checkpoint(DEFAULT_BACKBONE).name} for the missing splits")
                backbone = DecisionBackbone(checkpoint(DEFAULT_BACKBONE),
                                            ctx_tokens=base.ctx_tokens)
            splits = splits_for(rule, key, base, examples, backbone,
                                data_version_of=MASSIVE_DATA_VERSION)
            widths[key] = splits["train"].n_options
            soft = get_soft_targets(DEFAULT_BACKBONE, key, replace(base, feature_rule=rule),
                                    splits["train"])
            print(f"\n=== {key} / {rule} ===  {splits['train'].n_options} options, "
                  f"{len(splits['train'])} train / {len(splits['test'])} test")
            for seed in SEEDS:
                rows.append(run_one(key, rule, {}, seed, base, splits, soft))
        del examples

    print(f"\n########## option count vs the pooled - last delta ##########")
    print(f"{'task':<22}{'opts':>5}{'acc last':>10}{'acc pooled':>12}{'d_acc':>9}{'wins':>7}")
    out: dict = {"reference_60_intents": REFERENCE_60, "rows": rows}
    for count in counts:
        key = f"massive_en_i{count}"
        cell = {}
        for rule in RULES:
            got = [r for r in rows if r["task"] == key and r["arm"] == rule]
            cell[rule] = (st.mean([r["t1"]["accuracy"] for r in got]),
                          st.mean([r["t1"]["ece"] for r in got]),
                          [r["t1"]["accuracy"] for r in got])
        d = [(p - l) * 100 for l, p in zip(cell["last"][2], cell["pooled"][2])]
        print(f"{key:<22}{widths[key]:>5}{cell['last'][0] * 100:>9.2f}%"
              f"{cell['pooled'][0] * 100:>11.2f}%{st.mean(d):>+9.2f}"
              f"{sum(x > 0 for x in d):>4}/{len(d)}")
        out[key] = {"n_options": widths[key],
                    **{rule: {"accuracy": cell[rule][0], "ece": cell[rule][1]} for rule in RULES},
                    "delta_pt": st.mean(d)}
    print(f"\n{'massive_en (registry)':<22}{60:>5}{42.80:>9.2f}%{52.40:>11.2f}%"
          f"{REFERENCE_60:>+9.2f}{'':>7}")
    print("\nIf the delta tracks the label space it should fall toward 0 as the option count drops;")
    print("acc last/pooled are 3-seed means at T=1 and d_acc is the mean of the paired per-seed deltas.")
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
