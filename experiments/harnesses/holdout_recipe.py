"""The frozen recipe on a corpus that never entered a sweep: does the gather rule generalise?

**The prediction is stated before the numbers, because that is the whole point of this run.** The
`pooled` gather is pinned per task on twelve tasks, chosen because every task with a large label
space gained (12/12, mean +9.8 pt) while the sixteen smaller ones came out a wash (+0.7).  Those
twelve were also the tasks that *chose* the rule, so the honest question is whether the finding
survives on data that took no part in it.  **If `pooled` does not win here, on a 151-label corpus,
the regime rule is overfit to the tasks that produced it and the pin has to come out.**

That is why the corpus is not in `REAL_TASKS` (being registered is what puts a corpus into every
later sweep) and why nothing about the recipe is tuned here: `Config()` as it stands, the shipped
60 + 30 epochs, the shipped head.  The only variable is the gather rule, and the two arms read
identical rows.

Per-row extraction here is the most expensive in the project bar the biggest banking77-style blocks:
151 option labels make the *block* ~2x banking77's 77, and the block is what costs.  The sizes are
therefore modest, and the extraction is timed so the next person has a measured number instead of
an estimate.

Usage::

    python experiments/harnesses/holdout_recipe.py [corpus] [n_train] [n_dev] [n_test]
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_sweep import run_one  # noqa: E402
from thick_features import splits_for  # noqa: E402

from rwkv_system1.backbone import DecisionBackbone  # noqa: E402
from rwkv_system1.cache_layout import DEFAULT_BACKBONE, ROOT, checkpoint  # noqa: E402
from rwkv_system1.datasets.clinc_oos import load_clinc_oos  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402

SEEDS = (0, 1, 2)
RULES = ("last", "pooled")
#: corpus -> (loader, default sizes).  One entry today; the table is the extension point, and a
#: corpus only belongs in it if it is absent from REAL_TASKS.
CORPORA = {
    "clinc_oos": (load_clinc_oos, (600, 150, 250)),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if argv else "clinc_oos"
    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")
    loader, defaults = CORPORA[name]
    n_train, n_dev, n_test = (int(x) for x in argv[1:4]) if len(argv) > 3 else defaults

    base = Config()
    base.n_train, base.n_dev, base.n_test = n_train, n_dev, n_test
    key = f"{name}_holdout"  # not in the registry: the sizes come from the config, as above
    dest = ROOT / "experiments" / "results" / f"holdout_{name}.json"

    print(f"backbone {DEFAULT_BACKBONE}   corpus {name}   {n_train}/{n_dev}/{n_test} rows   "
          f"seeds {SEEDS}")
    print("PREDICTION (stated before the numbers): `pooled` beats `last`.  A loss or a tie here "
          "refutes\n  the per-task pin, because this corpus took no part in choosing it.\n")

    examples = loader(n_train=n_train, n_dev=n_dev, n_test=n_test)
    n_options = len(examples["train"][0].question.options)

    rows, wall = [], {}
    for rule in RULES:
        print(f"loading backbone {checkpoint(DEFAULT_BACKBONE).name}")
        backbone = DecisionBackbone(checkpoint(DEFAULT_BACKBONE), ctx_tokens=base.ctx_tokens)
        t0 = time.perf_counter()
        splits = splits_for(rule, key, base, examples, backbone)
        wall[rule] = time.perf_counter() - t0
        del backbone
        print(f"\n=== {key} / {rule} ===  {splits['train'].n_options} options, "
              f"{len(splits['train'])} train / {len(splits['test'])} test  "
              f"[{wall[rule]:.0f}s to load or extract]")
        for seed in SEEDS:
            rows.append(run_one(key, rule, {}, seed, base, splits, None))
        del splits

    print(f"\n########## {name}, {n_options} options: pooled - last ##########")
    print(f"{'rule':<10}{'acc':>9}{'ece':>9}{'nll':>9}")
    accs = {}
    for rule in RULES:
        got = [r for r in rows if r["arm"] == rule]
        accs[rule] = [r["t1"]["accuracy"] for r in got]
        print(f"{rule:<10}{st.mean(accs[rule]) * 100:>8.2f}%"
              f"{st.mean([r['t1']['ece'] for r in got]):>9.4f}"
              f"{st.mean([r['t1']['nll'] for r in got]):>9.4f}")
    d = [(p - l) * 100 for l, p in zip(accs["last"], accs["pooled"])]
    verdict = "CONFIRMED" if st.mean(d) > 0 else "REFUTED"
    print(f"\npaired delta by seed: {', '.join(f'{x:+.2f}' for x in d)}pt")
    print(f"mean {st.mean(d):+.2f}pt   pooled wins {sum(x > 0 for x in d)}/{len(d)}   -> {verdict}")
    print("extraction wall clock: " + ", ".join(f"{r} {wall[r] / 60:.1f} min" for r in RULES))
    dest.write_text(
        json.dumps({"corpus": name, "n_options": n_options, "sizes": [n_train, n_dev, n_test],
                    "wall_seconds": wall, "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
