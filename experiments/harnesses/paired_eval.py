"""Two ``*_eval.json`` tables, side by side, task by task.

The pipeline writes one eval table per run, keyed by task, and a rollout that changes a recipe
needs the *paired* difference rather than two tables read by eye -- the same reason
``arm_sweep.report`` exists for the head sweeps.  Tasks missing from either file are named and
excluded, never silently averaged over.

Usage::

    python -m experiments.harnesses.paired_eval BASE.json NEW.json [--base-label a] [--new-label b]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ("accuracy", "ece", "nll", "posterior_l1", "posterior_kl", "fitted_temperature")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="paired_eval")
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--new-label", default="new")
    args = ap.parse_args(argv)

    base, new = load(Path(args.base)), load(Path(args.new))
    shared = sorted(set(base) & set(new))
    only_base, only_new = sorted(set(base) - set(new)), sorted(set(new) - set(base))
    print(f"{args.base_label}: {args.base} ({len(base)} tasks)")
    print(f"{args.new_label}: {args.new} ({len(new)} tasks)")
    if only_base:
        print(f"  only in {args.base_label}, excluded: {', '.join(only_base)}")
    if only_new:
        print(f"  only in {args.new_label}, excluded: {', '.join(only_new)}")

    print(f"\n{'task':<20}" + "".join(f"{c:>12}" for c in COLS) + f"{'d_acc':>10}{'d_ece':>10}")
    sums = {c: 0.0 for c in COLS}
    wins = 0
    for key in shared:
        b, n = base[key], new[key]
        for c in COLS:
            sums[c] += n.get(c, float("nan")) - b.get(c, float("nan"))
        d_acc = n["accuracy"] - b["accuracy"]
        wins += d_acc > 0
        print(
            f"{key:<20}"
            + "".join(
                f"{b.get(c, float('nan')):>7.4f}>{n.get(c, float('nan')):<4.4f}" for c in COLS
            )
            + f"{d_acc:>+10.4f}{n['ece'] - b['ece']:>+10.4f}"
        )
    n_tasks = len(shared)
    print(f"\n{'mean delta':<20}" + "".join(f"{sums[c] / n_tasks:>+12.4f}" for c in COLS)
          + f"{sums['accuracy'] / n_tasks:>+10.4f}{sums['ece'] / n_tasks:>+10.4f}")
    print(f"{args.new_label} wins on accuracy: {wins} / {n_tasks}")
    print("\nColumns are base>new. d_acc positive = the new recipe is better; d_ece negative = better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
