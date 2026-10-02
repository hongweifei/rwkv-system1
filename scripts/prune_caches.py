"""List and retire superseded feature caches.  Run: python scripts/prune_caches.py [--prune]

A cache's name carries a fingerprint of everything that shapes the vectors (prompt template,
token budgets, marker rule, data version), so changing any of those leaves the old files behind
under a name nothing will ever look up again.  They are dead weight at ~50 MB per task set.

`--prune` deletes an old set **only** when a current-fingerprint file with the same task, split
and sizes exists AND agrees tensor-for-tensor, label-for-label, option-text-for-option-text.  If
any of that cannot be shown, nothing under that fingerprint is deleted and the reason is printed
-- the old file is then the only record of that extraction.  Measured 2026-09-23: the twelve sets
replaced by the `max_suffix_tokens` removal came out byte-identical (max|delta| = 0.000e+00), so
deleting them lost nothing; that is also the reason this guard is cheap to satisfy rather than a
formality.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rwkv_system1.cache_layout import CACHE, DEFAULT_BACKBONE, feature_fingerprint
from rwkv_system1.pipeline.config import Config


def group(cache_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = collections.defaultdict(list)
    for p in sorted(cache_dir.glob("*.pt")):
        out[p.name.rsplit("_", 1)[1][:-3]].append(p)
    return dict(out)


def size_of(paths: list[Path]) -> float:
    total = 0
    for p in paths:
        for q in (p, p.with_suffix(".json")):
            total += os.path.getsize(q) if q.exists() else 0
    return total / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prune", action="store_true", help="delete proven-superseded sets")
    args = ap.parse_args()

    current = feature_fingerprint(Config(), DEFAULT_BACKBONE)
    sets = group(CACHE)
    print(f"{CACHE}")
    for fp, paths in sorted(sets.items(), key=lambda kv: -size_of(kv[1])):
        mark = "current" if fp == current else "STALE  "
        print(f"  {mark} {fp}  {len(paths):>3} caches  {size_of(paths):>7.1f} MB")
    orphans = [p.name for p in CACHE.glob("*.pt") if not p.with_suffix(".json").exists()]
    orphans += [p.name for p in CACHE.glob("*.json") if not p.with_suffix(".pt").exists()]
    if orphans:
        print(f"  incomplete pairs (a write was interrupted): {orphans}")

    stale = {fp: paths for fp, paths in sets.items() if fp != current}
    if not stale:
        print("\nnothing stale")
        return 0
    if not args.prune:
        print(f"\n{sum(len(v) for v in stale.values())} stale sets, "
              f"{sum(size_of(v) for v in stale.values()):.1f} MB -- rerun with --prune to delete "
              f"the ones provably identical to their replacement")
        return 0

    doomed, kept = [], []
    for fp, paths in stale.items():
        for old in paths:
            stem = old.name[: -len(f"_{fp}.pt")]
            new = CACHE / f"{stem}_{current}.pt"
            if not new.exists():
                kept.append((old.name, "no current-fingerprint counterpart"))
                continue
            a = torch.load(old, map_location="cpu", weights_only=True)
            b = torch.load(new, map_location="cpu", weights_only=True)
            if tuple(a["features"].shape) != tuple(b["features"].shape):
                kept.append((old.name, "shapes differ"))
                continue
            delta = float((a["features"].float() - b["features"].float()).abs().max())
            same_labels = bool(torch.equal(a["labels"], b["labels"]))
            same_text = (json.loads(old.with_suffix(".json").read_text(encoding="utf-8"))["options"]
                         == json.loads(new.with_suffix(".json").read_text(encoding="utf-8"))["options"])
            print(f"  {stem:<52} max|delta| {delta:.3e}  labels {same_labels}  options {same_text}")
            if delta == 0.0 and same_labels and same_text:
                doomed.append(old)
            else:
                kept.append((old.name, f"differs (max {delta:.3e})"))

    if kept:
        print("\nNOT deleting anything: some sets could not be proven equal --")
        for name, why in kept:
            print(f"  kept {name}: {why}")
        return 2
    freed = 0
    for old in doomed:
        for q in (old, old.with_suffix(".json")):
            if q.exists():
                freed += os.path.getsize(q)
                q.unlink()
    print(f"\ndeleted {len(doomed) * 2} files, freed {freed / 1e6:.1f} MB; "
          f"{len(list(CACHE.glob('*.pt')))} cache sets remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
