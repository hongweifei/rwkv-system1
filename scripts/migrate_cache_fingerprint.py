"""Move the feature caches onto the current fingerprint, re-extracting only what must change.

Run:  python scripts/migrate_cache_fingerprint.py [--apply]

Why this is not just `mv`: a fingerprint covers the whole pipeline, so *adding or removing any
input* renames every cache even when the vectors cannot differ.  Here the change was the removal
of the 192-token state cap, and whether it changed anything is a per-task, measurable question --
`encode_state` did `ids[:192]`, so for a split whose longest state was 192 tokens or fewer the
slice was the identity and the cached vectors are the same vectors.

So the rule is: measure the longest state in the stored sidecar (the raw states are already
there, so this needs only the tokeniser), and

* **longest <= 192** -> the cap never fired -> rename the pair to the new fingerprint;
* **longest > 192**   -> the cap fired -> delete the pair, and let `extract` rebuild it.

Nothing is renamed before that measurement, and the script prints every decision.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

from rwkv_system1.backbone import PromptTemplate  # noqa: E402
from rwkv_system1.cache_layout import (  # noqa: E402
    BACKBONES, CACHE, DEFAULT_BACKBONE, data_version, feature_fingerprint, feature_inputs,
)
from rwkv_system1.pipeline.config import Config  # noqa: E402
from rwkv_system1.task_registry import REAL_TASKS  # noqa: E402

OLD_CAP = 192


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="rename/delete; without it, only report")
    ap.add_argument("--identity-only", action="store_true",
                    help="the change altered only what is *recorded*, not any vector: rename every "
                         "set and write the current inputs record into its sidecar")
    args = ap.parse_args()

    cfg = Config()
    new = feature_fingerprint(cfg, DEFAULT_BACKBONE)
    tok = TRIE_TOKENIZER(str(Path(__file__).resolve().parent.parent /
                             "weights" / "rwkv_vocab_v20230424.txt"))
    template = PromptTemplate()

    rows = []
    for pt in sorted(glob.glob(str(CACHE / "*.pt"))):
        fp = os.path.basename(pt).rsplit("_", 1)[1][:-3]
        if fp == new:
            continue
        side = json.loads(Path(pt[:-3] + ".json").read_text(encoding="utf-8"))
        longest = max((len(tok.encode(template.prefix(s))) for s in side["states"]), default=0)
        name = os.path.basename(pt)[:-len(f"_{fp}.pt")]
        rows.append((name, fp, len(side["states"]), longest, longest > OLD_CAP))

    if not rows:
        print(f"every cache is already on {new}")
        return 0

    if args.identity_only:
        backbone = next((b for b in BACKBONES if rows[0][0].startswith(b + "_")), DEFAULT_BACKBONE)
        print(f"target fingerprint {new} (identity-only change: no vector moved)\n")
        print(f"{'cache':<50}{'rows':>6}  action")
        moved = 0
        for name, fp, n, _longest, _capped in rows:
            print(f"{name:<50}{n:>6}  rename + record inputs")
            if not args.apply:
                continue
            key = next((k for k in REAL_TASKS if f"_{k}_" in name), None)
            old_pt, old_js = CACHE / f"{name}_{fp}.pt", CACHE / f"{name}_{fp}.json"
            new_pt, new_js = CACHE / f"{name}_{new}.pt", CACHE / f"{name}_{new}.json"
            old_pt.rename(new_pt)
            side = json.loads(old_js.read_text(encoding="utf-8"))
            side["inputs"] = feature_inputs(cfg, backbone,
                                           data_version(key) if key else None)
            new_js.write_text(json.dumps(side, ensure_ascii=False), encoding="utf-8")
            old_js.unlink()
            moved += 1
        print(f"\n{'renamed' if args.apply else 'would rename'} {moved or len(rows)} sets")
        return 0

    print(f"target fingerprint {new}\n")
    print(f"{'cache':<50}{'rows':>6}{'longest':>9}  decision")
    moved = rebuilt = 0
    for name, fp, n, longest, capped in rows:
        decision = "RE-EXTRACT (the cap cut it)" if capped else "rename (cap was a no-op)"
        print(f"{name:<50}{n:>6}{longest:>9}  {decision}")
        if not args.apply:
            continue
        old_pt = CACHE / f"{name}_{fp}.pt"
        old_js = CACHE / f"{name}_{fp}.json"
        if capped:
            for q in (old_pt, old_js):
                if q.exists():
                    q.unlink()
            rebuilt += 1
        else:
            old_pt.rename(CACHE / f"{name}_{new}.pt")
            old_js.rename(CACHE / f"{name}_{new}.json")
            moved += 1
    if args.apply:
        print(f"\nrenamed {moved}, queued {rebuilt} for re-extraction")
    else:
        print(f"\n{sum(1 for r in rows if not r[4])} would be renamed, "
              f"{sum(1 for r in rows if r[4])} re-extracted -- rerun with --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
