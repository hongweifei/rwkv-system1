"""Does `pooled`'s gain track how long the options are?

The full suite says `pooled` is a wash overall (+1.4 pt over 17 paired tasks, 9/17 wins) but a
large win on the tasks with the biggest option blocks (`massive_en`, 60 options, +9.6 pt) and a
loss on small ones (`prompt_injections`, 2 options, -3.5 pt; `laya_urgency`, 4, -4.3 pt).  If the
gain tracks the *span the rule averages over*, then the rule can be chosen from a property of the
data instead of from the test set -- which is the difference between a mechanism and picking the
winner per task, and the 34 test sets have already been used as validation sets once.

The span is computed the way :func:`rwkv_system1.features._gather_markers` reads it: the tokens
between consecutive terminators in the option block.  Only the tokeniser is loaded, so this costs
nothing and can be run while an extraction owns the box.

Usage::

    python experiments/harnesses/option_span.py BASE_EVAL.json NEW_EVAL.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rwkv_system1.backbone import PromptTemplate, find_vocab  # noqa: E402
from rwkv_system1.cache_layout import CACHE, DEFAULT_BACKBONE, cache_name  # noqa: E402
from rwkv_system1.features import read_sidecar  # noqa: E402
from rwkv_system1.pipeline.config import Config  # noqa: E402


def spans(ids: list[int], terminator: int, n_options: int) -> list[int]:
    stops = [i for i, t in enumerate(ids) if t == terminator]
    if len(stops) != n_options + 1:
        raise ValueError(f"expected {n_options + 1} terminators, found {len(stops)}")
    return [stops[k + 1] - stops[k] for k in range(n_options)]


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())

    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

    tokenizer = TRIE_TOKENIZER(str(find_vocab()))
    terminator = tokenizer.encode(PromptTemplate().terminator)[0]

    base = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    new = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    cfg = Config()
    cfg.feature_rule = "pooled"

    rows = []
    for key in sorted(set(base) & set(new)):
        sidecar = read_sidecar(CACHE / cache_name(DEFAULT_BACKBONE, key, "train", cfg))
        if sidecar is None:
            print(f"  {key}: no pooled train sidecar, skipped")
            continue
        options, qtype = sidecar["options"], sidecar["qtype_name"]
        # The suffix is `header + terminator + option + terminator + ...`, and the tokeniser is
        # greedy left-to-right, so the token that *starts* at the header's terminator can absorb
        # the first characters of option 0 (`"\ne"` is one token, not two).  Prefixing the block
        # with the terminator reproduces that boundary; without it the count comes out short.
        term = PromptTemplate().terminator
        ids = tokenizer.encode(term + "".join(f"{o}{term}" for o in options))
        sp = spans(ids, terminator, len(options))
        rows.append((key, len(options), sum(sp) / len(sp), max(sp), new[key]["accuracy"] - base[key]["accuracy"]))

    rows.sort(key=lambda r: -r[2])
    print(f"{'task':<20}{'opts':>5}{'mean_span':>10}{'max_span':>9}{'d_acc_pt':>10}")
    for key, n, mean, mx, d in rows:
        print(f"{key:<20}{n:>5}{mean:>10.2f}{mx:>9}{d * 100:>+10.2f}")

    xs = [r[2] for r in rows]
    ys = [r[4] * 100 for r in rows]
    mx_, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx_) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    print(f"\nPearson r(mean_span, d_acc) = {cov / (sx * sy):+.3f} over {len(rows)} tasks")
    print(f"\n{'mean span':<16}{'n':>4}{'mean d_acc':>12}{'wins':>7}")
    for lo, hi in ((1, 2), (2, 4), (4, 9), (9, 10**9)):
        band = [r for r in rows if lo <= r[2] < hi]
        if not band:
            continue
        wins = sum(r[4] > 0 for r in band)
        label = f"{lo}-{hi} tok" if hi < 10**9 else f"{lo}+ tok"
        print(f"{label:<16}{len(band):>4}"
              f"{sum(r[4] for r in band) / len(band) * 100:>+12.2f}{wins:>4}/{len(band)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
