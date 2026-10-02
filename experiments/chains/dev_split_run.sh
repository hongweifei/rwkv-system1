#!/usr/bin/env bash
# Take dev from each dataset's own validation split, and re-run the tasks whose rows that moves.
#
# Why: dev was carved out of train in every family, which spent the row budget twice and left dev
# at 100-400 rows.  This project has measured that a fitted temperature is earned only from a
# ~500-row dev (`ag_news` is the only task that ever earned one), and `amazon_reviews_multi` was
# using its `validation.jsonl` as *test* while an unused `test.jsonl` sat beside it.  Now: dev from
# `validation`, train absorbing the freed rows, test unchanged wherever the dataset ships a test
# file.  `boolq` is left alone -- it has no labelled test split, so carving dev out of train is
# unavoidable there, and its dev was never the unmeasured one.
#
# Which tasks move, and how many rows each needs:
#   massive_*  x10  1100+500+300 (was 1000+300+300)  ~25 min each at 0.78 s/row   ~4.1 h
#   amazon_*   x6   3500+500+500 (was 3000+500+500)  ~26 min each at 0.34 s/row   ~2.6 h
#   xnli_*     x8   500+500+300  (was 400+100+300)   ~5 min each                  ~0.6 h
#   emotion         1300+500+600 (was 1000+300+600)                                ~0.2 h
# The other nine tasks keep their exact cache names, which is also the isolation check at the end.
#
# One arm per task, not two: the gather rule is pinned per task now, so the shipped arm is a single
# extraction, and the comparison this is for is against the *previous* shipped table (paired on
# identical test rows for the four families whose test split does not move -- `amazon_*` is the
# exception, since its test split is corrected here from `validation` to `test`).
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

MOVED=massive_zh,massive_en,massive_de,massive_fr,massive_es,massive_pt,massive_ru,massive_ar,massive_ja,massive_ko,xnli_zh,xnli_en,xnli_de,xnli_fr,xnli_es,xnli_ru,xnli_ar,xnli_vi,emotion,amazon_reviews_en,amazon_reviews_de,amazon_reviews_es,amazon_reviews_fr,amazon_reviews_ja,amazon_reviews_zh

echo "dev-split move: keeping the previous shipped table for the comparison"
cp experiments/results/eval_shipped.json experiments/results/eval_shipped_prev.json 2>/dev/null || echo "  (no previous table)"

REAL=$($PY -c "from rwkv_system1.task_registry import REAL_TASKS; print(','.join(REAL_TASKS))")
BB=$($PY -c "from rwkv_system1.cache_layout import DEFAULT_BACKBONE; print(DEFAULT_BACKBONE)")
echo "backbone $BB   moving $(echo "$MOVED" | tr ',' '\n' | wc -l) tasks"

echo "########## 1) extract the moved tasks, two per process ##########"
for grp in $($PY -c "
ks = '$MOVED'.split(',')
for i in range(0, len(ks), 2):
    print(','.join(ks[i:i + 2]))
"); do
  echo "--- $grp"
  $PY -m rwkv_system1.pipeline extract --tasks "$grp"
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EXTRACT_RC=$rc"
done

echo "########## 2) train their heads ##########"
$PY -m rwkv_system1.pipeline train --tasks "$MOVED"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"

echo "########## 3) eval all 34 under the shipped recipe ##########"
$PY -m rwkv_system1.pipeline eval --tasks "$REAL"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_RC=$rc"; EVAL_RC_VALUE=$rc
# A failed eval writes no table, and `runs/<bb>_eval.json` then still holds the previous run's.
# Copying that and comparing it against itself produced a plausible "+0.00pt / 0 drift" verdict once
# already, so nothing after a failed step runs: no table is better than a stale one.
if [ "$EVAL_RC_VALUE" != "0" ]; then
  echo "NO SHIPPED TABLE: eval exited $EVAL_RC_VALUE (missing caches above); stopping rather than"
  echo "copying and comparing the previous run's table, which would look like a result."
  echo "DEV_SPLIT_RC=$EVAL_RC_VALUE"
  exit 1
fi
cp "runs/${BB}_eval.json" "experiments/results/eval_shipped.json"

echo "########## 4) the shipped row must equal the task's pinned arm, and nothing else may move ##########"
$PY - <<PYEOF
import json
from pathlib import Path

from rwkv_system1.cache_layout import resolve_rule

new = json.loads(Path("experiments/results/eval_shipped.json").read_text(encoding="utf-8"))
prev_path = Path("experiments/results/eval_shipped_prev.json")
prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {}
moved = set("$MOVED".split(","))

print(f"\n{len(new)} tasks in the shipped table")
untouched = [k for k in new if k not in moved and k in prev]
drifted = [k for k in untouched if abs(new[k]["accuracy"] - prev[k]["accuracy"]) > 1e-9]
print(f"isolation: {len(untouched)} tasks were not supposed to move; {len(drifted)} did"
      + (f" -> {drifted}" if drifted else ""))

print(f"\n{'task':<16}{'acc before':>11}{'acc now':>9}{'delta':>8}   {'ECE before':>10}{'ECE now':>8}")
tot_b = tot_n = 0.0
for k in sorted(moved):
    if k not in prev:
        print(f"{k:<16}{'(new)':>11}{new[k]['accuracy'] * 100:>8.2f}%")
        continue
    b, n = prev[k]["accuracy"], new[k]["accuracy"]
    tot_b += b
    tot_n += n
    print(f"{k:<16}{b * 100:>10.2f}%{n * 100:>8.2f}%{(n - b) * 100:>+8.2f}   "
          f"{prev[k]['ece']:>10.4f}{new[k]['ece']:>8.4f}")
n_moved = sum(1 for k in moved if k in prev)
print(f"\nmean over the {n_moved} moved tasks: {tot_b / n_moved * 100:.2f}% -> {tot_n / n_moved * 100:.2f}%"
      f"  ({(tot_n - tot_b) / n_moved * 100:+.2f}pt)")
pooled = [k for k in new if resolve_rule(None, k) == "pooled"]
acc = [r["accuracy"] for r in new.values()]
print(f"shipped table: {len(new)} tasks, {len(pooled)} on pooled, mean {sum(acc) / len(acc):.4f}")
PYEOF
rc=$?; [ $rc -ne 0 ] && S=$rc

echo "DEV_SPLIT_RC=$S"
