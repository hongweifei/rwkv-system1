#!/usr/bin/env bash
# Close out the pooled rollout: train + eval the two splits its crashes had lost, fold them into
# the pooled table, print the 19-task paired table, and put the standing config back.
#
# The `last` arm already has these two tasks (eval_last.json holds 19), so this is the step that
# makes the comparison complete -- and they are the two most decision-relevant rows in it:
# `banking77` has the most options in the suite (77) and is the task the cap raise helped most
# (46.0 -> 61.4%), `ag_news` is the one with a published Laya number.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

BB=$($PY -c "from rwkv_system1.cache_layout import DEFAULT_BACKBONE; print(DEFAULT_BACKBONE)")
TASKS=ag_news,banking77

echo "########## train + eval the recovered $TASKS on the pooled arm ##########"
$PY -m rwkv_system1.pipeline train --tasks "$TASKS" --feature-rule pooled
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"
$PY -m rwkv_system1.pipeline eval --tasks "$TASKS" --feature-rule pooled
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_RC=$rc"

# `eval` writes a table for the tasks it was given, so the two new rows are merged into the
# 32-task table rather than replacing it.
$PY - "$BB" <<'PYEOF'
import json, sys
from pathlib import Path
bb = sys.argv[1]
pooled = Path("experiments/results/eval_pooled.json")
merged = json.loads(pooled.read_text(encoding="utf-8"))
add = json.loads(Path(f"runs/{bb}_eval.json").read_text(encoding="utf-8"))
for key, row in add.items():
    merged[key] = row
pooled.write_text(json.dumps(merged, indent=1), encoding="utf-8")
print(f"eval_pooled.json now holds {len(merged)} tasks (added: {', '.join(sorted(add))})")
PYEOF

echo "########## paired table, 19 tasks ##########"
$PY experiments/harnesses/paired_eval.py \
  experiments/results/eval_last.json experiments/results/eval_pooled.json \
  --base-label last --new-label pooled
$PY experiments/harnesses/option_span.py \
  experiments/results/eval_last.json experiments/results/eval_pooled.json

# `extract` is the only writer of the standing config and the recovery run left it saying
# `feature_rule: pooled`.  The rule is closed, so the recipe has to go back to `last`: run a
# one-task extract, which rewrites the config and skips every split it finds cached.
echo "########## restore the standing recipe to feature_rule=last ##########"
$PY -m rwkv_system1.pipeline extract --tasks laya_category --feature-rule last
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "RESTORE_RC=$rc"
$PY -c "import json; d = json.load(open('runs/${BB}_config.json', encoding='utf-8')); print('standing config feature_rule =', d['feature_rule'])"

echo "POOLED_FINISH_RC=$S"
