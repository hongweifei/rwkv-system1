#!/usr/bin/env bash
# Roll the `pooled` marker gather out to the whole suite, and keep the `last` arm alongside it.
#
# `pooled` averages the hidden states over each option's token span and beat the shipped
# terminator-only gather by +8.93 +/- 2.40 pt over three tasks, 3/3 wins, at the same top-label
# ECE (experiments/results/thick_features_report.txt).  It reads the same single forward pass, so
# the alternatives cost cache width, not compute -- but it *is* a different cache, so the whole
# suite has to be extracted again before anything can be trained on it.  The `last` caches stay on
# disk, which is what makes the paired table possible without a second extraction.
#
# Two tasks per extract process: `cmd_extract` builds every requested corpus before it extracts
# anything, so the chunk size is a memory bound.  See the longer note in `suite_run.sh` for what a
# five-task chunk did to this box on 2026-09-23.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0
NEW=pooled
BASE=last

BB=$($PY -c "from rwkv_system1.cache_layout import DEFAULT_BACKBONE; print(DEFAULT_BACKBONE)")
echo "backbone $BB   tasks $($PY -c "from rwkv_system1.task_registry import REAL_TASKS; print(len(REAL_TASKS))")   new rule $NEW   base rule $BASE"

echo "########## 1) extract $NEW, two tasks per process ##########"
for grp in $($PY -c "
from rwkv_system1.task_registry import REAL_TASKS
ks = list(REAL_TASKS)
for i in range(0, len(ks), 2):
    print(','.join(ks[i:i + 2]))
"); do
  echo "--- $grp"
  $PY -m rwkv_system1.pipeline extract --tasks "$grp" --feature-rule "$NEW"
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EXTRACT_RC=$rc"
done

# `train --tasks $ALL` aborts on the first missing cache, so the ready set is computed from disk
# and the gap is printed rather than hidden.  Re-running the chain fills it: extraction is
# idempotent and skips any split whose cache already matches.
cat > /tmp/_complete.py <<'PYEOF'
import sys
from pathlib import Path
from rwkv_system1.cache_layout import DEFAULT_BACKBONE, CACHE, cache_name
from rwkv_system1.pipeline.config import Config
from rwkv_system1.task_registry import REAL_TASKS

cfg = Config()
cfg.feature_rule = sys.argv[1]
out = []
for key in REAL_TASKS:
    dests = [CACHE / cache_name(DEFAULT_BACKBONE, key, s, cfg) for s in ("train", "dev", "test")]
    if all(Path(str(d) + ".pt").exists() for d in dests):
        out.append(key)
print(",".join(out))
PYEOF

READY=$($PY /tmp/_complete.py "$NEW")
N_READY=$(echo "$READY" | tr ',' '\n' | grep -c .)
echo "caches complete for $NEW: $N_READY / 34"
MISSING=$($PY -c "
import sys
from rwkv_system1.task_registry import REAL_TASKS
have = {k for k in '$READY'.split(',') if k}
print(','.join(k for k in REAL_TASKS if k not in have))
")
[ -n "$MISSING" ] && { echo "NO CACHE, NOT TRAINED: $MISSING"; S=1; }

echo "########## 2) train + eval the $NEW arm ##########"
$PY -m rwkv_system1.pipeline train --tasks "$READY" --feature-rule "$NEW"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"
$PY -m rwkv_system1.pipeline eval --tasks "$READY" --feature-rule "$NEW"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_RC=$rc"
cp "runs/${BB}_eval.json" "experiments/results/eval_${NEW}.json"

# The base arm can only be paired on the tasks whose *sizes* did not move with this rollout: the
# 2026-09-24 cap raise changed what `massive_*` and `amazon_reviews_*` ask the loader for, so a
# `last` cache at the new sizes does not exist and training one would need a second extraction.
# Those fifteen tasks are compared as old-recipe vs new-recipe in the table, and their rule-vs-rows
# split is measured separately by re-running the knee harness on the pooled cache.
BASEREADY=$($PY /tmp/_complete.py "$BASE")
N_BASE=$(echo "$BASEREADY" | tr ',' '\n' | grep -c .)
echo "########## 3) train + eval the $BASE arm on the $N_BASE tasks that still pair ##########"
$PY -m rwkv_system1.pipeline train --tasks "$BASEREADY" --feature-rule "$BASE"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "BASETRAIN_RC=$rc"
$PY -m rwkv_system1.pipeline eval --tasks "$BASEREADY" --feature-rule "$BASE"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "BASEEVAL_RC=$rc"
cp "runs/${BB}_eval.json" "experiments/results/eval_${BASE}.json"

echo "########## 4) paired table ##########"
$PY experiments/harnesses/paired_eval.py \
  "experiments/results/eval_${BASE}.json" "experiments/results/eval_${NEW}.json" \
  --base-label "$BASE" --new-label "$NEW"
rc=$?; [ $rc -ne 0 ] && S=$rc

echo "POOLED_ROLLOUT_RC=$S"
