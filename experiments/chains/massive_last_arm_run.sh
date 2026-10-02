#!/usr/bin/env bash
# Pair the nine MASSIVE tasks whose cap moved, by extracting the arm that is now missing: `last`.
#
# The rollout's paired table has 19 rows, and the 15 tasks it cannot pair are exactly the ones the
# 2026-09-24 cap raise re-sized (`massive_*` 400/100 -> 1000/300, `amazon_reviews_*` 1500/400 ->
# 3000/500).  Their `pooled` caches exist at the new sizes but their `last` caches are at the old
# ones, and the registry pins the new sizes -- so the missing arm is `last`, not `pooled`, and the
# measurement lands on the *shipping* configuration instead of on the old near-floor one.
#
# Why it matters: every task with a large label space gained under `pooled` (4/4, ~+10 pt), and the
# nine MASSIVE tasks are one task in nine languages sharing a 60-option block.  Two of them are
# measured (`massive_en` +9.60 in the rollout, `massive_zh` +12.44 in the screen) and this makes it
# nine, which is the difference between a within-family extrapolation and a measurement.  `amazon_*`
# is deliberately left out: it has 5 options, where the mechanism predicts no gain.
#
# One task per process: `ag_news@16000` and `banking77@4800` each died with exit 139 (segfault) when
# a big extraction shared a process with a smaller one, and MASSIVE at 0.78 s/example is the second
# most expensive family per example.  `extract` skips any split whose cache already matches, so
# re-running this is free.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0
ARMS=last
LANG_TASKS=massive_zh,massive_de,massive_fr,massive_es,massive_pt,massive_ru,massive_ar,massive_ja,massive_ko

BB=$($PY -c "from rwkv_system1.cache_layout import DEFAULT_BACKBONE; print(DEFAULT_BACKBONE)")
echo "backbone $BB   arm $ARMS   tasks $LANG_TASKS"

echo "########## 1) extract the $ARMS arm, one task per process ##########"
for task in $(echo "$LANG_TASKS" | tr ',' ' '); do
  echo "--- $task"
  $PY -m rwkv_system1.pipeline extract --tasks "$task" --feature-rule "$ARMS"
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EXTRACT_RC=$rc"
done

echo "########## 2) train + eval the $ARMS arm ##########"
$PY -m rwkv_system1.pipeline train --tasks "$LANG_TASKS" --feature-rule "$ARMS"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"
$PY -m rwkv_system1.pipeline eval --tasks "$LANG_TASKS" --feature-rule "$ARMS"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_RC=$rc"

# `eval` writes a table for the tasks it was given, so the nine new rows are merged into the 19-task
# table rather than replacing it.
$PY - "$BB" <<'PYEOF'
import json, sys
from pathlib import Path
bb = sys.argv[1]
dest = Path("experiments/results/eval_last.json")
merged = json.loads(dest.read_text(encoding="utf-8"))
add = json.loads(Path(f"runs/{bb}_eval.json").read_text(encoding="utf-8"))
merged.update(add)
dest.write_text(json.dumps(merged, indent=1), encoding="utf-8")
print(f"eval_last.json now holds {len(merged)} tasks (added: {', '.join(sorted(add))})")
PYEOF

echo "########## 3) paired table ##########"
$PY experiments/harnesses/paired_eval.py \
  experiments/results/eval_last.json experiments/results/eval_pooled.json \
  --base-label last --new-label pooled
$PY experiments/harnesses/option_span.py \
  experiments/results/eval_last.json experiments/results/eval_pooled.json

echo "MASSIVE_RULE_RC=$S"
