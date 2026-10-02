#!/usr/bin/env bash
# Finish the amazon repair: retry the split that died, then eval the *real* 34 and run the check.
#
# Two separate faults in `amazon_last_run.sh`, both visible in its log:
#   * `amazon_reviews_zh` segfaulted on its last split (train and dev landed, test did not).  It was
#     already one task per process, so chunk size is not the whole story -- amazon's reviews are
#     long documents and this box's commit wall is intermittent.  Extraction is idempotent, so a
#     retry only pays for the missing split.
#   * `pipeline eval` with **no `--tasks`** iterates `all_task_keys()`, which includes the generated
#     corpora, and those have no cache on disk -- it died on `ticket_routing` with
#     FileNotFoundError and never wrote a table.  The check then ran against a *stale*
#     `runs/*_eval.json` (the nine MASSIVE languages from the previous chain) and reported nine
#     disagreements, which is the check doing its job: it compared shipped against the pinned arm
#     and did not pass silently.  The list has to be the registered real tasks, so it is computed
#     from `REAL_TASKS` here rather than left to the default.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

# Printed before anything that costs memory.  The first version of this script computed its two
# constants with `$PY -c` *before* its first echo, and when the box ran out of commit charge those
# imports wedged -- so the log stayed 0 bytes for fifteen minutes and the failure looked like
# nothing happening at all.  Anything that can hang belongs after a line that says it started.
echo "amazon finish: retry amazon_reviews_zh, train the six, eval the 34, check the shipped table"

BB=$($PY -c "from rwkv_system1.cache_layout import DEFAULT_BACKBONE; print(DEFAULT_BACKBONE)")
AMAZON=amazon_reviews_en,amazon_reviews_de,amazon_reviews_es,amazon_reviews_fr,amazon_reviews_ja,amazon_reviews_zh
REAL=$($PY -c "from rwkv_system1.task_registry import REAL_TASKS; print(','.join(REAL_TASKS))")
echo "backbone $BB   ${REAL//,/ } "

echo "########## 1) retry the split that died (up to 3 attempts) ##########"
for attempt in 1 2 3; do
  $PY -m rwkv_system1.pipeline extract --tasks amazon_reviews_zh --feature-rule last
  rc=$?; echo "ATTEMPT_RC=$rc"
  [ $rc -eq 0 ] && break
  S=$rc
done

echo "########## 2) train the six amazon heads (train/dev were all written, so this is quick) ##########"
$PY -m rwkv_system1.pipeline train --tasks "$AMAZON"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"

echo "########## 3) eval every registered real task under the shipped recipe ##########"
$PY -m rwkv_system1.pipeline eval --tasks "$REAL"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_RC=$rc"; EVAL_RC_VALUE=$rc
# A failed eval writes no table, and `runs/<bb>_eval.json` then still holds the previous run's.
# Copying that and comparing it against itself produced a plausible "+0.00pt / 0 drift" verdict once
# already, so nothing after a failed step runs: no table is better than a stale one.
if [ "$EVAL_RC_VALUE" != "0" ]; then
  echo "NO SHIPPED TABLE: eval exited $EVAL_RC_VALUE (missing caches above); stopping rather than"
  echo "copying and comparing the previous run's table, which would look like a result."
  echo "AMAZON_FINISH_RC=$EVAL_RC_VALUE"
  exit 1
fi
cp "runs/${BB}_eval.json" "experiments/results/eval_shipped.json"

echo "########## 4) the shipped table must equal the arm each task is pinned to ##########"
$PY - <<'PYEOF'
import json
from pathlib import Path

from rwkv_system1.cache_layout import resolve_rule

shipped = json.loads(Path("experiments/results/eval_shipped.json").read_text(encoding="utf-8"))
arms = {n: json.loads(Path(f"experiments/results/eval_{n}.json").read_text(encoding="utf-8"))
        for n in ("last", "pooled")}
bad = [k for k, row in shipped.items()
       if k in arms[resolve_rule(None, k)]
       and abs(row["accuracy"] - arms[resolve_rule(None, k)][k]["accuracy"]) > 1e-9]
pooled_tasks = [k for k in shipped if resolve_rule(None, k) == "pooled"]
print(f"\n{len(shipped)} tasks in the shipped table, {len(pooled_tasks)} of them on pooled")
print(f"disagreements with the pinned arm: {len(bad)}")
for k in bad:
    want = resolve_rule(None, k)
    print(f"  {k}: shipped {shipped[k]['accuracy']:.4f} vs {want} {arms[want][k]['accuracy']:.4f}")
acc = [r["accuracy"] for r in shipped.values()]
print(f"mean accuracy over the {len(acc)} tasks: {sum(acc) / len(acc):.4f}")
PYEOF
rc=$?; [ $rc -ne 0 ] && S=$rc

echo "AMAZON_FINISH_RC=$S"
