#!/usr/bin/env bash
# Close the gap the amazon cap raise left, then produce the shipped table under the pins.
#
# Raising `amazon_reviews_*` to 3000/500/500 orphaned their `last` caches (still at 1500/400/500)
# and the rollout only extracted them under `pooled` -- so those six tasks had no features at all
# in the configuration they actually ship in.  The pooled caches were never the shipped ones: the
# gather rule pays off by label-space size, amazon has 5 options, and the small-option tasks that
# were measured are a wash with real losses (`prompt_injections` -3.5 pt, ECE 0.028 -> 0.095), so
# `last` is what these six keep.
#
# Then one `eval` over all 34 with no `--feature-rule` at all: that is the shipped configuration
# now that the rule is pinned per task, and its table is the deliverable.  The last step checks it
# against the arms it should agree with -- a pinned task must equal the pooled arm's row and an
# unpinned one the last arm's, task by task, or the resolution is not reaching everything it should.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

BB=$($PY -c "from rwkv_system1.cache_layout import DEFAULT_BACKBONE; print(DEFAULT_BACKBONE)")
AMAZON=amazon_reviews_en,amazon_reviews_de,amazon_reviews_es,amazon_reviews_fr,amazon_reviews_ja,amazon_reviews_zh

echo "########## 1) extract the missing $AMAZON last features, one task per process ##########"
for task in $(echo "$AMAZON" | tr ',' ' '); do
  echo "--- $task"
  $PY -m rwkv_system1.pipeline extract --tasks "$task" --feature-rule last
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EXTRACT_RC=$rc"
done

echo "########## 2) train their heads ##########"
$PY -m rwkv_system1.pipeline train --tasks "$AMAZON"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"

echo "########## 3) eval all 34 under the shipped recipe (no --feature-rule) ##########"
$PY -m rwkv_system1.pipeline eval
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_RC=$rc"
cp "runs/${BB}_eval.json" "experiments/results/eval_shipped.json"

echo "########## 4) the shipped table must equal the arm each task is pinned to ##########"
$PY - <<'PYEOF'
import json
from pathlib import Path

shipped = json.loads(Path("experiments/results/eval_shipped.json").read_text(encoding="utf-8"))
arms = {name: json.loads(Path(f"experiments/results/eval_{name}.json").read_text(encoding="utf-8"))
        for name in ("last", "pooled")}
from rwkv_system1.cache_layout import resolve_rule  # noqa: E402

bad, checked = [], 0
for key, row in shipped.items():
    want = resolve_rule(None, key)
    if key not in arms[want]:
        print(f"  {key}: no {want} arm to check against (cap moved since it was run)")
        continue
    checked += 1
    if abs(row["accuracy"] - arms[want][key]["accuracy"]) > 1e-9:
        bad.append(f"{key}: shipped {row['accuracy']:.4f} vs {want} {arms[want][key]['accuracy']:.4f}")
print(f"\n{checked} tasks checked against their pinned arm; disagreements: {len(bad)}")
for line in bad:
    print("  " + line)
pooled = [k for k in shipped if resolve_rule(None, k) == "pooled"]
print(f"\nshipped table: {len(shipped)} tasks, {len(pooled)} of them on pooled")
print(f"mean accuracy over the suite: {sum(r['accuracy'] for r in shipped.values()) / len(shipped):.4f}")
PYEOF
rc=$?; [ $rc -ne 0 ] && S=$rc

echo "AMAZON_REPAIR_RC=$S"
