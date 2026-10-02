#!/usr/bin/env bash
# The frozen recipe's one honest test, queued behind the amazon repair.
#
# The gather rule is pinned per task on the tasks that chose it, so its number there is optimistic
# by construction.  This runs the shipped `Config()` unchanged on `clinc_oos` -- 151 labels, never in
# any sweep -- and the prediction is printed before the numbers: `pooled` must win, or the pin is
# overfit.  Queued rather than started because two extractors on this box roughly halve each other's
# throughput and have taken the box down once; bounded at four hours, then it proceeds on whatever
# state the tree is in (extraction is resumable, so a partial suite is recoverable).
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1

for _ in $(seq 1 240); do
  grep -q "^AMAZON_REPAIR_RC=" experiments/results/amazon_last.log && break
  sleep 60
done
echo "########## frozen recipe on clinc_oos (151 labels, never swept) ##########"
$PY experiments/harnesses/holdout_recipe.py clinc_oos
echo "HOLDOUT_RC=$?"
