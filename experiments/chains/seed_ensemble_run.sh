#!/usr/bin/env bash
# Three seeds per task, merged per chunk.
#
# Four tasks per process: the head trainings are cheap (the features are cached), but a process
# that loops all 34 tasks has died on this box before, and re-running is free because each
# invocation merges into experiments/results/seed_ensemble.json and replaces only what it redid.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

for grp in $($PY -c "
from rwkv_system1.task_registry import REAL_TASKS
ks = list(REAL_TASKS)
for i in range(0, len(ks), 4):
    print(','.join(ks[i:i + 4]))
"); do
  echo "--- $grp"
  $PY experiments/harnesses/seed_ensemble.py "$grp"
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "CHUNK_RC=$rc"
done
echo "SEED_ENSEMBLE_RC=$S"
