#!/usr/bin/env bash
# Re-extract the two splits the pooled pass lost, one task per process.
#
# `ag_news@16000` and `banking77@4800` are the two most expensive single splits in the suite
# (0.22 s and 1.85 s per example, 16000 and 4800 rows), and both died with `Segmentation fault`
# (exit 139) at the moment their extraction started, while sharing a process with a smaller task.
# That is this box's documented commit-charge wall, not a code fault: the pooled gather cannot
# read out of bounds (`encode_query` guarantees exactly `n_options + 1` terminators, and `stops`
# is derived from the same ids), and `tests/invariants.py::pooled_gathers_the_option_span` pins
# the arithmetic.  So the fix is to hold less at once -- one task per process -- not to change
# the rule.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

for task in ag_news banking77; do
  echo "########## $task (alone) ##########"
  $PY -m rwkv_system1.pipeline extract --tasks "$task" --feature-rule pooled
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EXTRACT_RC=$rc"
done

# `extract` is the only writer of the standing config, so this run has left it saying
# `feature_rule: pooled`.  Whether that is right depends on the decision the paired table is
# waiting on, so the caller settles it rather than this script guessing.
echo "MISSING_RECOVERY_RC=$S"
