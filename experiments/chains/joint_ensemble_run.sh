#!/usr/bin/env bash
# The stacking question, on the machine's only free CPU budget.
#
# One process: it never touches the backbone, so the peak RSS is the ten pooled caches plus three
# seeds of probabilities (~1.6 GB measured on joint_family.py), not the multi-GB extraction path that
# needs one-task-per-process.  The harness writes its own identity sidecar, so a later edit to the
# feature pipeline cannot silently invalidate this table without the two disagreeing.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1

$PY experiments/harnesses/joint_ensemble.py
rc=$?
echo "JOINT_ENSEMBLE_RC=$rc"
exit $rc
