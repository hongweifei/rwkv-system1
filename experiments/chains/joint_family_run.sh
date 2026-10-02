#!/usr/bin/env bash
# The per-family joint head, queued behind the seed-ensemble sweep.
#
# Queued rather than started in parallel because two trainers on this box halve each other's
# throughput and have taken it down before. Waits for the ensemble sweep's final status line, then
# runs every family whose tasks share one option list: massive (10 languages), xnli (8), amazon (6,
# a score question -- the earlier joint comparison lost on its one score task, so it is the family
# most likely to say no).  Nothing here extracts: all features are already on disk.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

for _ in $(seq 1 240); do
  grep -q "^SEED_ENSEMBLE_RC=" experiments/results/seed_ensemble.log && break
  sleep 60
done
echo "ensemble sweep finished; starting the joint-family arms"

for family in massive xnli amazon; do
  echo "--- $family"
  $PY experiments/harnesses/joint_family.py "$family"
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "FAMILY_RC=$rc"
done
echo "JOINT_FAMILY_RC=$S"
