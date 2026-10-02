set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1

# Wait for the baseline chain's last line before starting.  Two reasons, both measured today:
# this box cannot hold two of these jobs at once (16 GB, small page file -- the last time it tried,
# the extractor took a SIGSEGV and every later process died with STATUS_INSUFFICIENT_RESOURCES), and
# the comparison is only meaningful against the same task list, so the sweep has to see whatever
# the baseline ended up finishing.  Bounded at eight hours, then it proceeds on what is cached.
for _ in $(seq 1 480); do
  grep -q "^SUITE_STATUS=" experiments/results/suite_run3.log && break
  sleep 60
done

S=0
# Four tasks per process.  The task list is the *cached* one, not the registry, because
# `arm_sweep.py` raises on a task whose features are missing rather than skipping it quietly.
echo "########## arm sweep: shipped vs head_d_model=64 vs rlcd_epochs=0 ##########"
for grp in $($PY -c "
import sys
sys.path.insert(0, 'experiments/harnesses')
from arm_sweep import cached_tasks
ks = cached_tasks()
print(f'{len(ks)} cached tasks', file=sys.stderr)
for i in range(0, len(ks), 4):
    print(','.join(ks[i:i + 4]))
"); do
  echo "--- $grp"
  $PY experiments/harnesses/arm_sweep.py "$grp"
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "ARM_RC=$rc"
done
echo "ARM_SWEEP_STATUS=$S"
