#!/usr/bin/env bash
# Land the MASSIVE joint head the registry now pins, then prove the pipeline reproduces the harness.
#
# Queued behind the stacking sweep rather than run beside it: two trainers on this box halve each
# other's throughput.  The eval list is generated from REAL_TASKS, not written out by hand -- a
# literal task list is the one place a task silently goes missing.
#
# The check that makes this worth running is step 3: `pipeline eval` must reproduce the joint arm's
# per-language accuracies from experiments/results/joint_family.json.  If it does, the shipped path
# (registry group -> joined caches -> one head -> per-task eval) is the same computation the
# measurement was made on.  If it does not, the pin is wired to something other than what was
# measured, which is exactly the failure mode the identity layer exists to prevent.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0
LOG=experiments/results/joint_head_landing.log

for _ in $(seq 1 120); do
  grep -q "^JOINT_ENSEMBLE_RC=" experiments/results/joint_ensemble.log && break
  sleep 30
done
echo "stacking sweep finished; landing the joint head"

MASSIVE=$($PY -c "
from rwkv_system1.task_registry import REAL_TASKS
from rwkv_system1.cache_layout import joint_group
print(','.join(k for k in REAL_TASKS if joint_group(k) == 'massive'))
")
echo "group members: $MASSIVE"

echo "--- 1. train the group head (requesting one member must build the whole group)"
$PY -m rwkv_system1.pipeline train --tasks massive_zh
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"

echo "--- 2. every member must now be nameable, and all from one file"
$PY -c "
from rwkv_system1.cache_layout import RUNS, DEFAULT_BACKBONE, load_features, run_name, group_members
from rwkv_system1.pipeline.config import Config
cfg = Config()
names = {}
bad = []
for k in group_members('massive_zh'):
    fs = load_features(DEFAULT_BACKBONE, k, 'test', cfg)
    n = run_name(DEFAULT_BACKBONE, k, cfg, fs)
    names[n] = names.get(n, 0) + 1
    if not (RUNS / f'{n}.pt').exists():
        bad.append(k)
print('distinct head files over the group:', len(names), list(names.values()))
print('members whose head is missing:', bad)
raise SystemExit(1 if bad or len(names) != 1 else 0)
"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "NAMECHECK_RC=$rc"

echo "--- 3. eval the group and compare against the harness's joint arm"
$PY -m rwkv_system1.pipeline eval --tasks "$MASSIVE" > experiments/results/eval_joint_massive.log 2>&1
rc=$?; echo "EVAL_RC=$rc"
[ $rc -ne 0 ] && S=$rc
if [ "$rc" = "0" ]; then
  $PY - <<'PY'
import json, statistics as st
from pathlib import Path

# The shipped eval is one head at cfg.seed = 0, so the thing to compare it against is the harness's
# seed-0 joint rows, not its 3-seed mean.  Both paths now call train(seed=0), so agreement here
# means the pipeline reached the same weights the measurement used; CPU matmul order still leaves a
# fraction of a point, which is why the max diff is printed rather than an equality asserted.
ev = json.loads(Path("runs/rwkv7-g1d-0.1b_eval.json").read_text(encoding="utf-8"))
fams = json.loads(Path("experiments/results/joint_family.json").read_text(encoding="utf-8"))
massive = next(f for f in fams if f["family"] == "massive")
harn = {r["task"]: r["t1"]["accuracy"] for r in massive["rows"]
        if r["arm"] == "joint" and r["seed"] == 0}
print(f"{'task':<16}{'pipeline':>10}{'harness s0':>12}{'diff':>9}")
diffs = []
for k in sorted(harn):
    got = ev[k]["accuracy"]
    diffs.append((abs(got - harn[k]), k))
    print(f"{k:<16}{got * 100:>9.2f}%{harn[k] * 100:>11.2f}%{(got - harn[k]) * 100:>+9.2f}")
worst, name = max(diffs)
print(f"\nmax |pipeline - harness| = {worst * 100:.2f} pt ({name}) over {len(diffs)} languages")
print("family mean  pipeline",
      round(st.mean(ev[k]["accuracy"] for k in harn) * 100, 2), "%  harness",
      round(st.mean(harn.values()) * 100, 2), "%")
PY
else
  echo "NO COMPARISON -- the eval step failed, so there is no shipped number to line up"
fi
echo "JOINT_HEAD_LANDING_RC=$S"
