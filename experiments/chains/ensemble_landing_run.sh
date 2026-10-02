#!/usr/bin/env bash
# Ship the 3-seed ensemble: train the seed replicas, then prove the pipeline reproduces the numbers
# the two harnesses measured, and price the drift that retraining seed 0 causes.
#
# Why retrain seed 0 instead of only adding seeds 1 and 2: the members then all come out of one code
# version in one run, which is the honest meaning of "the shipped model is the average of three
# heads".  That overwrites the artifact whose eval was verified against the harness at
# max|dNLL| = 0.00e+00, so step 0 snapshots the seed-0 heads and step 5 measures the drift rather
# than pretending there is none.
#
# Every task list is generated from REAL_TASKS: a hand-written literal is where a task silently goes
# missing, and a table missing one task still prints a mean.
set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0
RES=experiments/results
FAIL=0

REAL=$($PY -c "
from rwkv_system1.task_registry import REAL_TASKS
print(','.join(REAL_TASKS))
")
echo "suite: $(echo $REAL | tr ',' '\n' | wc -l) tasks"

echo "--- 0. snapshot the seed-0 heads the harness cross-check was made on"
$PY - <<'PY'
import json, shutil
from pathlib import Path
from rwkv_system1.cache_layout import RUNS
dest = RUNS / "_pre_ensemble_seed0"
dest.mkdir(exist_ok=True)
n = 0
for p in sorted(RUNS.glob("*.pt")):
    meta = p.with_suffix(".json")
    if not meta.exists():
        continue
    if json.loads(meta.read_text(encoding="utf-8")).get("head_inputs", {}).get("seed") == 0:
        shutil.copy2(p, dest / p.name)
        shutil.copy2(meta, dest / meta.name)
        n += 1
print(f"  snapshotted {n} seed-0 heads to {dest.name}/")
PY
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "SNAPSHOT_RC=$rc"

echo "--- 1. train three seeds of every head (a group's head is trained once per seed, shared)"
$PY -m rwkv_system1.pipeline train --tasks "$REAL" --ensemble-seeds 3
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "TRAIN_RC=$rc"

echo "--- 2. eval the ensemble, then the single head, and keep both tables"
$PY -m rwkv_system1.pipeline eval --tasks "$REAL" --ensemble-seeds 3 > $RES/eval_ens3.log 2>&1
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_ENS_RC=$rc"
if [ "$rc" = "0" ]; then
  cp runs/rwkv7-g1d-0.1b_eval.json $RES/eval_shipped_ens3.json
  cp runs/rwkv7-g1d-0.1b_eval.meta.json $RES/eval_shipped_ens3.meta.json
else
  echo "NO ENSEMBLE TABLE -- eval failed, so $RES/eval_shipped_ens3.json was NOT written"
  FAIL=1
fi

$PY -m rwkv_system1.pipeline eval --tasks "$REAL" > $RES/eval_single.log 2>&1
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EVAL_SINGLE_RC=$rc"
if [ "$rc" = "0" ]; then
  cp runs/rwkv7-g1d-0.1b_eval.json $RES/eval_shipped_seed0.json
  cp runs/rwkv7-g1d-0.1b_eval.meta.json $RES/eval_shipped_seed0.meta.json
else
  echo "NO SINGLE TABLE -- eval failed"
  FAIL=1
fi

echo "--- 3. acceptance: the pipeline's ensemble against the harnesses that measured it"
if [ "$FAIL" = "0" ]; then
$PY - <<'PY'
import json, statistics as st
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

# Read the saved copies, not runs/*_eval.json: step 4 re-evaluates the single head and overwrites
# those files, so a check placed after it would read a single-head table and call it the ensemble.
# That ordering mistake is what made the first run of this chain report ACCEPTANCE_RC=1 on a model
# whose numbers were in fact exact.
meta = load("experiments/results/eval_shipped_ens3.meta.json")
assert meta["ensemble_seeds"] == 3, f"step 2 left no ensemble table: {meta}"
ens = load("experiments/results/eval_shipped_ens3.json")
one = load("experiments/results/eval_shipped_seed0.json")

print("A. every ensemble member must exist and be a distinct artifact")
import dataclasses
from rwkv_system1.cache_layout import RUNS, DEFAULT_BACKBONE, load_features, run_name
from rwkv_system1.pipeline.config import Config
from rwkv_system1.task_registry import REAL_TASKS
cfg = Config()
counts = {}
for k in REAL_TASKS:
    fs = load_features(DEFAULT_BACKBONE, k, "test", cfg)
    names = {run_name(DEFAULT_BACKBONE, k, dataclasses.replace(cfg, seed=s), fs) for s in (0, 1, 2)}
    counts[k] = [n for n in names if not (RUNS / f"{n}.pt").exists()]
bad = {k: v for k, v in counts.items() if v}
print(f"   tasks missing an ensemble member: {bad or 'none'}")
if bad:
    raise SystemExit(f"the ensemble is incomplete: {bad}")

print("\nB. the 24 non-MASSIVE tasks must reproduce seed_ensemble.py's ensemble arm exactly")
# `seed_ensemble.json` rows carry `single` / `ensemble` directly (joint_ensemble.json nests them
# under `arms`), so reading the wrong key here would silently compare against nothing.
harn = {r["task"]: r["ensemble"] for r in load("experiments/results/seed_ensemble.json")}
solo = [k for k in harn if not k.startswith("massive_")]
print(f"   {len(harn)} harness rows, {len(ens)} pipeline rows, missing: "
      f"{[k for k in harn if k not in ens] or 'none'}")
for metric in ("accuracy", "ece", "nll"):
    d = [ens[k][metric] - harn[k][metric] for k in solo]
    scale = 100 if metric != "nll" else 1
    worst = max(abs(x) for x in d) * scale
    print(f"   {metric:9} mean {st.mean(d) * scale:+.4f}  max|d| {worst:.4f}  over {len(solo)}")
    if worst > 1e-6:
        raise SystemExit(f"the pipeline's ensemble does not reproduce the harness on {metric}")

print("\nC. MASSIVE differs by design, and the difference is the joint head")
# The harness averaged three *per-task* heads here; this package serves one joint head per seed.
je = {r["task"]: r["arms"]["joint_ens"] for r in load("experiments/results/joint_ensemble.json")}
for metric in ("accuracy", "nll"):
    d_per_task = [ens[k][metric] - harn[k][metric] for k in je]
    d_joint = [ens[k][metric] - je[k][metric] for k in je]
    scale = 100 if metric != "nll" else 1
    print(f"   {metric:9} vs per-task ensemble {st.mean(d_per_task) * scale:+.3f}   "
          f"vs joint ensemble {st.mean(d_joint) * scale:+.3f} "
          f"(max|d| {max(abs(x) for x in d_joint) * scale:.4f})")
    if max(abs(x) for x in d_joint) * scale > 1e-6:
        raise SystemExit(f"joint+ensemble does not reproduce joint_ensemble.py on {metric}")

print("\nD. how much of any disagreement is retrain drift, not the ensemble")
old = load("experiments/results/eval_shipped.json")
moved = [(k, old[k]["accuracy"], one[k]["accuracy"]) for k in one
         if k in old and old[k]["accuracy"] != one[k]["accuracy"]]
same = len(one) - len(moved)
print(f"   seed-0 single head, old table vs retrained: {same}/{len(one)} tasks bit-identical,"
      f" {len(moved)} moved")
for k, a, b in moved[:12]:
    print(f"     {k:<20}{a * 100:>8.2f}% ->{b * 100:>8.2f}%  {(b - a) * 100:+.2f}pt")
PY
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "ACCEPTANCE_RC=$rc"

echo "--- 4. what the ensemble bought, task by task"
$PY - <<'PY'
import json, statistics as st
from pathlib import Path
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
one = load("experiments/results/eval_shipped_seed0.json")
ens = load("experiments/results/eval_shipped_ens3.json")
print(f"{'task':<20}{'acc s0':>9}{'acc ens':>10}{'dpt':>7}   {'nll s0':>8}{'nll ens':>9}{'dnll':>8}")
da, dn = [], []
for k in one:
    a, n = ens[k]["accuracy"] - one[k]["accuracy"], ens[k]["nll"] - one[k]["nll"]
    da.append(a); dn.append(n)
    print(f"{k:<20}{one[k]['accuracy'] * 100:>8.2f}%{ens[k]['accuracy'] * 100:>9.2f}%{a * 100:>+7.2f}"
          f"   {one[k]['nll']:>8.4f}{ens[k]['nll']:>9.4f}{n:>+8.4f}")
print(f"\nmean accuracy {st.mean(da) * 100:+.2f}pt   mean nll {st.mean(dn):+.4f} nats"
      f"   acc wins {sum(x > 0 for x in da)}/{len(da)}   nll wins {sum(x < 0 for x in dn)}/{len(dn)}")
print("suite mean acc: single", round(st.mean(v['accuracy'] for v in one.values()), 4),
      " ensemble", round(st.mean(v['accuracy'] for v in ens.values()), 4))
PY
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "DELTA_RC=$rc"
else
  echo "SKIPPING acceptance and delta -- a table is missing"
fi
echo "ENSEMBLE_LANDING_RC=$S"
