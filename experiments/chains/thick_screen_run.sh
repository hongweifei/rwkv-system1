set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1

# Wait for the cap-raise chain's last line before starting.  This screen loads the backbone and runs
# forwards, and two of those at once is what took the box down on 2026-09-23 (16 GB and a small page
# file, commit charge exhausted).  Bounded at eight hours, then it proceeds on whatever state the
# tree is in -- the screen only needs the caches for its `last` arm, so a partial suite still works.
for _ in $(seq 1 480); do
  grep -q "^SUITE_STATUS=" experiments/results/suite_run5.log && break
  sleep 60
done

echo "########## thick-feature screen: massive_zh, last vs pooled vs two_layer ##########"
echo "(30 minutes: the `last` arm reads its cache, the other two each extract 800 rows)"
$PY experiments/harnesses/thick_features.py massive_zh
echo "SCREEN_RC=$?"
echo "THICK_SCREEN_DONE"
