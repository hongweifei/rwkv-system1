set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
NEW=massive_zh,xnli_en,xnli_zh,emotion
ALL=laya_urgency,laya_category,laya_needs_human,ag_news,massive_en,massive_zh,xnli_en,xnli_zh,emotion,banking77
S=0

# The first chain is already using every core; wait for its final status line rather than
# competing with it, since two extractors on this box roughly halve each other's throughput.
echo "waiting for the six-task chain to report its status ..."
if [ ! -f D:/tmp/six_task.log ]; then
  echo "no first-chain log found -- running immediately"
else
for i in $(seq 1 300); do
  if grep -q "SIX_TASK_STATUS=" /tmp/six_task.log 2>/dev/null || \
     grep -q "SIX_TASK_STATUS=" D:/tmp/six_task.log 2>/dev/null; then
    echo "first chain finished: $(grep -h SIX_TASK_STATUS= D:/tmp/six_task.log | tail -1)"
    break
  fi
  sleep 60
done
fi

echo "########## 1) extract the Chinese/English additions + emotion ##########"
$PY -m rwkv_system1.pipeline extract --tasks $NEW
rc=$?; [ $rc -ne 0 ] && S=1; echo "STEP1_RC=$rc"

echo "########## 2) train one head per new task ##########"
$PY -m rwkv_system1.pipeline train --tasks $NEW
rc=$?; [ $rc -ne 0 ] && S=2; echo "STEP2_RC=$rc"

echo "########## 3) eval all ten ##########"
$PY -m rwkv_system1.pipeline eval --tasks $ALL
rc=$?; [ $rc -ne 0 ] && S=3; echo "STEP3_RC=$rc"

echo "REMAINING_STATUS=$S"
