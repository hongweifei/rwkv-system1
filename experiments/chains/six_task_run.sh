set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
OLD=laya_urgency,laya_category,laya_needs_human,ag_news
NEW=massive_en,banking77
ALL=$OLD,$NEW
S=0

echo "########## 1) re-extract the four existing splits under the new fingerprint ##########"
echo "    (max_suffix_tokens left the payload and ctx_tokens entered it, so every cache name"
echo "     moved even though no vector can differ -- the old blocks were 14-70 tokens, under 96)"
$PY -m rwkv_system1.pipeline extract --tasks $OLD
rc=$?; [ $rc -ne 0 ] && S=1; echo "STEP1_RC=$rc"

echo "########## 2) extract the two new corpora ##########"
$PY -m rwkv_system1.pipeline extract --tasks $NEW
rc=$?; [ $rc -ne 0 ] && S=2; echo "STEP2_RC=$rc"

echo "########## 3) train one head per task ##########"
$PY -m rwkv_system1.pipeline train --tasks $ALL
rc=$?; [ $rc -ne 0 ] && S=3; echo "STEP3_RC=$rc"

echo "########## 4) eval ##########"
$PY -m rwkv_system1.pipeline eval --tasks $ALL
rc=$?; [ $rc -ne 0 ] && S=4; echo "STEP4_RC=$rc"

echo "SIX_TASK_STATUS=$S"
