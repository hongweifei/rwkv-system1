set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1

# 1) go_emotions: the second corpus with a real annotator distribution (the first is
#    typed-decisions), so the posterior metrics and the Bayes ceiling get a second task.
# 2) amazon_reviews_multi x6: ordinal score, in six declared languages.
# 3) sst5 / boolq / prompt_injections: the rest of the suite Laya publish a table for.
NEW=go_emotions,sst5,boolq,prompt_injections,amazon_reviews_en,amazon_reviews_de,amazon_reviews_es,amazon_reviews_fr,amazon_reviews_ja,amazon_reviews_zh
# Every real task, from the registry rather than a literal -- a hand-written list of 34 names
# is one typo away from silently evaluating 33 of them.
ALL=$($PY -c "from rwkv_system1.task_registry import REAL_TASKS; print(','.join(REAL_TASKS))")
echo "will evaluate $(echo $ALL | tr ',' '\n' | wc -l) tasks"
S=0

echo "waiting for the declared-languages chain ..."
if [ ! -f D:/tmp/declared.log ]; then
  echo "no third-chain log found -- running immediately"
else
  for i in $(seq 1 400); do
    if grep -q "DECLARED_STATUS=" D:/tmp/declared.log 2>/dev/null; then
      echo "third chain finished: $(grep DECLARED_STATUS= D:/tmp/declared.log | tail -1)"
      break
    fi
    sleep 60
  done
fi

echo "########## 1) extract the ten additions ##########"
$PY -m rwkv_system1.pipeline extract --tasks $NEW
rc=$?; [ $rc -ne 0 ] && S=1; echo "STEP1_RC=$rc"

echo "########## 2) train a head per addition ##########"
$PY -m rwkv_system1.pipeline train --tasks $NEW
rc=$?; [ $rc -ne 0 ] && S=2; echo "STEP2_RC=$rc"

echo "########## 3) eval every real task ##########"
$PY -m rwkv_system1.pipeline eval --tasks $ALL
rc=$?; [ $rc -ne 0 ] && S=3; echo "STEP3_RC=$rc"

echo "ADDITIONS_STATUS=$S"
