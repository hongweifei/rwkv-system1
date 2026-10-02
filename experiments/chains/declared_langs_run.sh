set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1

# Only the languages the checkpoint declares: en zh fr es de pt ru it ja ko vi ar.  What that
# leaves out of these two benchmarks -- MASSIVE's tr hi ta sw, XNLI's tr hi ur th el bg sw -- is
# not run, because a number for an undeclared language measures extrapolation, not coverage.
NEW=massive_de,massive_fr,massive_es,massive_pt,massive_ru,massive_ar,massive_ja,massive_ko,xnli_de,xnli_fr,xnli_es,xnli_ru,xnli_ar,xnli_vi
ALL=laya_urgency,laya_category,laya_needs_human,ag_news,massive_en,massive_zh,massive_de,massive_fr,massive_es,massive_pt,massive_ru,massive_ar,massive_ja,massive_ko,xnli_en,xnli_zh,xnli_de,xnli_fr,xnli_es,xnli_ru,xnli_ar,xnli_vi,emotion,banking77
S=0

echo "waiting for the Chinese/English chain to report its status ..."
if [ ! -f D:/tmp/remaining.log ]; then
  echo "no second-chain log found -- running immediately"
else
  for i in $(seq 1 300); do
    if grep -q "REMAINING_STATUS=" D:/tmp/remaining.log 2>/dev/null; then
      echo "second chain finished: $(grep REMAINING_STATUS= D:/tmp/remaining.log | tail -1)"
      break
    fi
    sleep 60
  done
fi

echo "########## 1) extract the fourteen declared languages ##########"
$PY -m rwkv_system1.pipeline extract --tasks $NEW
rc=$?; [ $rc -ne 0 ] && S=1; echo "STEP1_RC=$rc"

echo "########## 2) train a head per language ##########"
$PY -m rwkv_system1.pipeline train --tasks $NEW
rc=$?; [ $rc -ne 0 ] && S=2; echo "STEP2_RC=$rc"

echo "########## 3) eval all twenty-four ##########"
$PY -m rwkv_system1.pipeline eval --tasks $ALL
rc=$?; [ $rc -ne 0 ] && S=3; echo "STEP3_RC=$rc"

echo "DECLARED_STATUS=$S"
