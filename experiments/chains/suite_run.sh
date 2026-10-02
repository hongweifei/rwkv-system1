set -u
cd /d/Projects/rwkv-system1
PY=".venv/Scripts/python.exe"
export PYTHONPATH=. PYTHONUNBUFFERED=1
S=0

ALL=$($PY -c "from rwkv_system1.task_registry import REAL_TASKS; print(','.join(REAL_TASKS))")
echo "tasks: $(echo "$ALL" | tr ',' '\n' | wc -l)"

# Extract two tasks per process.  `cmd_extract` builds every requested corpus *before* it
# extracts anything, so the chunk size is a memory bound, not a scheduling preference -- and the
# bound is real: on 2026-09-23 a five-task chunk over the MASSIVE languages took the whole box
# down.  What it looked like, in the order it appeared: `Segmentation fault` from the extractor,
# then `OSError: [Errno 22]` from `torch.load(..., mmap=True)` reading the checkpoint, then
# `OSError: [WinError 1455] ... shm.dll` importing torch at all, then `dwm.exe` and an unrelated
# app in the Windows event log dying with STATUS_INSUFFICIENT_RESOURCES.  Every one of those is
# the same cause -- 16 GB and a small page file running out of commit charge -- so the fix is to
# hold less at once, and to run this chain on its own rather than beside another heavy job.
# Re-running is safe and resumable: `extract` skips any split whose cache already matches.
echo "########## 1) extract every remaining split, two tasks per process ##########"
for grp in $($PY -c "
from rwkv_system1.task_registry import REAL_TASKS
ks = list(REAL_TASKS)
for i in range(0, len(ks), 2):
    print(','.join(ks[i:i + 2]))
"); do
  echo "--- $grp"
  $PY -m rwkv_system1.pipeline extract --tasks "$grp"
  rc=$?; [ $rc -ne 0 ] && S=$rc; echo "EXTRACT_RC=$rc"
done

# Train and eval only what is actually on disk.  `train --tasks $ALL` aborts on the first missing
# cache, so on 2026-09-23 one extraction that died turned into "no table at all" for the other
# thirty.  The gap is printed rather than hidden, and re-running the chain fills it: extraction is
# idempotent, so the next pass skips what this one finished.
eval "$($PY -c "
import glob, os
from rwkv_system1.cache_layout import DEFAULT_BACKBONE
from rwkv_system1.task_registry import REAL_TASKS
have = {}
for p in glob.glob('cache/*.pt'):
    n = os.path.basename(p)[:-3]
    for s in ('train', 'dev', 'test'):
        if '_' + s + '_' in n:
            have.setdefault(n.split('_' + s + '_')[0], set()).add(s)
full = [k for k in REAL_TASKS if have.get(DEFAULT_BACKBONE + '_' + k) == {'train', 'dev', 'test'}]
print('READY=' + ','.join(full))
print('MISSING=' + ','.join(k for k in REAL_TASKS if k not in full))
")"
echo "caches complete: $(echo "$READY" | tr ',' '\n' | grep -c .) / 34"
if [ -n "$MISSING" ]; then echo "NO CACHE, NOT TRAINED: $MISSING"; S=1; fi

echo "########## 2) train one head per task ##########"
$PY -m rwkv_system1.pipeline train --tasks "$READY"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "STEP2_RC=$rc"

echo "########## 3) eval every task ##########"
$PY -m rwkv_system1.pipeline eval --tasks "$READY"
rc=$?; [ $rc -ne 0 ] && S=$rc; echo "STEP3_RC=$rc"

echo "SUITE_STATUS=$S"
