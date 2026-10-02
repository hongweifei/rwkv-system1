"""Stop the three queued chains as soon as chain 1 reports its status.

They are idle -- parked in a `sleep 60` poll waiting for their predecessor's status line -- so
killing them loses nothing and prevents the next chain from starting its extraction while the
package is being reorganised.  Polls every 5 s rather than 60: chain 2 waits on the same status
line with a 60 s poll, and whoever sees it first wins, so a slow poll here means killing chain 2
mid-write instead of before it starts.
"""
import re
import subprocess
import time

CHAIN_LOG = "D:/tmp/six_task.log"
WAITERS = ("remaining_run.sh", "declared_langs_run.sh", "additions_run.sh")
DEADLINE = time.time() + 6 * 3600


def waiters() -> list[int]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    pids = []
    for line in out.splitlines():
        if any(w in line for w in WAITERS):
            match = re.match(r"\s*(\d+)\s", line)
            if match:
                pids.append(int(match.group(1)))
    return pids


while time.time() < DEADLINE:
    try:
        text = open(CHAIN_LOG, encoding="utf-8", errors="replace").read()
    except OSError:
        text = ""
    if "SIX_TASK_STATUS=" in text:
        status = re.search(r"SIX_TASK_STATUS=(\d+)", text).group(1)
        print(f"chain 1 finished with status {status}; stopping the queued chains")
        for pid in waiters():
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            print(f"   killed pid {pid}")
        print("queued chains stopped -- the machine is idle for the refactor")
        raise SystemExit(0)
    time.sleep(5)
print("timed out waiting for chain 1")
raise SystemExit(1)
