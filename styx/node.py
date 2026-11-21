"""Runs one job on a pool node: kills it at its time limit, samples the GPU once a second, and
prints one record line that styx/pool.py reads back from the job's log.

python node.py <limit seconds> <command>
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time

MARK = "STYX_RECORD "


def main():
    limit, cmd = int(sys.argv[1]), sys.argv[2]
    samples = []
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-l", "1"],
        stdout=subprocess.PIPE, text=True,
    )
    threading.Thread(target=lambda: samples.extend(tuple(map(int, l.split(","))) for l in smi.stdout), daemon=True).start()
    t = time.time()
    p = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
    try:
        code, timeout = p.wait(timeout=limit), False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        code, timeout = 124, True
        print(f"styx: killed at the time limit of {limit} s", flush=True)
    seconds = time.time() - t
    smi.terminate()
    n = max(len(samples), 1)
    print(MARK + json.dumps(dict(
        seconds=round(seconds, 1), exit=code, timeout=timeout,
        peak_gpu_mb=max((m for m, _ in samples), default=0),
        gpu_util_mean=round(sum(u for _, u in samples) / n, 1),
        gpu_busy_share=round(sum(u >= 10 for _, u in samples) / n, 3),
    )), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
