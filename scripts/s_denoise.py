"""Do the denoising scripts published with PredictBeforeExecute (MLE-bench denoising-dirty-documents)
need a GPU, what do they take and score on ours, and what does one epoch of the first on a tenth
of the training images cost and tell on four CPU threads of this machine?

python -m scripts.s_denoise gpu <n> [job ...]        each job as published, on each GPU of the pool, n times
python -m scripts.s_denoise cpu <variant> <minutes>  job a: full | epoch128 | epoch420 here, on 4 threads and 8 GB
Appends to out/denoise/denoise.jsonl, keeps logs in out/denoise/logs/ and uploads both.

The jobs are styx.items.denoise; the CPU limits and environment are styx.agents.CAGE and CPU_ENV.
"""

import json
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time

from styx import agents, data, items, pool

OUT = "out/denoise"


def said(log):
    val = re.findall(r"Val RMSE: ([0-9.]+)", log)
    best = re.findall(r"Best validation RMSE: ([0-9.]+)", log)
    return dict(epochs=len(val), val_rmse_first=float(val[0]) if val else None, val_rmse_best=float(best[0]) if best else None)


def save(row):
    with open(f"{OUT}/denoise.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print({k: v for k, v in row.items() if k != "lines"}, flush=True)


def gpu(n, jobs):
    for job in jobs:
        d = items.denoise(job)
        for r, log in pool.ours(f"{OUT}/ledger.jsonl", [(g, "python solution.py") for g in pool.POOL] * n, d):
            r.update(where=r["gpu"], job=job, variant="full", **said("\n".join(log)))
            if r["state"] == "SUCCEEDED":
                subprocess.run(["rsync", f"{pool.POOL[r['gpu']]['cluster']}:jobs/{r['uid']}/submission/submission.csv", f"{d}/submission.csv"], check=True)
                r["test_rmse"] = items.rmse(f"{d}/submission.csv")
            save(r)


def cpu(variant, minutes):
    d, t, lines = items.denoise("a", variant), time.time(), []
    p = subprocess.Popen(agents.CAGE + [f"{agents.CPU_ENV}/bin/python", "-u", "solution.py"], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    clock = threading.Timer(minutes * 60, lambda: os.killpg(p.pid, signal.SIGKILL))
    clock.start()
    for l in p.stdout:
        lines.append((round(time.time() - t, 1), l.rstrip()))
    code, seconds = p.wait(), round(time.time() - t, 1)
    clock.cancel()
    log = "\n".join(l for _, l in lines)
    with open(f"{OUT}/logs/cpu-{variant}-{int(t)}.txt", "w") as f:
        f.write("\n".join(f"{s:8.1f}  {l}" for s, l in lines) + "\n")
    model = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    row = dict(where="cpu", job="a", variant=variant, cpu=model, threads=4, limit_s=minutes * 60, seconds=seconds, exit=code, timeout=code == -signal.SIGKILL,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, **said(log),
               lines=[(s, l) for s, l in lines if "Epoch" in l or "resolution" in l or "complete" in l])
    if code == 0:
        row["test_rmse"] = items.rmse(f"{d}/submission/submission.csv")
    save(row)


if __name__ == "__main__":
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    gpu(int(sys.argv[2]), sys.argv[3:] or list(items.JOBS)) if sys.argv[1] == "gpu" else cpu(sys.argv[2], int(sys.argv[3]))
    data.upload(OUT, f"denoise/{open('COMMIT').read().strip()}", ["*.jsonl", "logs/*"])
