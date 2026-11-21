"""Does the denoising script published with PredictBeforeExecute (MLE-bench denoising-dirty-documents;
test RMSE 0.03719 in 324 s on an RTX 3090 there) need a GPU, and what does one epoch of it on a
tenth of the training images cost and tell on four CPU threads of this machine?

python -m scripts.s_denoise gpu <n>                  the script as published, on each GPU of the pool, n times
python -m scripts.s_denoise cpu <variant> <minutes>  full | epoch128 | epoch420 here, on 4 threads and 8 GB
Appends to out/denoise/denoise.jsonl, keeps logs in out/denoise/logs/ and uploads both.

Needs ~/.cache/styx/data/data/denoising-dirty-documents, which is that folder of tasks/data.zip in the
HF dataset zjunlp/PredictBeforeExecute, and ~/.cache/styx/cpu-env: Python 3.12 with the CPU wheels of
torch 2.11.0 and torchvision 0.26.0, scikit-learn, pandas and pillow.
"""

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

import pandas as pd
from huggingface_hub import hf_hub_download

from styx import agents, data, pool

HUB = "zjunlp/PredictBeforeExecute"
SCRIPT = ("solutions_subset_15/denoising-dirty-documents/code/solution_Automind_wo_knowledge_denoising-dirty-documents_"
          "13a296eb-c605-4737-bea2-edd07d49161a_run_c4121e82042c4af2b8755e35432adc9d.py")
DATA = f"{agents.ROOT}/data/data/denoising-dirty-documents/prepared"
PYTHON = f"{agents.ROOT}/cpu-env/bin/python"
OUT = "out/denoise"
CAGE = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=8G", "-p", "MemorySwapMax=0", "taskset", "-c", "0-3"]
# ArchPilot's proxy: the same code for one epoch on a tenth of the training data. The script trains
# at three resolutions, so the epoch is taken at its first and at its last.
MARK = "    # Progressive resizing training\n"
TENTH = [
    (MARK, "    train_files, train_clean_files = train_files[: len(train_files) // 10], train_clean_files[: len(train_files) // 10]\n" + MARK),
    ("epochs_per_resolution = [5, 5, 10]", "epochs_per_resolution = [1]"),
    ("criterion, epochs=10\n", "criterion, epochs=0\n"),
]
ALL = "resolutions = [(128, 128), (256, 256), (420, 540)]"
VARIANTS = dict(full=[], epoch128=TENTH + [(ALL, "resolutions = [(128, 128)]")], epoch420=TENTH + [(ALL, "resolutions = [(420, 540)]")])


def job(variant):
    """A directory holding the script, changed as the variant says, and the task's public data."""
    d = f"{agents.ROOT}/denoise/{variant}"
    shutil.rmtree(d, ignore_errors=True)
    code = open(hf_hub_download(HUB, SCRIPT, repo_type="dataset", token=False)).read()
    for old, new in VARIANTS[variant]:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    for sub in ("train", "train_cleaned", "test"):
        shutil.copytree(f"{DATA}/public/{sub}", f"{d}/input/{sub}")
    with open(f"{d}/solution.py", "w") as f:
        f.write(code)
    return d


def rmse(submission):
    """MLE-bench's grade for this task: RMSE over the pixels, rows matched by id."""
    s, a = (pd.read_csv(f).sort_values("id") for f in (submission, f"{DATA}/private/answers.csv"))
    assert (s["id"].values == a["id"].values).all()
    return round(float(((s["value"].values - a["value"].values) ** 2).mean() ** 0.5), 5)


def said(log):
    val = re.findall(r"Val RMSE: ([0-9.]+)", log)
    best = re.findall(r"Best validation RMSE: ([0-9.]+)", log)
    return dict(epochs=len(val), val_rmse_first=float(val[0]) if val else None, val_rmse_best=float(best[0]) if best else None)


def save(row):
    with open(f"{OUT}/denoise.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print({k: v for k, v in row.items() if k != "lines"}, flush=True)


def gpu(n):
    d = job("full")
    for r, log in pool.ours(f"{OUT}/ledger.jsonl", [(g, "python solution.py") for g in pool.POOL] * n, d):
        r.update(where=r["gpu"], variant="full", **said("\n".join(log)))
        if r["state"] == "SUCCEEDED":
            subprocess.run(["rsync", f"{pool.POOL[r['gpu']]['cluster']}:jobs/{r['uid']}/submission/submission.csv", f"{d}/submission.csv"], check=True)
            r["test_rmse"] = rmse(f"{d}/submission.csv")
        save(r)


def cpu(variant, minutes):
    d, t, lines = job(variant), time.time(), []
    p = subprocess.Popen(CAGE + [PYTHON, "-u", "solution.py"], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
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
    row = dict(where="cpu", variant=variant, cpu=model, threads=4, limit_s=minutes * 60, seconds=seconds, exit=code, timeout=code == -signal.SIGKILL,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, **said(log),
               lines=[(s, l) for s, l in lines if "Epoch" in l or "resolution" in l or "complete" in l])
    if code == 0:
        row["test_rmse"] = rmse(f"{d}/submission/submission.csv")
    save(row)


if __name__ == "__main__":
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    gpu(int(sys.argv[2])) if sys.argv[1] == "gpu" else cpu(sys.argv[2], int(sys.argv[3]))
    data.upload(OUT, f"denoise/{open('COMMIT').read().strip()}", ["*.jsonl", "logs/*"])
