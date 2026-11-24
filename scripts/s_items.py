"""Do the training scripts published with PredictBeforeExecute (MLE-bench: denoising-dirty-documents, ventilator-pressure-prediction,
statoil-iceberg-classifier-challenge, stanford-covid-vaccine) need a GPU, what do they take and score on ours, how much does the score of a job vary with its seed and what does it lose to fewer epochs, and what does one epoch of job a on a tenth
of the training images cost and tell on four CPU threads of this machine?

python -m scripts.s_items gpu <n> [job ...]        each job as published, on each GPU of the pool, n times
python -m scripts.s_items sweep <n> [job ...]      each job on each GPU of the pool at each of its budgets of epochs, for seeds 42, 1, ..., n-1;
                                                   a run that out/items/items.jsonl already has is not made again
python -m scripts.s_items cpu <variant> <minutes>  job a: full | epoch128 | epoch420 here, on 4 threads and 8 GB
Appends to out/items/items.jsonl, keeps logs in out/items/logs/ and uploads both.

The jobs are styx.items.JOBS; the CPU limits and environment are styx.agents.CAGE and CPU_ENV.
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

from styx import agents, data, items, pool

OUT = "out/items"


def said(log):
    val = re.findall(r"Val RMSE: ([0-9.]+)", log)
    best = re.findall(r"Best validation RMSE: ([0-9.]+)", log)
    return dict(epochs=len(val), val_rmse_first=float(val[0]) if val else None, val_rmse_best=float(best[0]) if best else None)


def save(row):
    with open(f"{OUT}/items.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print({k: v for k, v in row.items() if k != "lines"}, flush=True)


def big(job):
    """The job's task and the public files of its input/ that `styx post` does not upload, being over 50 MB."""
    task = items.JOBS[job][0]
    public = f"{items.prepared(task)}/public"
    return task, [n for n in items.TASKS[task]["inputs"] if os.path.isfile(f"{public}/{n}") and os.path.getsize(f"{public}/{n}") > 50e6]


def place(job):
    """Puts those files on every node of the pool, under ~/data/<task>/; rsync sends only what is not there yet."""
    task, names = big(job)
    for g in pool.POOL if names else []:
        subprocess.run(["rsync", "-az", f"--rsync-path=mkdir -p data/{task} && rsync", *(f"{items.prepared(task)}/public/{n}" for n in names), f"{pool.POOL[g]['cluster']}:data/{task}/"], check=True)


def command(job):
    """What the job is posted with: links in input/ to the files place() put on the node, then the script."""
    task, names = big(job)
    return " && ".join([f"ln -sf ~/data/{task}/{n} input/{n}" for n in names] + ["python solution.py"])


def gpu(n, jobs):
    for job in jobs:
        d = items.item(job)
        place(job)
        for r, log in pool.ours(f"{OUT}/ledger.jsonl", [(g, command(job)) for g in pool.POOL] * n, d):
            r.update(where=r["gpu"], script=job, variant="full", **said("\n".join(log)))
            if r["state"] == "SUCCEEDED":
                subprocess.run(["rsync", f"{pool.POOL[r['gpu']]['cluster']}:jobs/{r['uid']}/submission/submission.csv", f"{d}/submission.csv"], check=True)
                r["test_score"] = items.score(job, f"{d}/submission.csv")
            save(r)


def sweep(n, jobs):
    todo, ds = [], {}
    have = {(r["script"], r["gpu"], r["seed"], r["epochs"]) for r in items.lines(f"{OUT}/items.jsonl") if r.get("variant") == "sweep" and r.get("state") == "SUCCEEDED"}
    for job in jobs:
        for seed in [42] + list(range(1, n)):
            for epochs, _, _ in items.BUDGETS[job]:
                lacking = [g for g in pool.POOL if (job, g, seed, epochs) not in have]
                if lacking:
                    d = items.item(job, seed=seed, epochs=epochs)
                    ds[d] = (job, seed, epochs)
                    todo += [(g, command(job), d) for g in lacking]
        place(job)
    for r, log in pool.ours(f"{OUT}/ledger.jsonl", todo, None):
        d = r.pop("cwd")
        job, seed, epochs = ds[d]
        r.update(where=r["gpu"], script=job, variant="sweep", **said("\n".join(log)))
        r.update(seed=seed, epochs=epochs)  # in place of said()'s count of the epochs in the log
        if r["state"] == "SUCCEEDED":
            subprocess.run(["rsync", "-z", f"{pool.POOL[r['gpu']]['cluster']}:jobs/{r['uid']}/submission/submission.csv", f"{d}/submission.csv"], check=True)
            r["test_score"] = items.score(job, f"{d}/submission.csv")
            os.remove(f"{d}/submission.csv")
        save(r)
    for d in ds:
        shutil.rmtree(d)


def cpu(variant, minutes):
    d, t, lines = items.item("a", variant), time.time(), []
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
    row = dict(where="cpu", script="a", variant=variant, cpu=model, threads=4, limit_s=minutes * 60, seconds=seconds, exit=code, timeout=code == -signal.SIGKILL,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, **said(log),
               lines=[(s, l) for s, l in lines if "Epoch" in l or "resolution" in l or "complete" in l])
    if code == 0:
        row["test_score"] = items.score("a", f"{d}/submission/submission.csv")
    save(row)


if __name__ == "__main__":
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    if sys.argv[1] == "cpu":
        cpu(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "gpu":
        gpu(int(sys.argv[2]), sys.argv[3:] or list(items.JOBS))
    else:
        sweep(int(sys.argv[2]), sys.argv[3:] or list(items.BUDGETS))
    data.upload(OUT, f"items/{open('COMMIT').read().strip()}", ["*.jsonl", "logs/*"])
