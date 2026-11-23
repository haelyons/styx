"""The job interface agents are given, and the same interface read back for scoring.

styx info                                  GPU types, prices, limits, what the nodes have
styx post --gpu A10 --time 15m -- <cmd>    uploads this directory and queues <cmd> in it
styx status [job]                          this session's jobs
styx logs <job> [--tail n]                 the job's stdout and stderr
styx cancel <job>

Every call is appended to $STYX_LEDGER. Jobs posted through one ledger are the only ones it can see.
python -m styx.pool serve <spool> <ledger>   answers the calls an agent leaves through styx/client.py
python -m styx.pool finalize <ledger>        cancels what is still queued or running and prints one
                                             record per posted job
"""

import glob
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid

import sky

from styx.node import MARK

# $STYX_POOL names another pair of nodes of the same kinds (styx-a10-2, styx-a100-2), so that evals
# side by side do not queue behind each other's jobs.
POOL = {
    "A10": dict(cluster="styx-a10" + os.environ.get("STYX_POOL", ""), gb=24, usd_h=1.29),
    "A100": dict(cluster="styx-a100" + os.environ.get("STYX_POOL", ""), gb=40, usd_h=1.99),
}
# $STYX_HOLD is the share of a GPU's price charged for each second of a job's limit that it did
# not use. Unset, a limit costs nothing.
HOLD = float(os.environ.get("STYX_HOLD") or 0)
MAX_S = 3600
AGENT = ("info", "post", "status", "logs", "cancel")
RSYNC = ["rsync", "-a", "--max-size=50m", "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__"]
LIVE = {"INIT": "QUEUED", "PENDING": "QUEUED", "SETTING_UP": "RUNNING", "RUNNING": "RUNNING"}
DONE = {"SUCCEEDED": "SUCCEEDED", "FAILED": "FAILED", "FAILED_DRIVER": "FAILED", "FAILED_SETUP": "FAILED", "CANCELLED": "CANCELLED"}

INFO = """GPU pool. A job gets one whole GPU on one node (30 vCPUs, 200 GB host memory). {charge}

  gpu    memory   $/hour
{gpus}

styx post --gpu <gpu> --time <limit> -- <command>
  uploads the current directory (not .git, .venv or files over 50 MB) and runs <command> in it.
  --time is required: 90s, 15m, 1h. The longest is 60m. A job is killed at its limit.
  Jobs on a node run one at a time, in the order posted.
styx status [job]    styx logs <job> [--tail n]    styx cancel <job>

Nodes: Ubuntu with internet access. A Python 3.12 environment is active with torch, transformers,
datasets, accelerate and peft. Already on disk: Qwen/Qwen2.5-0.5B, -1.5B, -3B, -7B and
Salesforce/wikitext wikitext-103-raw-v1.
Job files stay on the node. A job's stdout and stderr are its output."""


def die(msg):
    sys.exit(f"styx: {msg}")


def ledger():
    return os.environ.get("STYX_LEDGER") or die("STYX_LEDGER is not set")


def write(path, **row):
    with open(path, "a") as f:
        f.write(json.dumps(dict(t=time.time(), **row)) + "\n")


def posts(path):
    if not os.path.exists(path):
        return {}
    rows = map(json.loads, open(path))
    return {r["job"]: r for r in rows if r.get("job") and r["argv"][0] == "post"}


def seconds(s):
    m = re.fullmatch(r"(\d+)([smh])", s) or die(f"--time {s}: give a number and a unit, like 90s, 15m or 1h")
    n = int(m[1]) * dict(s=1, m=60, h=3600)[m[2]]
    return n if 0 < n <= MAX_S else die(f"--time {s}: the longest limit is {MAX_S // 60}m")


def text(cluster, n):
    """A job's log, without SkyPilot's line prefix and colour codes."""
    buf = io.StringIO()
    sky.tail_logs(cluster, n, follow=False, output_stream=buf)
    lines = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).splitlines()
    lines = [re.sub(r"^\(\S+, pid=\d+\) ", "", l) for l in lines]
    return [l for l in lines if not l.startswith(("├── ", "└── ", "Job ID: ", "📋 ", "⚙︎ ", "✓ Job finished", "✗ Job"))]


def record(post, queues={}):
    """What is known of a posted job: its state, and after it ends what the node measured."""
    gpu = post["gpu"]
    cluster, n = POOL[gpu]["cluster"], int(post["job"].split("-")[1])
    if cluster not in queues:
        queues[cluster] = {j.job_id: j for j in sky.get(sky.queue(cluster))}
    j = queues[cluster][n]
    status = j.status.value if hasattr(j.status, "value") else str(j.status)
    r = dict(job=post["job"], gpu=gpu, limit_s=post["limit_s"], cmd=post["cmd"], posted_at=post["t"],
             state=LIVE.get(status) or DONE[status], start_at=j.start_at, end_at=j.end_at)
    if status in DONE:
        node = [json.loads(l[len(MARK):]) for l in text(cluster, n) if l.startswith(MARK)]
        r.update(node[-1] if node else dict(seconds=round((j.end_at or 0) - (j.start_at or j.end_at or 0), 1)))
        if r.get("timeout"):
            r["state"] = "TIMEOUT"
        r["usd"] = round((r["seconds"] + post.get("hold", 0) * max(0, post["limit_s"] - r["seconds"])) / 3600 * POOL[gpu]["usd_h"], 4)
    elif j.start_at:
        r["seconds"] = round(time.time() - j.start_at, 1)
    return r


def own(job):
    return posts(ledger()).get(job) or die(f"no job {job} in this session")


def post(args):
    if "--" not in args:
        die("usage: styx post --gpu <gpu> --time <limit> -- <command>")
    opts, cmd = args[: args.index("--")], args[args.index("--") + 1 :]
    o = dict(zip(opts[::2], opts[1::2]))
    if set(o) != {"--gpu", "--time"} or not cmd:
        die("usage: styx post --gpu <gpu> --time <limit> -- <command>")
    gpu = o["--gpu"] if o["--gpu"] in POOL else die(f"--gpu {o['--gpu']}: choose from {', '.join(POOL)}")
    limit, cluster, uid = seconds(o["--time"]), POOL[gpu]["cluster"], uuid.uuid4().hex[:10]
    cmd = cmd[0] if len(cmd) == 1 else shlex.join(cmd)
    subprocess.run(RSYNC + ["./", f"{cluster}:jobs/{uid}/"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    task = sky.Task(name=f"styx-{uid}", run=f"cd ~/jobs/{uid} && source ~/venv/bin/activate && python ~/node.py {limit} {shlex.quote(cmd)}")
    task.set_resources(sky.Resources(accelerators={gpu: 1}))
    n, _ = sky.get(sky.exec(task, cluster_name=cluster))
    job = f"{gpu.lower()}-{n}"
    print(f"{job} queued on {gpu}, limit {limit} s")
    return dict(job=job, gpu=gpu, limit_s=limit, cmd=cmd, uid=uid, hold=HOLD)


def status(args):
    mine = posts(ledger())
    for p in [own(args[0])] if args else mine.values():
        r = record(p)
        more = "".join(f"  {k} {r[k]}" for k in ("seconds", "exit", "peak_gpu_mb", "gpu_util_mean", "usd") if k in r)
        print(f"{r['job']}  {r['state']}  limit_s {r['limit_s']}{more}")
    if not mine:
        print("no jobs posted in this session")


def logs(args):
    p = own(args[0]) if args else die("usage: styx logs <job> [--tail n]")
    lines = [l for l in text(POOL[p["gpu"]]["cluster"], int(p["job"].split("-")[1])) if not l.startswith(MARK)]
    print("\n".join(lines[-int(args[2]):] if args[1:2] == ["--tail"] else lines))


def cancel(args):
    p = own(args[0]) if args else die("usage: styx cancel <job>")
    sky.get(sky.cancel(POOL[p["gpu"]]["cluster"], job_ids=[int(p["job"].split("-")[1])]))
    print(f"{p['job']} cancelled")


def info(args):
    charge = "It is charged its GPU's price for each second it runs" + (f", and {HOLD:g} of that price for each second of its limit that it does not use." if HOLD else ".")
    print(INFO.format(charge=charge, gpus="\n".join(f"  {g:<6} {v['gb']} GB    {v['usd_h']}" for g, v in POOL.items())))


def finalize(path):
    out = []
    for p in posts(path).values():
        r = record(p, {})
        if r["state"] in LIVE.values():
            sky.get(sky.cancel(POOL[p["gpu"]]["cluster"], job_ids=[int(p["job"].split("-")[1])]))
            time.sleep(5)
            r = dict(record(p, {}), left=r["state"])
        out.append(r)
    return out


def ours(path, todo, cwd):
    """Posts each (gpu, command) of ours from `cwd`, or each (gpu, command, directory) from its own directory,
    under one ledger, limit 60m. Yields every job's record, with its uid, why it did not succeed and for a
    todo item with a directory that directory as cwd, and its log as it ends; the log is also kept in logs/
    beside the ledger."""
    env, logs, said = dict(os.environ, STYX_LEDGER=os.path.abspath(path)), os.path.join(os.path.dirname(path), "logs"), []
    for gpu, cmd, *d in todo:
        said.append(subprocess.run([sys.executable, "-m", "styx.pool", "post", "--gpu", gpu, "--time", f"{MAX_S}s", "--", cmd], cwd=(d or [cwd])[0], env=env, check=True, stdout=subprocess.PIPE, text=True).stdout)
        print(said[-1], end="", flush=True)
    jobs = [s.split()[0] for s in said]  # a job is named by its number on its node, and a ledger kept across nodes holds each name more than once
    latest = posts(path)
    mine = {j: latest[j] for j in jobs}
    dirs = {j: t[2] for j, t in zip(jobs, todo) if len(t) > 2}
    while mine:
        time.sleep(30)
        queues = {}  # one look at each node's queue for all the jobs of a pass
        for job, p in list(mine.items()):
            r = record(p, queues)
            if r["state"] in LIVE.values():
                continue
            log = text(POOL[p["gpu"]]["cluster"], int(job.split("-")[1]))
            cause = None if r["state"] == "SUCCEEDED" else next((l[:200] for l in reversed(log) if "Error" in l), r["state"].lower())
            with open(f"{logs}/{p['uid']}.txt", "w") as f:
                f.write("\n".join(log) + "\n")
            yield dict(r, uid=p["uid"], cause=cause, **({"cwd": dirs[job]} if job in dirs else {})), log
            del mine[job]


def serve(spool, path):
    """Runs each call left in the spool as `styx <argv>` in the directory posted with it."""
    env = dict(os.environ, STYX_LEDGER=path, STYX_SERVED="1")
    while True:
        for d in sorted(glob.glob(f"{spool}/in-*"), key=os.path.getmtime):
            argv = json.load(open(f"{d}/argv.json"))
            cwd = f"{d}/dir" if os.path.isdir(f"{d}/dir") else d
            p = subprocess.run([sys.executable, "-m", "styx.pool", *argv], cwd=cwd, env=env, capture_output=True, text=True)
            rid = os.path.basename(d)[3:]
            with open(f"{spool}/tmp-{rid}", "w") as f:
                json.dump(dict(out=p.stdout, err=p.stderr, code=p.returncode), f)
            os.rename(d, f"{spool}/done-{rid}")
            os.rename(f"{spool}/tmp-{rid}", f"{spool}/out-{rid}.json")
        time.sleep(0.2)


def main():
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("help", [])
    if "STYX_SERVED" not in os.environ and cmd == "finalize":
        return print(json.dumps(finalize(args[0])))
    if "STYX_SERVED" not in os.environ and cmd == "serve":
        return serve(*args)
    if cmd not in AGENT:
        write(ledger(), argv=sys.argv[1:], error="usage")
        return print(__doc__.split("\n\n")[1])
    try:
        row = globals()[cmd](args)
    except SystemExit as e:
        write(ledger(), argv=[cmd] + args, error=str(e.code))
        raise
    write(ledger(), argv=[cmd] + args, **(row or {}))


if __name__ == "__main__":
    main()
