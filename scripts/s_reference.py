"""What does each cell of tasks/post.py cost when we post it ourselves, on each GPU of the pool, at
each batch size given? And how much does the cheapest of those vary when posted again?

python -m scripts.s_reference [batch ...]     every cell on every GPU at each batch size
python -m scripts.s_reference again <n>       each cell's cheapest posting, n more times
Appends to out/reference.jsonl, keeps each job's log in out/logs/ and uploads both.
"""

import json
import os
import subprocess
import sys
import time

from styx import data, pool
from tasks.post import CELLS, reference

LEDGER, LIMIT = "out/reference-ledger.jsonl", "60m"


def main():
    os.makedirs("out/logs", exist_ok=True)
    env = dict(os.environ, STYX_LEDGER=os.path.abspath(LEDGER))
    if sys.argv[1:2] == ["again"]:
        todo = [(r["gpu"], r["cmd"]) for r in reference().values()] * int(sys.argv[2])
    else:
        todo = [(gpu, f"python loss.py --model {model} --docs {docs} --ctx {ctx} --batch {b}")
                for b in list(map(int, sys.argv[1:])) or [16] for model, docs, ctx in CELLS.values() for gpu in pool.POOL]
    t0 = time.time()
    for gpu, cmd in todo:
        subprocess.run([sys.executable, "-m", "styx.pool", "post", "--gpu", gpu, "--time", LIMIT, "--", cmd], cwd="jobs", env=env, check=True)
    mine = {j: p for j, p in pool.posts(LEDGER).items() if p["t"] >= t0}
    cells = {f"--model {m} --docs {d} --ctx {x} ": c for c, (m, d, x) in CELLS.items()}
    while mine:
        time.sleep(30)
        for job, p in list(mine.items()):
            r = pool.record(p, {})
            if r["state"] in pool.LIVE.values():
                continue
            log = pool.text(pool.POOL[p["gpu"]]["cluster"], int(job.split("-")[1]))
            find = lambda k: next((float(l.split()[1]) for l in log if l.startswith(k + " ")), None)
            cause = None if r["state"] == "SUCCEEDED" else next((l[:200] for l in reversed(log) if "Error" in l), r["state"].lower())
            r.update(cell=next(c for k, c in cells.items() if k in p["cmd"]), batch=int(p["cmd"].split()[-1]), loss=find("loss"), tok_s=find("tok_s"), uid=p["uid"], cause=cause)
            with open(f"out/logs/{p['uid']}.txt", "w") as f:
                f.write("\n".join(log) + "\n")
            with open("out/reference.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r["cell"], r["gpu"], r["batch"], r["state"], r.get("seconds"), r.get("peak_gpu_mb"), r["loss"], flush=True)
            del mine[job]
    data.upload("out", f"reference/{open('COMMIT').read().strip()}", ["reference*.jsonl", "logs/*"])


if __name__ == "__main__":
    main()
