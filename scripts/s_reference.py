"""What does each cell of tasks/post.py cost when we post it ourselves, on each GPU of the pool, at
each batch size given?

python -m scripts.s_reference [batch ...]     appends to out/reference.jsonl
"""

import json
import os
import subprocess
import sys
import time

from styx import pool
from tasks.post import CELLS

LEDGER, LIMIT = "out/reference-ledger.jsonl", "60m"


def main():
    batches = list(map(int, sys.argv[1:])) or [16]
    os.makedirs("out", exist_ok=True)
    env = dict(os.environ, STYX_LEDGER=os.path.abspath(LEDGER))
    before = set(pool.posts(LEDGER))
    for b in batches:
        for cell, (model, docs, ctx) in CELLS.items():
            for gpu in pool.POOL:
                cmd = f"python loss.py --model {model} --docs {docs} --ctx {ctx} --batch {b}"
                subprocess.run([sys.executable, "-m", "styx.pool", "post", "--gpu", gpu, "--time", LIMIT, "--", cmd], cwd="jobs", env=env, check=True)
    mine = {j: p for j, p in pool.posts(LEDGER).items() if j not in before}
    cells = {f"--model {m} --docs {d} --ctx {x} ": c for c, (m, d, x) in CELLS.items()}
    while mine:
        time.sleep(30)
        for job, p in list(mine.items()):
            r = pool.record(p, {})
            if r["state"] in pool.LIVE.values():
                continue
            log = pool.text(pool.POOL[p["gpu"]]["cluster"], int(job.split("-")[1]))
            find = lambda k: next((float(l.split()[1]) for l in log if l.startswith(k + " ")), None)
            r.update(cell=next(c for k, c in cells.items() if k in p["cmd"]), batch=int(p["cmd"].split()[-1]), loss=find("loss"), tok_s=find("tok_s"))
            with open("out/reference.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r["cell"], r["gpu"], r["batch"], r["state"], r.get("seconds"), r.get("peak_gpu_mb"), r["loss"], flush=True)
            del mine[job]


if __name__ == "__main__":
    main()
