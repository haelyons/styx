"""What does each cell of tasks/post.py cost when we post it ourselves, on each GPU of the pool, at
each batch size given? And how much does the cheapest of those vary when posted again?

python -m scripts.s_reference [batch ...]     every cell on every GPU at each batch size
python -m scripts.s_reference again <n>       each cell's cheapest posting, n more times
Appends to out/reference.jsonl, keeps each job's log in out/logs/ and uploads both.
"""

import json
import os
import sys

from styx import data, pool
from tasks.post import CELLS, reference

LEDGER = "out/reference-ledger.jsonl"


def main():
    os.makedirs("out/logs", exist_ok=True)
    if sys.argv[1:2] == ["again"]:
        todo = [(r["gpu"], r["cmd"]) for r in reference().values()] * int(sys.argv[2])
    else:
        todo = [(gpu, f"python loss.py --model {model} --docs {docs} --ctx {ctx} --batch {b}")
                for b in list(map(int, sys.argv[1:])) or [16] for model, docs, ctx in CELLS.values() for gpu in pool.POOL]
    cells = {f"--model {m} --docs {d} --ctx {x} ": c for c, (m, d, x) in CELLS.items()}
    for r, log in pool.ours(LEDGER, todo, "jobs"):
        find = lambda k: next((float(l.split()[1]) for l in log if l.startswith(k + " ")), None)
        r.update(cell=next(c for k, c in cells.items() if k in r["cmd"]), batch=int(r["cmd"].split()[-1]), loss=find("loss"), tok_s=find("tok_s"))
        with open("out/reference.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        print(r["cell"], r["gpu"], r["batch"], r["state"], r.get("seconds"), r.get("peak_gpu_mb"), r["loss"], flush=True)
    data.upload("out", f"reference/{open('COMMIT').read().strip()}", ["reference*.jsonl", "logs/*"])


if __name__ == "__main__":
    main()
