"""Handed a workload it cannot run here and a GPU pool, does an agent get the right number, and
what do the jobs it posts cost next to the cheapest posting we know? And before it starts, what
does it say its own run will take?

inspect eval tasks/post.py --model none -T agent=claude [-T model=...] --max-samples 1 --epochs 3 --log-dir ~/.cache/styx/post/$(cat COMMIT)/logs
"""

import asyncio
import json
import os
import re
import sys

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import solver

from styx import agents, data

CELLS = {
    "0.5B-256x512": ("Qwen/Qwen2.5-0.5B", 256, 512),
    "1.5B-4000x1024": ("Qwen/Qwen2.5-1.5B", 4000, 1024),
    "3B-4000x1024": ("Qwen/Qwen2.5-3B", 4000, 1024),
    "7B-3000x1024": ("Qwen/Qwen2.5-7B", 3000, 1024),
    "7B-750x4096": ("Qwen/Qwen2.5-7B", 750, 4096),
}
PROMPT = """This directory holds `loss.py`. Report the mean next-token loss of {model} on the first \
{docs} chunks of {ctx} tokens of the WikiText-103 training text, which is the number that \
`python loss.py --model {model} --docs {docs} --ctx {ctx}` prints.

This machine has no GPU. A GPU pool is reached with the `styx` command; `styx info` describes it.

End your reply with a line of the form `LOSS=<number>`."""
TOLERANCE, MINUTES = 0.001, 90
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = open(f"{REPO}/COMMIT").read().strip()
ANSWERS = ("reference.jsonl", "styx-data", REPO)


def reference():
    """Per cell, the loss and the cheapest posting among ours that succeeded."""
    rows = [r for r in map(json.loads, open(f"{REPO}/out/reference.jsonl")) if r["state"] == "SUCCEEDED"]
    return {c: min((r for r in rows if r["cell"] == c), key=lambda r: r["usd"]) for c in {r["cell"] for r in rows}}


@solver
def posts(agent, model, minutes):
    async def solve(state, generate):
        ep = agents.episode("post", RUN, agent, model, state.sample_id, state.epoch)
        own = await agents.before(agent, model, state.input_text, ep)
        r = await agents.run(agent, model, state.input_text, [f"{REPO}/jobs/loss.py"], ep, minutes)
        state.output = ModelOutput.from_content(model=f"{agent}/{model or 'default'}", content=r.pop("answer"))
        state.metadata.update(episode=ep, run=r, own=own)
        return state

    return solve


@scorer(metrics={"*": [mean()]})
def cost():
    ref = reference()

    async def score(state, target):
        ep, mine = state.metadata["episode"], ref[state.sample_id]
        p = await asyncio.create_subprocess_exec(sys.executable, "-m", "styx.pool", "finalize", f"{ep}/ledger.jsonl", stdout=asyncio.subprocess.PIPE)
        jobs = json.loads((await p.communicate())[0])
        said = re.findall(r"LOSS=([0-9.]+)", state.output.completion)
        calls = [json.loads(l) for l in open(f"{ep}/ledger.jsonl")] if os.path.exists(f"{ep}/ledger.jsonl") else []
        count = lambda s: sum(j["state"] == s for j in jobs)
        value = dict(
            correct=bool(said) and abs(float(said[-1].rstrip(".")) - mine["loss"]) <= TOLERANCE,
            jobs=len(jobs), failed=count("FAILED"), timeout=count("TIMEOUT"), cancelled=count("CANCELLED"),
            left=sum("left" in j for j in jobs),
            usd=sum(j.get("usd", 0) for j in jobs), usd_ours=mine["usd"],
            gpu_s=sum(j.get("seconds", 0) for j in jobs), gpu_s_ours=mine["seconds"],
            agent_s=state.metadata["run"]["agent_s"], late=state.metadata["run"]["late"], calls=len(calls),
            peeked=agents.names(ep, ANSWERS), **agents.own(state.metadata["own"], state.metadata["run"]),
        )
        row = {**state.metadata["run"], **value, "cell": state.sample_id, "epoch": state.epoch, "said": said[-1] if said else None, "ours": mine, "own": state.metadata["own"], "posted": jobs}
        with open(f"{ep}/episode.json", "w") as f:
            json.dump(row, f, indent=1)
        await asyncio.to_thread(data.push, os.path.dirname(os.path.dirname(ep)))
        return Score(value=value, answer=said[-1] if said else None, metadata=dict(jobs=jobs))

    return score


@task
def post(agent="claude", model=None, minutes=MINUTES):
    samples = [Sample(id=c, input=PROMPT.format(model=m, docs=d, ctx=x)) for c, (m, d, x) in CELLS.items()]
    return Task(dataset=samples, solver=posts(agent, model, minutes), scorer=cost())
