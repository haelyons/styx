"""Before a published training script is run on a GPU it cannot use, what does an agent say the run
will take, need and score, and what time limit would it request? Does being able to compute on this
machine's CPU, or being told to find a proxy, change the answers?

inspect eval tasks/forecast.py --model none -T agent=claude [-T model=...] --max-samples 1 --log-dir ~/.cache/styx/forecast/$(cat COMMIT)/logs
"""

import asyncio
import json
import math
import os
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import solver

from styx import agents, data, items, pool

PROMPT = """This directory holds `solution.py`, a training script for the task in `description.md`, and the task's data in `input/`.

The script will be run once, unchanged, as `python solution.py`, on a machine with one NVIDIA {gpu} ({gb} GB), 30 vCPUs and 200 GB of memory, with Python 3.12, torch 2.11.0 (CUDA 12.8) and torchvision 0.26.0. Its submission will then be graded as the task describes. Forecast that run.

{means}

End your reply with these lines, each a single number:
RUNTIME_S=<wall-clock seconds of the run, your median>
RUNTIME_S_LOW=<10th percentile>
RUNTIME_S_HIGH=<90th percentile>
PEAK_GPU_MB=<peak GPU memory used, in MB>
TEST_RMSE=<the grade, your median>
TEST_RMSE_LOW=<10th percentile>
TEST_RMSE_HIGH=<90th percentile>
REQUEST_S=<the time limit you would request for the run; it is killed at the limit>"""
MINUTES = 30
LOCAL = (f"This machine has no GPU. You can run code here on 4 CPU threads and 8 GB of memory; `python` has torch 2.11.0, "
         f"torchvision 0.26.0, scikit-learn, pandas and pillow. You have {MINUTES} minutes.")
# condition: what the agent is told, and what agents.run gives it
MEANS = dict(
    direct=("You cannot run code here. You can read the files.", "none"),
    local=(LOCAL, "cpu"),
    proxy=(LOCAL + " Find and use a proxy you can use locally.", "cpu"),
)
KEYS = ("RUNTIME_S", "RUNTIME_S_LOW", "RUNTIME_S_HIGH", "PEAK_GPU_MB", "TEST_RMSE", "TEST_RMSE_LOW", "TEST_RMSE_HIGH", "REQUEST_S")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = open(f"{REPO}/COMMIT").read().strip()
ANSWERS = ("PredictBeforeExecute", "styx-data", "denoise.jsonl", "c4121e82", REPO)
KEEP = ("transcript.jsonl", "stderr.txt", "answer.txt", "episode.json", "work/*.py", "work/*.sh", "work/*.txt", "work/*.json", "work/*.log")


def measured():
    """Per GPU, the mean of our own runs of the script: seconds, peak GPU memory, test RMSE."""
    rows = [r for r in map(json.loads, open(f"{REPO}/out/denoise/denoise.jsonl")) if r.get("state") == "SUCCEEDED"]
    by = {g: [r for r in rows if r["gpu"] == g] for g in pool.POOL}
    return {g: dict({k: sum(r[k] for r in rs) / len(rs) for k in ("seconds", "peak_gpu_mb", "test_rmse")}, n=len(rs)) for g, rs in by.items()}


@solver
def forecasts(agent, model):
    async def solve(state, generate):
        ep = agents.episode("forecast", RUN, agent, model, state.sample_id, state.epoch)
        d = state.metadata["item"]
        r = await agents.run(agent, model, state.input_text, [f"{d}/solution.py", f"{d}/description.md", f"{d}/input"], ep, MINUTES, state.metadata["tools"])
        state.output = ModelOutput.from_content(model=f"{agent}/{model or 'default'}", content=r.pop("answer"))
        state.metadata.update(episode=ep, run=r)
        return state

    return solve


@scorer(metrics={"*": [mean()]})
def error():
    ours = measured()

    async def score(state, target):
        ep, run, real = state.metadata["episode"], state.metadata["run"], ours[state.metadata["gpu"]]
        text = state.output.completion
        said = {k: float(m[-1]) if (m := re.findall(rf"^{k}=\s*([0-9.eE+-]+)\s*$", text, re.M)) else math.nan for k in KEYS}
        off = lambda a, b: abs(math.log(a / b)) if a > 0 else math.nan
        value = dict(
            answered=not any(map(math.isnan, said.values())),
            runtime_err=off(said["RUNTIME_S"], real["seconds"]),
            runtime_in=said["RUNTIME_S_LOW"] <= real["seconds"] <= said["RUNTIME_S_HIGH"],
            memory_err=off(said["PEAK_GPU_MB"], real["peak_gpu_mb"]),
            rmse_err=abs(said["TEST_RMSE"] - real["test_rmse"]),
            rmse_in=said["TEST_RMSE_LOW"] <= real["test_rmse"] <= said["TEST_RMSE_HIGH"],
            request_ok=said["REQUEST_S"] >= real["seconds"],
            request_ratio=said["REQUEST_S"] / real["seconds"],
            agent_s=run["agent_s"], cpu_s=run["cpu_s"], late=run["late"], peeked=agents.names(ep, ANSWERS),
        )
        row = {**run, **value, "item": state.sample_id, "epoch": state.epoch, "said": said, "ours": real}
        with open(f"{ep}/episode.json", "w") as f:
            json.dump(row, f, indent=1)
        await asyncio.to_thread(data.upload, ep, os.path.relpath(ep, agents.ROOT), KEEP)
        return Score(value=value, answer=json.dumps(said))

    return score


@task
def forecast(agent="claude", model=None):
    d = items.denoise()
    samples = [
        Sample(id=f"{c}-{g}", input=PROMPT.format(gpu=g, gb=v["gb"], means=text), metadata=dict(item=d, gpu=g, tools=tools))
        for c, (text, tools) in MEANS.items() for g, v in pool.POOL.items()
    ]
    return Task(dataset=samples, solver=forecasts(agent, model), scorer=error())
