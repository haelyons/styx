"""Before a published training script is run on a GPU it cannot use, what does an agent say the run
will take, need and score, and what time limit would it request? Does being able to compute on this
machine's CPU, or being told to find a proxy, change the answers? How near is its runtime to every
run of ours of the script at its full budget, and how often do those runs fall inside its range?
Are its answers nearer than a forecast that knows nothing of the script, the mean of our other jobs
on that GPU whose grade moves with their epochs? Is its grade nearer than that mean of the same
task's jobs, than the grade of the task's sample submission, and than the grade published for the
script, which remembering it would give? What grade does it say the same script earns when it
trains for fewer epochs, and how often do the grades of our runs fall inside its range? The time
limit it requests is priced: a run killed at its limit is run again in full, and each second of the
limit the run does not use is charged at a share of a second of running. Does the agent request the
limit that costs least? And before it starts, what does it say its own run will take?

inspect eval tasks/forecast.py --model none -T agent=claude [-T model=...] --max-samples 1 --epochs 3 --log-dir ~/.cache/styx/forecast/$(cat COMMIT)/logs
"""

import asyncio
import json
import math
import os
import statistics

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import solver

from styx import agents, data, items, pool

HOLD = 0.3
PROMPT = """This directory holds `solution.py`, a training script for the task in `description.md`, and the task's data in `input/`.

The script will be run once as `python solution.py`, unchanged but for its fixed random seeds, which are set to a number you are not told, on a machine with one NVIDIA {gpu} ({gb} GB), 30 vCPUs and 200 GB of memory, with Python 3.12, torch 2.11.0 (CUDA 12.8) and torchvision 0.26.0. Its submission will then be graded as the task describes. Forecast that run.

{means}

End your reply with these lines, each a single number:
RUNTIME_S=<wall-clock seconds of the run, your median>
RUNTIME_S_LOW=<10th percentile>
RUNTIME_S_HIGH=<90th percentile>
PEAK_GPU_MB=<peak GPU memory in use on the device, as nvidia-smi reports it, in MB>
TEST_SCORE=<the grade, its {metric}, your median>
TEST_SCORE_LOW=<10th percentile>
TEST_SCORE_HIGH=<90th percentile>{budgets}
REQUEST_S=<the time limit you would request for the run, in seconds; the run is killed at its limit and must then be run again in full, and each second of the limit the run does not use is charged at {hold} of a second of running>"""
MINUTES = 30
LOCAL = (f"This machine has no GPU. You can run code here on 4 CPU threads and 8 GB of memory; `python` has torch 2.11.0, "
         f"torchvision 0.26.0, scikit-learn, pandas, pillow and tqdm. You have {MINUTES} minutes.")
# condition: what the agent is told, and what agents.run gives it
MEANS = dict(
    direct=("You cannot run code here. You can read the files.", "none"),
    local=(LOCAL, "cpu"),
    proxy=(LOCAL + " Find and use a proxy you can use locally.", "cpu"),
)
KEYS = ("RUNTIME_S", "RUNTIME_S_LOW", "RUNTIME_S_HIGH", "PEAK_GPU_MB", "REQUEST_S")
QUANTILES = (0.1, 0.5, 0.9)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = open(f"{REPO}/COMMIT").read().strip()
ANSWERS = ("PredictBeforeExecute", "styx-data", "items.jsonl", "denoise.jsonl", REPO, *(v.split("_run_")[1][:12] for _, v in items.JOBS.values()))


def measured():
    """Per job and GPU, the median seconds and peak GPU memory of every run of ours of the script at its
    full budget (items.truth's `full`) and the seconds of each, the test score of every seed at the full
    budget and at each smaller budget of items.BUDGETS that was swept, and the median of the full budget's."""
    ours = {}
    for (j, g), t in items.truth(items.lines(f"{REPO}/out/items/items.jsonl"), pool.POOL).items():
        runs, score = t["full"], [items.got(r) for r in t["truth"]]
        ours[j, g] = dict({q: statistics.median(r[q] for r in runs) for q in ("seconds", "peak_gpu_mb")}, test_score=statistics.median(score), n=len(runs),
                          runs=[r["seconds"] for r in runs], score=score, budgets={e: [items.got(r) for r in s] for e, s in t["seeds"].items() if e != items.BUDGETS[j][-1][0]})
    return ours


def constant(ours):
    """Per job and GPU, the forecasts that know nothing of the script: the geometric mean of our other
    jobs on that GPU, none of items.FLAT, for the test score those of the same task, whose metric it is
    (nan if there are none); `sample` the grade of the task's sample submission; `published` the test score
    published for the script."""
    mean = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else math.nan
    other = lambda j, g, q: [r[q] for (k, h), r in ours.items() if h == g and k != j and k not in items.FLAT and (q != "test_score" or items.JOBS[k][0] == items.JOBS[j][0])]
    return {(j, g): dict({q: mean(other(j, g, q)) for q in ("seconds", "peak_gpu_mb", "test_score")}, sample=items.sample(items.JOBS[j][0]), published=items.PUBLISHED[j][1]) for j, g in ours}


off = lambda a, b: abs(math.log(a / b)) if a > 0 else math.nan
ln = lambda x: math.log(x) if x > 0 else math.nan


def pinball(ys, low, mid, high):
    """The mean over our runs' values `ys`, and over the quantiles 0.1, 0.5 and 0.9, of the pinball loss of
    ln(run's value) against ln of the forecast's three quantiles."""
    loss = lambda q, d: q * d if d >= 0 else (q - 1) * d
    return sum(loss(q, ln(y) - ln(p)) for y in ys for q, p in zip(QUANTILES, (low, mid, high))) / (len(ys) * len(QUANTILES))


def inside(ys, low, high):
    """The share of our runs' values `ys` in the forecast's 10th to 90th range; nan if it gave none."""
    return math.nan if math.isnan(low + high) else sum(low <= y <= high for y in ys) / len(ys)


def grade(said, seeds, base, e=None):
    """What the forecast of the test score at budget `e` (None the full run) says against the scores of
    our seeds: error of the median, share of seeds in its 10th to 90th range, pinball loss, and the
    same error and loss of each constant of `base` for the test score (`base`, `sample`, `published`),
    the constant as all three quantiles. Keys end in _e<epochs> but for the full run's."""
    s = "" if e is None else f"_E{e}"
    low, mid, high = (said[f"TEST_SCORE{s}{q}"] for q in ("_LOW", "", "_HIGH"))
    value = dict(score_err=off(mid, statistics.median(seeds)), score_in=inside(seeds, low, high), score_pin=pinball(seeds, low, mid, high))
    for name, c in dict(base=base["test_score"], sample=base["sample"], published=base["published"]).items():
        value |= {f"score_{name}": off(c, statistics.median(seeds)), f"score_pin_{name}": pinball(seeds, c, c, c)}
    return {f"{k}{s.lower()}": v for k, v in value.items()}


def request_cost(request, runs):
    """The mean over our runs of seconds t of what a run costs per second it takes, at a limit of
    `request` seconds: (t + HOLD * (request - t)) / t within the limit, (request + t) / t past it, where
    the run is killed and run again."""
    return sum((t + HOLD * (request - t)) / t if request >= t else (request + t) / t for t in runs) / len(runs)


@solver
def forecasts(agent, model):
    async def solve(state, generate):
        ep = agents.episode("forecast", RUN, agent, model, state.sample_id, state.epoch)
        d = state.metadata["item"]
        own = await agents.before(agent, model, state.input_text, ep)
        r = await agents.run(agent, model, state.input_text, [f"{d}/solution.py", f"{d}/description.md", f"{d}/input"], ep, MINUTES, state.metadata["tools"])
        state.output = ModelOutput.from_content(model=f"{agent}/{model or 'default'}", content=r.pop("answer"))
        state.metadata.update(episode=ep, run=r, own=own)
        state.messages = agents.messages(ep)
        return state

    return solve


@scorer(metrics={"*": [mean()]})
def error():
    ours = measured()
    blind = constant(ours)

    async def score(state, target):
        ep, run, real = state.metadata["episode"], state.metadata["run"], ours[state.metadata["job"], state.metadata["gpu"]]
        base = blind[state.metadata["job"], state.metadata["gpu"]]
        said, own = agents.numbers(state.output.completion, state.metadata["keys"]), state.metadata["own"]
        value = dict(
            answered=not any(map(math.isnan, said.values())),
            runtime_err=off(said["RUNTIME_S"], real["seconds"]),
            runtime_in=inside(real["runs"], said["RUNTIME_S_LOW"], said["RUNTIME_S_HIGH"]),
            runtime_pin=pinball(real["runs"], said["RUNTIME_S_LOW"], said["RUNTIME_S"], said["RUNTIME_S_HIGH"]),
            runtime_pin_base=pinball(real["runs"], base["seconds"], base["seconds"], base["seconds"]),
            memory_err=off(said["PEAK_GPU_MB"], real["peak_gpu_mb"]),
            **grade(said, real["score"], base),
            **{k: v for e, seeds in real["budgets"].items() for k, v in grade(said, seeds, base, e).items()},
            score_recalled=off(said["TEST_SCORE"], base["published"]),
            runtime_base=off(base["seconds"], real["seconds"]), memory_base=off(base["peak_gpu_mb"], real["peak_gpu_mb"]),
            request_ok=said["REQUEST_S"] >= real["seconds"],
            request_ratio=said["REQUEST_S"] / real["seconds"], request_cost=request_cost(said["REQUEST_S"], real["runs"]),
            agent_s=run["agent_s"], cpu_s=run["cpu_s"], late=run["late"], peeked=agents.names(ep, ANSWERS),
            **agents.own(own, run),
        )
        row = {**run, **value, "item": state.sample_id, "epoch": state.epoch, "budget_kind": items.BUDGET_KIND[state.metadata["job"]], "said": said, "own": own, "ours": {q: real[q] for q in ("seconds", "peak_gpu_mb", "test_score", "n")},
               "seeds": dict(seconds=real["runs"], score=real["score"], **{f"score_e{e}": v for e, v in real["budgets"].items()})}
        with open(f"{ep}/episode.json", "w") as f:
            json.dump(row, f, indent=1)
        await asyncio.to_thread(data.push, os.path.dirname(os.path.dirname(ep)))
        return Score(value=value, answer=json.dumps(said))

    return score


@task
def forecast(agent="claude", model=None):
    ours = measured()
    dirs = {j: items.item(j) for j in {j for j, _ in ours}}
    budget = lambda j, g: [(e, phrase) for e, phrase, _ in items.BUDGETS[j] if e in ours[j, g]["budgets"]]
    grades = lambda s: tuple(f"TEST_SCORE{s}{q}" for q in ("", "_LOW", "_HIGH"))
    metric = lambda j: items.TASKS[items.JOBS[j][0]]["metric"]
    lines = lambda j, g: "".join(f"\nTEST_SCORE_E{e}=<the grade, its {metric(j)}, if {phrase}, your median>\nTEST_SCORE_E{e}_LOW=<10th percentile>\nTEST_SCORE_E{e}_HIGH=<90th percentile>" for e, phrase in budget(j, g))
    samples = [
        Sample(id=f"{c}-{g}-{j}", input=PROMPT.format(gpu=g, gb=pool.POOL[g]["gb"], means=text, metric=metric(j), budgets=lines(j, g), hold=HOLD),
               metadata=dict(item=dirs[j], job=j, gpu=g, tools=tools, keys=KEYS + grades("") + sum((grades(f"_E{e}") for e, _ in budget(j, g)), ())))
        for j, g in ours for c, (text, tools) in MEANS.items()
    ]
    return Task(dataset=samples, solver=forecasts(agent, model), scorer=error())
