"""Results go to the HF dataset as <task>/<run>/. No value of .env leaves this machine: wherever
one occurs in a file, the copy that is uploaded has its name instead.
"""

import asyncio
import fnmatch
import glob
import json
import math
import os
import shutil
import tempfile
import time

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

DATASET = "hal2k/styx-data"
ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
MAX_BYTES = 50e6
ROOT = os.path.expanduser("~/.cache/styx")
# What of an episode is published: what it said and did, and the text files it wrote.
KEEP = ("episode.json", "ledger.jsonl", "prompt.txt", "transcript.jsonl", "clock.txt", "exec.jsonl", "stderr.txt", "answer.txt", "before/prompt.txt", "before/transcript.jsonl",
        "before/clock.txt", "before/exec.jsonl", "before/answer.txt", "work/*.py", "work/*.sh", "work/*.txt", "work/*.json", "work/*.log", "work/*.md")


def env():
    rows = (l.strip().split("=", 1) for l in open(ENV) if "=" in l and not l.startswith("#"))
    return {k: v.strip("\"'") for k, v in rows if v.strip("\"'")}


def stage(src, tmp, keep):
    """Copies the files of `src` whose relative path matches one of `keep` into `tmp`, each value of
    .env replaced by its name. Hidden directories, __pycache__, an episode's work/input/ (the task's data) and files over 50 MB stay behind."""
    names = {v.encode(): f"<{k}>".encode() for k, v in env().items()}
    for root, _, files in os.walk(src):
        for rel in (os.path.relpath(f"{root}/{f}", src) for f in files):
            hidden = "/." in f"/{os.path.dirname(rel)}" or "__pycache__" in rel
            if hidden or rel.startswith("work/input/") or os.path.getsize(f"{src}/{rel}") > MAX_BYTES or not any(fnmatch.fnmatch(rel, k) for k in keep):
                continue
            b = open(f"{src}/{rel}", "rb").read()
            for value, name in names.items():
                b = b.replace(value, name)
            os.makedirs(os.path.dirname(f"{tmp}/{rel}"), exist_ok=True)
            with open(f"{tmp}/{rel}", "wb") as f:
                f.write(b)


def upload(src, dst, keep, gone=None):
    """Sends the files of directory `src` matching `keep` to <dataset>/<dst>, in one commit. What is
    there under `gone` and not among them is deleted."""
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(src))) as tmp:  # /tmp has a quota here
        stage(src, tmp, keep)
        HfApi(token=env()["HF_W_KEY_ONE"]).upload_folder(folder_path=tmp, path_in_repo=dst, repo_type="dataset", repo_id=DATASET, delete_patterns=gone)


def push(run, every=900):
    """Sends every scored episode of the run directory <root>/<task>/<run> in one commit, and not
    again from this machine within `every` seconds: the dataset takes 128 commits an hour. What
    just missed goes with a later episode, or with `python -m styx.data`."""
    mark = f"{ROOT}/pushed"
    if os.path.exists(mark) and time.time() - os.path.getmtime(mark) < every:
        return
    open(mark, "w").close()
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        for scored in glob.glob(f"{run}/*/*/episode.json"):
            ep = os.path.dirname(scored)
            stage(ep, f"{tmp}/{os.path.relpath(ep, run)}", KEEP)
        try:
            HfApi(token=env()["HF_W_KEY_ONE"]).upload_folder(folder_path=tmp, path_in_repo=os.path.relpath(run, ROOT), repo_type="dataset", repo_id=DATASET)
        except HfHubHTTPError as e:
            print(f"not pushed: {str(e)[:200]}", flush=True)


def table():
    """Writes episodes.jsonl, one row per scored episode of every run: where it is, its own numbers,
    its first job, how its seconds went, and for a forecast the signed ln(said / ours) of its runtime, memory and test score."""
    from styx import agents

    os.makedirs(f"{ROOT}/table", exist_ok=True)
    with open(f"{ROOT}/table/episodes.jsonl", "w") as out:
        for scored in sorted(glob.glob(f"{ROOT}/*/*-*/*/*/episode.json")):
            ep = os.path.dirname(scored)
            task, run, agent, sample = ep.split("/")[-4:]
            row, flat = json.load(open(scored)), dict(task=task, run=run, agent=agent, episode=sample)
            for k, v in row.items():
                if isinstance(v, dict):
                    flat.update({f"{k}.{q}": x for q, x in v.items() if not isinstance(x, (dict, list))})
                elif k == "posted":
                    flat.update({f"first.{q}": v[0].get(q) for q in ("gpu", "limit_s", "state", "seconds", "cmd") if v})
                else:
                    flat[k] = v
            for k, a, b in (("runtime_ln", "said.RUNTIME_S", "ours.seconds"), ("memory_ln", "said.PEAK_GPU_MB", "ours.peak_gpu_mb"), ("score_ln", "said.TEST_SCORE", "ours.test_score")):
                if "said" in row:
                    x, y = (flat.get(n, flat.get(n.replace("TEST_SCORE", "TEST_RMSE").replace("test_score", "test_rmse"))) for n in (a, b))  # episodes from before there were many tasks have the RMSE
                    flat[k] = math.log(x / y) if (x or 0) > 0 and (y or 0) > 0 else None
            out.write(json.dumps({**flat, **(agents.spent(ep) if os.path.exists(f"{ep}/clock.txt") else {})}) + "\n")  # the first runs kept no clock


def transcripts():
    """Writes transcripts/, an Inspect Scout database: every episode of episodes.jsonl that left a
    transcript, and the session that forecast it, as agents.messages gives them, with the episode's
    row as columns (numbers as floats, a dot in a name as an underscore, anything nested as JSON)."""
    from inspect_scout import Transcript, transcripts_db

    from styx import agents

    async def every(tmp):
        for row in map(json.loads, open(f"{ROOT}/table/episodes.jsonl")):
            task, run, agent, sample = (row.pop(k) for k in ("task", "run", "agent", "episode"))
            cli, _, model = agent.partition("-")
            cell, _, epoch = sample.rpartition("-")
            columns = {k.replace(".", "_"): v if isinstance(v, (str, bool, type(None))) else float(v) if isinstance(v, (int, float)) else json.dumps(v) for k, v in row.items()}
            stage(f"{ROOT}/{task}/{run}/{agent}/{sample}", f"{tmp}/ep", ("prompt.txt", "transcript.jsonl", "clock.txt", "exec.jsonl", "before/prompt.txt", "before/transcript.jsonl", "before/clock.txt", "before/exec.jsonl"))
            for part, name in ((f"{tmp}/ep", task), (f"{tmp}/ep/before", f"{task}-before")):
                if os.path.exists(f"{part}/transcript.jsonl") and os.path.getsize(f"{part}/transcript.jsonl"):
                    whole = name == task
                    yield Transcript(
                        transcript_id=f"{name}__{run}__{agent}__{sample}", source_type="styx", source_id=run, task_set=name, task_id=cell, task_repeat=int(epoch),
                        agent=cli, model=model or None, score=row.get("correct") if whole else None, success=row.get("correct") if whole else None,
                        total_time=row.get("agent_s") if whole else None, metadata=columns, messages=agents.messages(part))
            shutil.rmtree(f"{tmp}/ep", ignore_errors=True)

    async def write():
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            async with transcripts_db(f"{ROOT}/table/transcripts") as db:
                await db.insert(every(tmp))

    shutil.rmtree(f"{ROOT}/table/transcripts", ignore_errors=True)
    asyncio.run(write())


if __name__ == "__main__":
    for run in glob.glob(f"{ROOT}/post/*-*") + glob.glob(f"{ROOT}/forecast/*-*"):
        print(run, flush=True)
        push(run, 0)
    table()
    transcripts()
    upload(f"{ROOT}/table", "", ("episodes.jsonl", "transcripts/*"), "transcripts/*")  # each build of the database names its files anew
