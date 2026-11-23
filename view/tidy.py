"""The rows the view draws, as flat columns: every scored episode (styx.data.table), every run of ours of the jobs
(scripts.s_items, read while it may be mid-append: a broken last line is skipped), and the eval logs with their samples.
"""

import glob
import json
import math

import pandas as pd

from scripts import s_items
from styx import data, items


def episodes():
    """A key's dots become underscores, lower case. A forecast's test score is its TEST_SCORE, or its TEST_RMSE from before there were many tasks;
    `harness` and `model` split the agent; condition, gpu and job come from the episode id (`<condition>-<gpu>[-<job>]-<epoch>`, no job being
    job a; a post episode's job is its item and it has no condition or given GPU)."""
    df = pd.DataFrame(items.lines(f"{data.ROOT}/table/episodes.jsonl"))
    df.columns = [c.replace(".", "_").lower() for c in df.columns]
    col = lambda c: df[c] if c in df else pd.Series(math.nan, index=df.index)
    for new, old in [(f"said_test_score{q}", f"said_test_rmse{q}") for q in ("", "_low", "_high")] + [("ours_test_score", "ours_test_rmse")]:
        df[new] = col(new).fillna(col(old))
    for c in ("answered", "correct"):
        df[c] = col(c).astype("boolean")
    df["harness"], df["model"] = zip(*[(a.partition("-")[0], a.partition("-")[2] or "(not recorded)") for a in df["agent"]])
    cells = [e.rsplit("-", 1)[0].split("-") if t == "forecast" else ["n/a", "n/a", e.rsplit("-", 1)[0]] for t, e in zip(df["task"], df["episode"])]
    df["condition"], df["gpu"], df["job"] = zip(*[(c[0], c[1], c[2] if len(c) > 2 else "a") for c in cells])
    df["runtime_ratio"] = col("runtime_ln").map(math.exp)
    df["margin"] = col("said_request_s") / col("said_runtime_s")
    df["usd_ratio"] = col("usd") / col("usd_ours")
    return df


def ours():
    """`score` is items.got's; `budget` the epochs a GPU run trained: a sweep row's, the job's last budget for the script as published
    (none for a job without BUDGETS or a CPU run); `truth` marks the seeds a forecast is scored against: per job and GPU the sweep's at the
    full budget, else the runs as published (items.truth)."""
    rows = items.lines(f"{s_items.OUT}/items.jsonl")
    df = pd.DataFrame(rows).rename(columns={"job": "pool_job", "script": "job"})
    df["score"] = [items.got(r) if {"test_score", "test_rmse"} & r.keys() else math.nan for r in rows]
    df["gpu"] = df["gpu"].fillna(df["where"])
    df["metric"] = [items.TASKS[items.JOBS[j][0]]["metric"] for j in df["job"]]
    full = {j: b[-1][0] for j, b in items.BUDGETS.items()}
    df["budget"] = [e if v == "sweep" else full.get(j) if v == "full" and w != "cpu" else math.nan for j, v, w, e in zip(df["job"], df["variant"], df["where"], df["epochs"])]
    df["budget_label"] = [str(int(b)) if b == b else v for b, v in zip(df["budget"], df["variant"])]
    df["seed_label"] = ["none" if s != s else str(int(s)) for s in df.get("seed", pd.Series(math.nan, index=df.index))]
    df["line"] = range(1, len(df) + 1)
    df["truth"] = df["uid"].isin({r["uid"] for t in items.truth(rows, set(df["gpu"])).values() for r in t["truth"]})
    return df


def logs(tasks):
    return sorted(p for t in tasks for p in glob.glob(f"{data.ROOT}/{t}/*/logs/*.eval"))


def evals(tasks=("forecast", "post")):
    """One row per eval log. Its model is none/none; the real one is the agent and model args. Scores are floats (stderr reads as text)."""
    from inspect_ai.analysis import evals_df

    df = evals_df(logs(tasks), quiet=True)
    for c in [c for c in df.columns if c.startswith("score_") and c.endswith(("_value", "_stderr", "_mean"))]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["run"] = [p.split("/")[-3] for p in df["log"]]
    df["model"] = [f"{a}-{m}" if isinstance(m, str) and m else str(a) for a, m in zip(df["task_arg_agent"], df["task_arg_model"])]
    return df


def samples(tasks=("forecast", "post")):
    """One row per sample, with its eval's model; each task's one scorer gives a dict (forecast score_error, post score_cost) whose numbers become s_ columns."""
    from inspect_ai.analysis import samples_df

    df = samples_df(logs(tasks), quiet=True).merge(evals(tasks)[["eval_id", "task_name", "run", "model"]], on="eval_id", how="left")
    as_dict = lambda v: (json.loads(v) if isinstance(v, str) else v) if isinstance(v, (str, dict)) else {}
    cols = [c for c in df.columns if c.startswith("score_")]
    scores = pd.DataFrame([{k: v for c in cols for k, v in as_dict(r[c]).items()} for _, r in df[cols].iterrows()], index=df.index)
    for c in scores.columns:
        if scores[c].dropna().map(lambda v: isinstance(v, (bool, int, float))).all():
            df[f"s_{c}"] = pd.to_numeric(scores[c].map(lambda v: math.nan if v is None else float(v)))
    return df
