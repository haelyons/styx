"""The rows the view draws, as flat columns: every scored episode (styx.data.table), each forecast episode at each budget it
was asked, every run of ours of the jobs (scripts.s_items, read while it may be mid-append: a broken last line is skipped), and
the eval logs with their samples; and what the repo says each column is, which the view prints under what draws it.
"""

import ast
import glob
import inspect
import json
import math
import os
import re
import statistics
import textwrap

import pandas as pd

from scripts import s_items
from styx import agents, data, items
from tasks import forecast, post

# Each column the view derives, as DataFrame.eval computes it from the columns of a row; a value that is not finite is left out.
DERIVED = dict(
    runtime_ratio="said_runtime_s / ours_seconds",
    request_ratio="said_request_s / ours_seconds",
    margin="said_request_s / said_runtime_s",
    score_ratio="said_test_score / ours_test_score",
    usd_ratio="usd / usd_ours",
    first_limit_ratio="first_limit_s / first_seconds",
    self_seconds_ratio="own_self_seconds / agent_s",
    self_tokens_ratio="own_self_output_tokens / output_tokens",
    self_rate_ratio="(own_self_output_tokens / own_self_seconds) / (output_tokens / agent_s)",
    seconds_gain_ours="log(ours_median_s / ours_median_s_full)",
    seconds_gain_said="log(said_runtime_s_e / said_runtime_s_e_full)",
    score_gain_ours="log(ours_median_score / ours_median_score_full)",
    score_gain_said="log(said_test_score_e / said_test_score_e_full)",
)
# Each number a plot computes in the browser from the rows of one group of a mark, as DuckDB reads it.
AGGREGATE = dict(
    runtime_miss="exp(avg(abs(ln(runtime_ratio))))",
    score_miss="exp(avg(abs(ln(score_ratio))))",
    margin_mean="exp(avg(ln(margin)))",
    seconds_slope="regr_slope(seconds_gain_said, seconds_gain_ours)",
    seconds_points="regr_count(seconds_gain_said, seconds_gain_ours)",
    score_slope="regr_slope(score_gain_said, score_gain_ours)",
    score_points="regr_count(score_gain_said, score_gain_ours)",
    episodes="count(*)",
)
# Models weakest to strongest within a CLI as far as their names say it: Claude's as its family goes, an OpenCode -flash before its
# -pro sibling. The others keep this list's order, which their names do not give; a model not in it comes after the rest of its CLI.
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2", "(not recorded)", "gpt-6-luna", "gpt-6.2-sol",
          "opencode-go_deepseek-v4.1-flash", "opencode-go_deepseek-v4-pro", "opencode_deepseek-v4.1-flash-free", "opencode-go_glm-5.3-flash",
          "opencode-go_minimax-m3.1-flash", "opencode-go_qwen3.8-flash", "opencode_hy4-flash-free", "opencode-go_kimi-k3", "opencode_big-pickle",
          "opencode_nemotron-3.5-lightning-free", "opencode_nemotron-3.5-super-vl-free", "h100_gpt-oss-120b", "h100x2_gpt-oss-120b"]


def asked(row):
    """The keys a prompt asked, from those of its episode's said: a forecast's RUNTIME_S and TEST_SCORE (TEST_RMSE before there were
    many tasks), and each at a smaller budget as _E*; a post prompt asks LOSS."""
    if row["task"] == "post":
        return " ".join(prompts(post.PROMPT))
    return " ".join(sorted({re.sub(r"_E\d+", "_E*", k[5:]) for k in row if re.fullmatch(r"said\.(RUNTIME_S|TEST_SCORE|TEST_RMSE)(_E\d+)?", k)}))


def episodes():
    """A key's dots become underscores, lower case. A forecast's test score is its TEST_SCORE, or its TEST_RMSE from before there were many tasks;
    `harness` and `model` split the agent; condition, gpu and job come from the episode id (`<condition>-<gpu>[-<job>]-<epoch>`, no job being
    job a; a post episode's job is its item and it has no condition or given GPU); `epochs` lists the smaller budgets its prompt asked."""
    rows = items.lines(f"{data.ROOT}/table/episodes.jsonl")
    df = pd.DataFrame(rows)
    df.columns = [c.replace(".", "_").lower() for c in df.columns]
    col = lambda c: df[c] if c in df else pd.Series(math.nan, index=df.index)
    for new, old in [(f"said_test_score{q}", f"said_test_rmse{q}") for q in ("", "_low", "_high")] + [("ours_test_score", "ours_test_rmse")]:
        df[new] = col(new).fillna(col(old))
    for c in ("answered", "correct"):
        df[c] = col(c).astype("boolean")
    df["runtime_in"] = [math.nan if isinstance(v, bool) or v is None else v for v in col("runtime_in")]  # whether our median fell inside, before it was a share of our runs
    df["harness"], df["model"] = zip(*[(a.partition("-")[0], a.partition("-")[2] or "(not recorded)") for a in df["agent"]])
    cells = [e.rsplit("-", 1)[0].split("-") if t == "forecast" else ["n/a", "n/a", e.rsplit("-", 1)[0]] for t, e in zip(df["task"], df["episode"])]
    df["condition"], df["gpu"], df["job"] = zip(*[(c[0], c[1], c[2] if len(c) > 2 else "a") for c in cells])
    df["asked"] = [asked(r) for r in rows]
    df["epochs"] = [sorted({int(e) for k in r for e in re.findall(r"^said\.\w+_E(\d+)$", k)}) for r in rows]
    return derive(df)


def derive(df):
    """The rows with each column of DERIVED whose expression reads only columns they have."""
    can = {n: e for n, e in DERIVED.items() if set(re.findall(r"[a-z_][a-z0-9_]*", e)) - {"log"} <= set(df.columns)}
    return df.assign(**{n: df.eval(e).replace([math.inf, -math.inf], math.nan) for n, e in can.items()})


def budgets(eps):
    """One row per forecast episode and budget its prompt asked, the full one included, of the episodes whose scorer stored the
    `seeds` of our runs it scored against in its episode.json (styx.data.table leaves lists out): what the episode said the run
    takes and scores at that many epochs (its key at that budget, _E<epochs> but at the full one), and the medians of those runs'
    seconds and seeds' test scores at that budget, each list kept as runs and scores; a column ending in _full holds its value
    at the job's full budget."""
    rows, said = [], [(q, x) for q in ("runtime_s", "test_score") for x in ("", "_low", "_high")]
    for r in eps[eps["task"] == "forecast"].to_dict("records"):
        seeds, last = json.load(open(f"{data.ROOT}/{r['task']}/{r['run']}/{r['agent']}/{r['episode']}/episode.json")).get("seeds"), items.BUDGETS[r["job"]][-1][0]
        if not seeds:
            continue
        at = lambda e: {f"said_{q}_e{x}": r.get(f"said_{q}{'' if e == last else f'_e{e}'}{x}", math.nan) for q, x in said}
        got = {e: dict(runs=seeds.get("seconds" if e == last else f"seconds_e{e}", []), scores=seeds.get("score" if e == last else f"score_e{e}", []), **at(e)) for e in r["epochs"] + [last]}
        for g in got.values():
            g.update(ours_median_s=statistics.median(g["runs"]) if g["runs"] else math.nan, ours_median_score=statistics.median(g["scores"]) if g["scores"] else math.nan)
        for e, g in got.items():
            rows.append({**{k: r[k] for k in ("task", "run", "harness", "model", "agent", "episode", "condition", "gpu", "job", "asked", "answered")}, "epochs": e, "full": e == last,
                         "kind": "flat" if r["job"] in items.FLAT else items.BUDGET_KIND[r["job"]], **g, **{f"{k}_full": got[last][k] for k in ("said_runtime_s_e", "said_test_score_e", "ours_median_s", "ours_median_score")}})
    return derive(pd.DataFrame(rows))


def seeds(b):
    """Each value of ours behind the medians of budgets once: per job, GPU and epochs, the seconds of a run or the test score of a seed."""
    each = [b[["job", "gpu", "epochs", lists]].explode(lists).dropna().rename(columns={lists: q}).astype({q: float}) for lists, q in (("runs", "seconds"), ("scores", "test_score"))]
    return pd.concat(each).drop_duplicates().reset_index(drop=True)


def ordered(df):
    """The models of the rows, by CLI and within it by MODELS."""
    pairs = df[["harness", "model"]].drop_duplicates()
    return sorted(pairs["model"], key=lambda m: (pairs.set_index("model")["harness"][m], MODELS.index(m) if m in MODELS else len(MODELS), m))


def prompts(*texts):
    """Each KEY=<...> line of the prompts, by its key; a smaller budget's epochs stay {e}."""
    return {k: f"{k}=<{v}>" for t in texts for k, v in re.findall(r"([A-Z][A-Z0-9_]*(?:\{e\}[A-Z_]*)?)=<([^>]*)>", t)}


# Where the repo computes each column the view draws that no prompt asks.
SOURCE = dict(answered=forecast.error, runtime_in=forecast.timed, cpu_s=agents.burnt, agent_s=agents.run, output_tokens=agents.answer,
              ours_seconds=forecast.measured, ours_test_score=forecast.measured, usd=post.cost, usd_ours=post.reference,
              first_gpu=data.table, first_limit_s=data.table, first_seconds=data.table, asked=asked, epochs=budgets, kind=budgets, ours_median_s=budgets, ours_median_score=budgets,
              harness=episodes, model=episodes, condition=episodes, gpu=episodes, job=episodes)


def source(name, fn):
    """Where fn is, then `name = <expression>` where fn gives name a value once, as a keyword or a dict's key (a bare name
    followed to where fn assigns it, else left out), with the first sentence of the docstring of each function of fn's module it
    calls; without one, fn's docstring: all of it if fn is name, else its clauses that name it, else its first sentence."""
    fn = inspect.unwrap(fn)
    clauses = lambda f: re.split(r";\s|(?<=[.?])\s", " ".join((f.__doc__ or "").split()))
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    found = [k.value for k in ast.walk(tree) if isinstance(k, ast.keyword) and k.arg == name]
    found += [v for d in ast.walk(tree) if isinstance(d, ast.Dict) for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant) and k.value == name]
    if len(found) == 1 and isinstance(found[0], ast.Name):
        found = [a.value for a in ast.walk(tree) if isinstance(a, ast.Assign) and any(getattr(t, "id", None) == found[0].id for t in a.targets)][:1]
    if len(found) == 1:
        calls = [getattr(inspect.getmodule(fn), c.func.id, None) for c in ast.walk(found[0]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
        doc = [f"{name} = {ast.unparse(found[0])}"] + [clauses(c)[0] for c in calls if inspect.isfunction(c) and c.__doc__]
    else:
        doc = [" ".join(fn.__doc__.split()) if fn.__name__ == name else "; ".join([c for c in clauses(fn) if re.search(rf"\b{name}\b", c, re.I)] or clauses(fn)[:1])]
    return " · ".join([f"{os.path.relpath(inspect.getsourcefile(fn), forecast.REPO)} {fn.__name__}"] + doc)


def defined(*names):
    """What the repo says each name is, and each name its expression reads, once each: a derived column's or a browser number's
    expression, a key's line of its prompt, and where the repo computes any other, after the names it defines."""
    lines, sourced, said = {}, set(), prompts(forecast.PROMPT, inspect.getsource(forecast.forecast), agents.BEFORE, post.PROMPT)

    def add(n):
        n = n.removesuffix("_full")
        if any(n in ns for ns in lines.values()):
            return
        if n in DERIVED | AGGREGATE:
            lines[f"{n} = {(DERIVED | AGGREGATE)[n]}"] = [n]
            for m in re.findall(r"[a-z_][a-z0-9_]*", (DERIVED | AGGREGATE)[n]):
                add(m)
        elif n.startswith(("said_", "own_")):
            k = n.split("_", 1)[1].upper()
            lines.setdefault(" · ".join(dict.fromkeys(said[q] for q in (re.sub(r"_E(?=_|$)", "", k), re.sub(r"_E(?=_|$)", "_E{e}", k)) if q in said)), []).append(n)
        elif n in SOURCE:
            sourced.add(text := source(n, SOURCE[n]))
            lines.setdefault(text, []).append(n)

    for n in names:
        add(n)
    return [f"{', '.join(ns)}: {t}" if t in sourced else t for t, ns in lines.items()]


def ours():
    """`score` is items.got's; `budget` the epochs a GPU run trained: a sweep row's, the job's last budget for the script as published
    (none for a job without BUDGETS or a CPU run); `truth` marks the seeds a forecast's test score is scored against: per job and GPU the
    sweep's at the full budget, else the runs as published; `full` the runs its seconds and memory are scored against: every run at the
    job's full budget, as published and in the sweep (items.truth)."""
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
    truth = items.truth(rows, set(df["gpu"])).values()
    df["truth"] = df["uid"].isin({r["uid"] for t in truth for r in t["truth"]})
    df["full"] = df["uid"].isin({r["uid"] for t in truth for r in t["full"]})
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
