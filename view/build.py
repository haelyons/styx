"""Each row as it is: which agents were run, task by prompt, model, condition, GPU and job? What did an agent forecast a job would
take, need and score, at its full budget and at each smaller one, and what limit did it request? What did its first posted job ask
and take, and what did its jobs cost against the reference? What did it forecast its own run would take? And what do the runs we
made of the jobs take and score, seed by seed and budget by budget?

python -m view.build [dir]            writes index.html and questions.html (view.questions) to dir, view/dist by default, from the
                                      episodes table, out/items/items.jsonl and the eval logs as they are now
python -m http.server -d view/dist    serves them on http://localhost:8000

One Inspect Viz HtmlPage per page: one in-browser DuckDB and one selection for every widget, so the bar at the top filters every view
of the agents; the runs we made have filters of their own. The data is inline; the scripts come from cdn.jsdelivr.net.
"""

import html
import os
import sys

import pandas as pd
from inspect_viz import Data, Selection
from inspect_viz.input import checkbox_group, select
from inspect_viz.mark import cell, dot, line, rule_x, rule_y, text, title
from inspect_viz.plot import HtmlPage, legend, plot
from inspect_viz.table import column, table
from inspect_viz.transform import count
from inspect_viz.view import sample_heatmap, scores_heatmap

from styx import data, items
from view import tidy

OUT = "view/dist"
CLI = dict(color_domain=["claude", "codex", "opencode"], color_range=["#2a78d6", "#eb6834", "#1baf7a"])  # fixed order, by CLI
SEEDS = ["#9a9a96", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#2a78d6", "#1baf7a"]
SIDE = 440
WIDE = 620  # a plot titled with its job, as wide as its longest title
WHO = {"model": "model", "id": "episode", "run": "run"}
BESIDE = dict(y_label_anchor="center", margin_left=70)  # a titled plot's y label runs up its axis: at the top it would sit under the title
BAR = ["task", "cli", "model", "condition", "gpu", "job", "run", "asked"]  # what the bar filters on: every table it filters has them
KEY = ["task", "asked", "cli", "model", "condition", "gpu", "job"]  # an agent's cell of what was run
EPISODE = ["task", "run", "cli", "model", "episode", "condition", "gpu", "job", "asked", "answered", "rank",
           "forecast_runtime_s", "forecast_runtime_s_low", "forecast_runtime_s_high", "request_s", "actual_seconds", "runtime_ln", "runtime_ratio", "margin",
           "forecast_test_score", "forecast_test_score_low", "forecast_test_score_high", "actual_test_score", "score_ln",
           "first_gpu", "first_state", "first_limit_s", "first_seconds", "jobs", "failed", "usd", "usd_reference", "usd_ratio", "correct",
           "forecast_self_seconds", "forecast_self_seconds_low", "forecast_self_seconds_high", "agent_s", "forecast_self_output_tokens", "output_tokens",
           "forecast_self_tool_calls", "tool_calls"]
RUNS = ["line", "job", "metric", "gpu", "where", "variant", "seed_label", "budget", "budget_label", "state", "exit", "seconds", "score", "peak_gpu_mb", "pool_job", "truth", "full"]


def episodes(df):
    return Data.from_dataframe(compact(df.reindex(columns=EPISODE)))


def ran(df):
    return Data.from_dataframe(df[RUNS].reset_index(drop=True))


def compact(df):
    """The rows with each column of text as a dictionary of its values, which keeps an inline table small."""
    return df.astype({c: "category" for c in df.columns if pd.api.types.is_string_dtype(df[c])}).reset_index(drop=True)


def fits(name, left, right):
    """The width of a plot whose title `name` sits over its frame, between margins `left` and `right`, without being cut."""
    return left - right + 11 * len(name)


def positive(df, *cols):
    """The rows a log axis can draw."""
    return df[(df[list(cols)] > 0).all(axis=1)].reset_index(drop=True)


def span(*series, pad=1.3):
    return [min(s[s > 0].min() for s in series) / pad, max(s.max() for s in series) * pad]


def ticks(domain):
    """1, 1.5, 2, 3, 5 and 7 times the powers of ten inside a log axis's domain: its own ticks leave narrow ones bare."""
    return [m * 10.0**k for k in range(-4, 6) for m in (1, 1.5, 2, 3, 5, 7) if domain[0] <= m * 10.0**k <= domain[1]]


def named(j):
    """The job's letter and task, what a smaller budget does to it (items.BUDGET_KIND) and whether its score does not move with its budget (items.FLAT)."""
    marks = ([f"smaller budget: {items.BUDGET_KIND[j]}"] if j in items.BUDGET_KIND else []) + (["does not learn"] if j in items.FLAT else [])
    return title(" · ".join([f"job {j}", items.JOBS[j][0], *marks]), margin_top=40)


def diagonal(xdom, ydom, power=1, dash="4 4"):
    """y = x**power across both domains, clipped to the frame."""
    lo, hi = min(xdom[0], ydom[0]), max(xdom[1], ydom[1])
    return line(x=[lo, hi], y=[lo**power, hi**power], clip=True, stroke="#777", stroke_dasharray=dash)


def loglog(*marks, x_label, y_label, xdom, ydom, fmt=None):
    return plot(*marks, x_scale="log", y_scale="log", x_domain=xdom, y_domain=ydom, x_label=x_label, y_label=y_label, x_tick_format=fmt, y_tick_format=fmt,
                grid=True, width=SIDE, height=SIDE, legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **CLI)


def dots(data, filt, x, y, symbol, extra, **options):
    return dot(data, x=x, y=y, fill="cli", symbol=symbol, filter_by=filt, r=4, fill_opacity=0.65, channels=WHO | extra, **options)


def rows(data, filt, cols, width=980, height=SIDE + 80):
    return table(data, filter_by=filt, columns=cols, width=width, height=height, pagination=True)


def num(name, label=None, fmt=None, width=85):
    return column(name, label=label or name, format=fmt, width=width)


WHO_COLS = [num("cli", "CLI", width=75), num("model", width=190), num("episode", "id", width=125), num("run", width=125)]


def told(label, text, open=False):
    """What an agent was told, as the code or a stored prompt holds it, under its file and object name or its run and id, folded."""
    return f'<details class="told"{" open" if open else ""}><summary>{html.escape(label)}</summary><pre>{html.escape(text)}</pre></details>'


def bar(ep, filt):
    """The filters of every view of the agents, on the columns of BAR."""
    return [select(ep, column="task", label="Task", target=filt, filter_by=filt),
            checkbox_group(ep, column="cli", label="CLI", target=filt, filter_by=filt),
            select(ep, column="model", label="Model", target=filt, filter_by=filt, multiple=True),
            checkbox_group(ep, column="condition", label="Condition", target=filt, filter_by=filt),
            checkbox_group(ep, column="gpu", label="GPU", target=filt, filter_by=filt),
            select(ep, column="job", label="Job", target=filt, filter_by=filt, multiple=True),
            select(ep, column="run", label="Run", target=filt, filter_by=filt, multiple=True),
            select(ep, column="asked", label="Prompt asked", target=filt, filter_by=filt, multiple=True)]


def matrix(eps, filt):
    """Every agent in one cell of KEY: a plot of counts per task and prompt, and the same cells as a table."""
    assert eps.groupby(KEY).size().sum() == len(eps), "an agent outside every cell"

    def counts(d):
        task, asked, models, jobs = d["task"].iloc[0], d["asked"].iloc[0], tidy.ordered(d), sorted(d["job"].unique())
        facets, rows_ = (dict(fx="gpu", fy="condition"), d["condition"].nunique()) if task == "forecast" else ({}, 1)
        data = Data.from_dataframe(compact(d[BAR + ["episode"]]))
        return (f"count(*) by model and job{', per gpu and condition' if facets else ''} · task = {task} · asked = {asked}",
                [plot(cell(data, x="job", y="model", fill=count(), filter_by=filt, inset=0.5, **facets),
                      text(data, x="job", y="model", text=count(), filter_by=filt, fill="#0b0b0b", **facets),
                      x_domain=jobs, y_domain=models, x_label=None, y_label=None, fx_label=None, fy_label=None,
                      x_tick_rotate=-30 if task == "post" else 0, margin_left=250, margin_bottom=70 if task == "post" else 30, margin_right=60,
                      width=310 + 30 * len(jobs) * d["gpu"].nunique(), height=18 * len(models) * rows_ + 90, color_scale="log", color_range=["#e3edfa", "#6f9fe0"])], [])

    every = Data.from_dataframe(compact(eps[BAR + ["episode"]].assign(answered=eps["answered"].astype(float))))  # DuckDB sums no booleans
    cols = [num(c, width=w) for c, w in zip(KEY, (75, 300, 75, 250, 80, 65, 120))] + [column({"count": "episode"}, label="agents", width=75), column({"sum": "answered"}, label="answered", width=100)]
    latest = lambda d: d.sort_values(["run", "episode"]).iloc[-1]
    prompt = lambda r: told(f"{r['run']} · {r['episode']} · prompt.txt", open(f"{data.ROOT}/{r['task']}/{r['run']}/{r['agent']}/{r['episode']}/prompt.txt").read())
    return ("M", "What was run: an agent's task, the keys its prompt asked, its CLI and model, condition, GPU and job", "", [],
            [c for _, d in sorted(eps.groupby(["task", "asked"]), key=lambda g: (g[0][0], g[1]["run"].min())) for c in (prompt(latest(d)), counts(d))]
            + [("count(*) and sum(answered) by task, asked, cli, model, condition, gpu and job", [rows(every, filt, cols, width=1220)],
                ["fill, text, agents = count(*)", "answered = sum(answered)", *tidy.defined(*KEY, "answered")])])


def at_budgets(eps, made, filt):
    """Each agent of the forecast task at each budget its prompt asked: per job, its forecast against epochs, our runs as black dots, and the rows."""
    forecast = [f"forecast_{q}_e{x}" for q in ("runtime_s", "test_score") for x in ("", "_low", "_high")]
    b = tidy.budgets(eps).assign(uid=lambda d: d["run"] + "/" + d["agent"] + "/" + d["episode"]).sort_values(["uid", "epochs"])
    cols, done = BAR + ["episode", "uid", "epochs", "kind", *forecast, "actual_seconds_e", "actual_test_score_e"], made[made["state"].eq("SUCCEEDED") & made["where"].ne("cpu") & made["budget"].notna()]

    def per(j, data, f, y, value, label):
        g, o = positive(f, y), done[(done["job"] == j) & (done[value] > 0)]
        epochs, dom = sorted(set(f["epochs"]) | set(o["budget"])), span(g[y], o[value])
        return plot(line(data, x="epochs", y=y, z="uid", fx="gpu", stroke="cli", stroke_opacity=0.35, sort={"channel": "x"}, filter_by=filt),
                    dot(data, x="epochs", y=y, fx="gpu", fill="cli", r=2, fill_opacity=0.5, filter_by=filt, channels=WHO | {"epochs": "epochs", "forecast": y}),
                    dot(Data.from_dataframe(compact(o[["gpu", "budget", value]])), x="budget", y=value, fx="gpu", fill="black", r=2.5),
                    title=named(j), x_scale="log", x_domain=[epochs[0] / 1.3, epochs[-1] * 1.3], x_ticks=epochs, x_tick_format="~g", x_label="epochs",
                    y_scale="log", y_domain=dom, y_ticks=ticks(dom), y_tick_format="~g", y_label=label, fx_label=None, **BESIDE, grid=True, width=WIDE, height=320,
                    legend=legend("color", frame_anchor="bottom"), **CLI)

    plots = []
    for j in sorted(b["job"].unique()):
        f = b[b["job"] == j]
        data = Data.from_dataframe(compact(f[cols]))
        plots += [per(j, data, f, "forecast_test_score_e", "score", f"forecast_test_score_e ({items.TASKS[items.JOBS[j][0]]['metric']})"),
                  per(j, data, f, "forecast_runtime_s_e", "seconds", "forecast_runtime_s_e (seconds)")]
    table_cols = WHO_COLS + [num("condition"), num("gpu", width=65), num("job", width=55), num("epochs", width=85)] + [
        num(c, fmt=".4~f" if "score" in c else None, width=190) for c in forecast] + [num("actual_seconds_e", fmt=".1f", width=140), num("actual_test_score_e", fmt=".4~f", width=160)]
    return ("R", "Forecasts at each budget, as rows: what an agent forecast a job's run takes and scores at each number of epochs, beside the actual", "", [],
            [("forecast_test_score_e and forecast_runtime_s_e against epochs, a line per agent, per job and gpu · black dots: score and seconds of our runs at each budget",
              plots + [rows(Data.from_dataframe(compact(b[cols])), filt, table_cols, width=1220)],
              [*tidy.defined("epochs", *forecast, "actual_seconds_e", "actual_test_score_e", "budget")])])


def build():
    eps, made = tidy.episodes(), tidy.runs()
    page, filt, mine = HtmlPage(), Selection.crossfilter(), Selection.intersect()
    fc, po = (eps[eps["task"] == t].reset_index(drop=True) for t in ("forecast", "post"))
    ep, fc_all, t = episodes(eps), episodes(fc), ran(made)
    sections = [matrix(eps, filt), at_budgets(eps, made, filt)]

    def ranked(col):
        """The agents of the forecast task with a positive `col`, its LOW only where positive, each with its rank by `col` within its job and GPU."""
        f = positive(fc, col)
        return f.assign(**{f"{col}_low": f[f"{col}_low"].where(f[f"{col}_low"] > 0)}, rank=f.groupby(["job", "gpu"])[col].rank(method="first"))

    def against(j, d, runs, col, value, scored, label):
        """Per GPU, job j's forecasts `col` (bar: LOW to HIGH) by their median, beside the `value` of each of our runs `runs` (black lines)."""
        f, s = d[d["job"] == j], runs[runs["job"] == j]
        data, y = episodes(f), span(f[col], f[f"{col}_low"], f[f"{col}_high"], s[value])
        return plot(rule_y(ran(s), y=value, fx="gpu", stroke="black", stroke_opacity=0.7),
                    rule_x(data, x="rank", y1=f"{col}_low", y2=f"{col}_high", fx="gpu", stroke="cli", stroke_opacity=0.35, filter_by=filt),
                    dots(data, filt, "rank", col, "condition", {"forecast": col, "low": f"{col}_low", "high": f"{col}_high", "actual": scored}, fx="gpu"),
                    title=named(j), x_domain=[0, f["rank"].max() + 1], x_ticks=[], x_label="rank", fx_label=None,
                    y_scale="log", y_domain=y, y_ticks=ticks(y), y_tick_format="~g", y_label=label, **BESIDE, grid=True, width=WIDE, height=380,
                    legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **CLI)

    a, timed = ranked("forecast_runtime_s"), made[made["full"]]
    sections.append(("A", "Forecast runtime: forecast.RUNTIME_S (LOW to HIGH) beside the seconds of our runs of that job and GPU",
                     f"{len(a)} of {len(fc)} agents of the forecast task drawn, ordered by their median within each job and GPU; the other {len(fc) - len(a)} gave no runtime. "
                     f"Black lines: each of our {len(timed)} runs at a job's full budget (as published and in the sweep), the actual seconds a forecast is now scored against; not filtered. "
                     "Colour: CLI. Symbol: condition. The table's actual_seconds is what an agent was scored against when it ran.", [],
                     [("forecast_runtime_s, bar forecast_runtime_s_low to forecast_runtime_s_high, by rank, per job and gpu · black lines: seconds of our runs at the full budget",
                       [against(j, a, timed, "forecast_runtime_s", "seconds", "actual_seconds", "forecast_runtime_s") for j in sorted(a["job"].unique())], []),
                      rows(fc_all, filt, WHO_COLS + [num("condition"), num("gpu", width=65), num("job", width=55), num("forecast_runtime_s", width=140), num("forecast_runtime_s_low", width=170),
                                                     num("forecast_runtime_s_high", width=170), num("actual_seconds", fmt=".1f", width=120), num("runtime_ln", fmt=".2f", width=100), num("answered", width=90)], width=1300)]))

    b = positive(fc, "margin", "runtime_ratio")
    xd, yd = span(b["runtime_ratio"]), span(b["margin"])
    sections.append(("B", "Requested limit: request.REQUEST_S / forecast.RUNTIME_S against forecast.RUNTIME_S / actual.seconds",
                     f"{len(b)} of {len(fc)} agents of the forecast task drawn. Solid grey: 1 on each axis. Dotted: the request equals the actual (y = 1/x); below it the requested limit is shorter than our run.", [],
                     [("margin = request_s / forecast_runtime_s against runtime_ratio = forecast_runtime_s / actual_seconds",
                       [loglog(rule_x(x=[1], stroke="#999"), rule_y(y=[1], stroke="#999"), diagonal(xd, yd, power=-1, dash="1 3"),
                               dots(episodes(b), filt, "runtime_ratio", "margin", "condition", {"request_s": "request_s", "forecast": "forecast_runtime_s", "actual": "actual_seconds"}),
                               x_label="runtime_ratio", y_label="margin", xdom=xd, ydom=yd, fmt="~g")], []),
                      rows(fc_all, filt, WHO_COLS + [num("condition"), num("gpu", width=65), num("job", width=55), num("request_s", width=90), num("forecast_runtime_s", width=140),
                                                     num("actual_seconds", fmt=".1f", width=120), num("margin", fmt=".2f"), num("runtime_ratio", fmt=".2f", width=110)], width=1100)]))

    seeds = ["none"] + sorted({s for s in made["seed_label"] if s != "none"}, key=int)
    by_seed = dict(color_domain=seeds, color_range=(SEEDS * len(seeds))[:len(seeds)])
    gpu_runs = made[made["where"].ne("cpu") & (made["score"] > 0)]

    def curve(j):
        g = gpu_runs[gpu_runs["job"] == j]
        epochs, y = sorted({e for e, _, _ in items.BUDGETS.get(j, [])} | set(g["budget"].dropna())), span(g["score"], pad=1.05)
        return plot(dot(ran(g), x="budget", y="score", fx="gpu", fill="seed_label", symbol="variant", r=4, fill_opacity=0.8, filter_by=mine,
                        channels={"line": "line", "variant": "variant", "seed": "seed_label", "epochs": "budget", "score": "score", "seconds": "seconds", "pool job": "pool_job"}),
                    title=named(j), x_scale="log", x_domain=[epochs[0] / 1.3, epochs[-1] * 1.3], x_ticks=epochs, x_tick_format="~g", x_label="budget",
                    y_scale="log", y_domain=y, y_ticks=ticks(y), y_tick_format="~g", y_label=f"score ({items.TASKS[items.JOBS[j][0]]['metric']})", fx_label=None, **BESIDE, grid=True, width=WIDE, height=320,
                    legend=[legend("color", label="seed", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **by_seed)

    labels = sorted({l for l in made["budget_label"] if l.isdigit()}, key=int) + sorted({l for l in made["budget_label"] if not l.isdigit()})
    sections.append(("C", "Our runs of the jobs (out/items/items.jsonl): score against epochs per job and GPU, one dot per seed; and seconds",
                     f"{len(made)} rows, {made['score'].notna().sum()} with a score. A score is comparable only within a job (its task's metric), so each job has its own y scale. "
                     "Epochs: a sweep row's budget; a run of the script as published sits at the job's last budget. CPU runs (job a's proxies) are only in the seconds plot and the table. "
                     "A title says whether a smaller budget cuts the job's run off or rescales it, and marks a job whose score does not move with its budget. "
                     "In the table, truth marks the seeds a forecast's score is scored against, full the runs its runtime and memory are. "
                     "These filters act on this section only.",
                     [checkbox_group(t, column=c, label=c.capitalize(), target=mine) for c in ("job", "gpu", "variant", "state")],
                     [("score against budget, a dot per seed, per job and gpu", [curve(j) for j in sorted(gpu_runs["job"].unique())], []),
                      ("seconds against budget_label, per job and gpu",
                       [plot(dot(t, x="budget_label", y="seconds", fx="gpu", fy="job", fill="seed_label", symbol="state", r=4, fill_opacity=0.8, filter_by=mine,
                                 channels={"line": "line", "variant": "variant", "seed": "seed_label", "epochs": "budget_label", "state": "state", "seconds": "seconds", "pool job": "pool_job"}),
                             y_scale="log", y_label="seconds", x_domain=labels, x_tick_rotate=-40, margin_bottom=70, x_label="budget_label", fx_label=None, fy_label=None, grid=True,
                             width=900, height=90 * made["job"].nunique() + 110, legend=[legend("color", label="seed", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **by_seed)], []),
                      rows(t, mine, [num("line", width=60), num("job", width=55), num("metric", width=85), num("gpu", width=65), num("variant", width=85), num("seed_label", "seed", width=65),
                                     num("budget", "epochs", width=75), num("state", width=105), num("exit", width=60), num("seconds", width=80), num("score", fmt=".5~f", width=85),
                                     num("peak_gpu_mb", "peak MB"), num("pool_job", "pool job"), num("truth", width=70), num("full", width=60)], height=440)]))

    d, truth = ranked("forecast_test_score"), made[made["truth"]]
    sections.append(("D", "Forecast score: forecast.TEST_SCORE (LOW to HIGH) beside the seeds of our runs of that job and GPU",
                     f"{len(d)} of {len(fc)} agents of the forecast task drawn, ordered by their median within each job and GPU; bar: LOW to HIGH. Black lines: each of our {len(truth)} seeds at a job's full budget "
                     "(the sweep's at its last budget, else the runs as published), the actual score a forecast is now scored against; not filtered. "
                     "The table's actual_test_score is what an agent was scored against when it ran.", [],
                     [("forecast_test_score, bar forecast_test_score_low to forecast_test_score_high, by rank, per job and gpu · black lines: score of our seeds at the full budget",
                       [against(j, d, truth, "forecast_test_score", "score", "actual_test_score", f"forecast_test_score ({items.TASKS[items.JOBS[j][0]]['metric']})") for j in sorted(d["job"].unique())], []),
                      rows(fc_all, filt, WHO_COLS + [num("condition"), num("job", width=55), num("gpu", width=65), num("forecast_test_score", fmt=".4~f", width=150),
                                                     num("forecast_test_score_low", fmt=".4~f", width=180), num("forecast_test_score_high", fmt=".4~f", width=180),
                                                     num("actual_test_score", fmt=".4~f", width=140), num("score_ln", fmt=".2f", width=90)], width=1350)]))

    e1, e2 = positive(po, "first_limit_s", "first_seconds"), positive(po, "usd_ratio")
    xd, yd = span(e1["first_seconds"]), span(e1["first_limit_s"])
    sections.append(("E", "Post: the first job's limit against its seconds; and usd / usd_reference",
                     f"Left: {len(e1)} of {len(po)} agents of the post task posted a first job. Symbol: its state. Right: {len(e2)} of {len(po)} spent money (usd > 0); dashed: the cost of the reference. Colour: CLI.", [],
                     [("first_limit_s against first_seconds", [loglog(diagonal(xd, yd), dots(episodes(e1), filt, "first_seconds", "first_limit_s", "first_state",
                                                                                                {"gpu": "first_gpu", "state": "first_state", "limit s": "first_limit_s", "seconds": "first_seconds"}),
                                                                                         x_label="first_seconds", y_label="first_limit_s", xdom=xd, ydom=yd)], []),
                      ("usd_ratio = usd / usd_reference by job", [plot(rule_x(x=[1], stroke="#777", stroke_dasharray="4 4"),
                                                                       dot(episodes(e2), x="usd_ratio", y="job", fill="cli", filter_by=filt, r=4, fill_opacity=0.65,
                                                                           channels=WHO | {"usd": "usd", "usd_reference": "usd_reference", "jobs": "jobs", "correct": "correct"}),
                                                                       x_scale="log", x_tick_format="~g", x_label="usd_ratio", y_label=None, grid=True, width=SIDE, height=300, margin_left=110,
                                                                       legend=legend("color", frame_anchor="bottom"), **CLI)], []),
                      rows(episodes(po), filt, WHO_COLS + [num("job", width=120), num("first_gpu", width=80), num("first_state", width=105), num("first_limit_s", width=90),
                                                           num("first_seconds", width=100), num("jobs", width=60), num("failed", width=65), num("usd", fmt=".4~f", width=75),
                                                           num("usd_reference", fmt=".4~f", width=110), num("usd_ratio", fmt=".2f", width=80), num("correct", width=80)], width=1300)]))

    f1, f2 = positive(eps, "forecast_self_seconds", "agent_s"), positive(eps, "forecast_self_output_tokens", "output_tokens")
    x1, y1, x2, y2 = span(f1["agent_s"]), span(f1["forecast_self_seconds"]), span(f2["output_tokens"]), span(f2["forecast_self_output_tokens"])
    sections.append(("F", "An agent's own run: forecast.SELF_SECONDS against agent_s; forecast.SELF_OUTPUT_TOKENS against output_tokens",
                     f"Left: {len(f1)} of {len(eps)} agents; right: {len(f2)} of {len(eps)} (earlier runs did not ask, and zeros cannot sit on a log axis). Symbol: task.", [],
                     [("forecast_self_seconds against agent_s", [loglog(diagonal(x1, y1), dots(episodes(f1), filt, "agent_s", "forecast_self_seconds", "task",
                                                                                                 {"forecast": "forecast_self_seconds", "low": "forecast_self_seconds_low", "high": "forecast_self_seconds_high", "agent_s": "agent_s"}),
                                                                                          x_label="agent_s", y_label="forecast_self_seconds", xdom=x1, ydom=y1)], []),
                      ("forecast_self_output_tokens against output_tokens", [loglog(diagonal(x2, y2), dots(episodes(f2), filt, "output_tokens", "forecast_self_output_tokens", "task",
                                                                                                            {"forecast": "forecast_self_output_tokens", "actual": "output_tokens"}),
                                                                                     x_label="output_tokens", y_label="forecast_self_output_tokens", xdom=x2, ydom=y2)], []),
                      rows(ep, filt, WHO_COLS + [num("task", width=75), num("forecast_self_seconds", width=160), num("forecast_self_seconds_low", width=190), num("forecast_self_seconds_high", width=190),
                                                 num("agent_s", width=80), num("forecast_self_output_tokens", width=210), num("output_tokens", width=110),
                                                 num("forecast_self_tool_calls", width=180), num("tool_calls", width=80)], width=1500)]))

    sections.append(prebuilt())
    return render(page, bar(ep, filt), sections, "styx agents", '<a href="questions.html">questions</a> — the filters above act on M, R, A, B, D, E and F; C has its own. '
                  f'{len(eps)} agents in {eps["run"].nunique()} runs; {len(made)} runs we made of the jobs. Hover a point for its agent; tables sort and filter by column.')


def prebuilt():
    """The library's own heatmaps of the eval logs, by the real model."""
    ev, sa = tidy.evals(), tidy.samples()
    ev = ev[(ev["status"] == "success") & (ev["task_name"] == "post")]
    keep = ["id", "model", "s_correct", "s_runtime_in"]
    post, fc = (Data.from_dataframe(sa.loc[sa["task_name"] == t].reindex(columns=keep).reset_index(drop=True)) for t in ("post", "forecast"))
    return ("G", "Inspect Viz's prebuilt heatmaps of the .eval logs",
            f"{len(ev)} post logs with status success, {len(sa)} samples; not filtered by the bar above. Models are the logs' agent and model args (opencode names keep their slash). "
            "A cell averages every log or sample in it; white is a share of 0 or no sample at all.", [],
            [("correct by run and model; correct and runtime_in by sample id and model, of the eval logs",
              [scores_heatmap(Data.from_dataframe(ev[["run", "model", "score_correct_mean"]].reset_index(drop=True)), task_name="run", model_name="model", score_value="score_correct_mean",
                              title="scores_heatmap: post, correct share per log, run × model", width=900, height=520, margin_left=280, x_tick_rotate=-30, margin_bottom=90),
               sample_heatmap(post, id="id", model_name="model", score_value="s_correct", title="sample_heatmap: post, correct share per sample id", width=900, height=520, margin_left=280),
               sample_heatmap(fc, id="id", model_name="model", score_value="s_runtime_in", title="sample_heatmap: forecast, share with runtime in LOW..HIGH per sample id",
                              width=900, height=560, margin_left=280, x_tick_rotate=-45, margin_bottom=90)], [])])


def render(page, bar, sections, name, nav):
    """The page: the bar, links to its sections and `nav`, and each section. A section's (heading, components, lines) is a block of its
    own row: the heading, which says what its plots draw, the components side by side, and under them the lines, as the repo defines
    what they draw; a string is HTML of its own row (told)."""
    inputs = lambda comps, cls: f'<details class="{cls}" open><summary>Filters</summary><div>' + "".join(f"<div>{page.fragment(c)}</div>" for c in comps) + "</div></details>"
    shown = lambda c: f"<div>{page.fragment(c)}</div>"
    block = lambda heading, cs, lines: ('<div class="block">' + (f'<h3>{html.escape(heading)}</h3>' if heading else "") + '<div class="row">' + "".join(map(shown, cs)) + "</div>"
                                        + ('<ul class="defs">' + "".join(f"<li>{html.escape(l)}</li>" for l in lines) + "</ul>" if lines else "") + "</div>")
    parts = [inputs(bar, "bar"), f'<p class="nav">{" · ".join(f"<a href=#s{k}>{k}</a>" for k, *_ in sections)} — {nav}</p>']
    for key, title, note, ins, comps in sections:
        parts.append(f'<section id="s{key}"><h2>{key}. {html.escape(title)}</h2>' + (f'<p class="note">{html.escape(note)}</p>' if note else "")
                     + (inputs(ins, "filters") if ins else "") + '<div class="row">' + "".join(block(*c) if isinstance(c, tuple) else c if isinstance(c, str) else shown(c) for c in comps) + "</div></section>")
    return PAGE.format(name=name, head=page.head(), body="\n".join(parts))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{name}</title>
<style>
body {{ margin: 0; padding: 0 16px 40px; background: #fcfcfb; color: #0b0b0b; font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", Arial, sans-serif; }}
.bar > div, .filters > div {{ display: flex; flex-wrap: wrap; gap: 4px 28px; align-items: flex-end; padding: 2px 0 8px; }}
.bar {{ position: sticky; top: 0; z-index: 10; background: #fcfcfb; border-bottom: 1px solid #ccc; max-height: 70vh; overflow-y: auto; }}
summary {{ cursor: pointer; font-size: 13px; font-weight: 600; padding: 8px 0; }}
.bar .mosaic-widget, .filters .mosaic-widget {{ margin: 0 !important; }}
.nav {{ font-size: 13px; color: #52514e; }}
section {{ margin-top: 28px; scroll-margin-top: 110px; }}
h2 {{ font-size: 17px; margin: 0 0 4px; }}
.note {{ color: #52514e; font-size: 13px; margin: 0 0 8px; max-width: 1100px; }}
.row {{ display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }}
.row > div {{ max-width: 100%; overflow-x: auto; }}
.row > .block {{ flex: 0 0 100%; }}
.block h3 {{ font: 600 12.5px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; margin: 14px 0 6px; overflow-wrap: anywhere; }}
.told {{ flex: 0 0 100%; max-width: 100%; }}
.told summary {{ font: 600 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; padding: 4px 0; }}
.told pre {{ margin: 2px 0 6px; padding: 8px 10px; max-width: 1100px; background: #f1f0ec; font: 11.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }}
.defs {{ margin: 6px 0 0; padding: 0; list-style: none; font: 11.5px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; color: #52514e; overflow-wrap: anywhere; }}
.defs li {{ margin: 0 0 3px; }}
@media (max-width: 800px) {{ body {{ padding: 0 10px 40px; }} section {{ scroll-margin-top: 50px; }} .row svg[class^="plot-"] {{ max-width: none; }} }}
</style>
{head}
</head><body>
{body}
<script>if (matchMedia("(max-width: 800px)").matches) document.querySelectorAll("details").forEach(d => d.open = false);</script>
</body></html>
"""


if __name__ == "__main__":
    from view import questions

    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    os.makedirs(out, exist_ok=True)
    for name, make in (("index.html", build), ("questions.html", questions.build)):
        with open(f"{out}/{name}", "w") as f:
            f.write(make())
        print(f"{out}/{name}", os.path.getsize(f"{out}/{name}"), "bytes")
