"""Each row as it is: which episodes were run, task by prompt, model, condition, GPU and job? What did an agent forecast a job would
take, need and score, at its full budget and at each smaller one, and what limit would it ask? What did its first posted job ask and
take, and what did its jobs cost against ours? What did it say its own run would take? And what do our own runs of the jobs take and
score, seed by seed and budget by budget?

python -m view.build [dir]            writes index.html and questions.html (view.questions) to dir, view/dist by default, from the
                                      episodes table, the episodes' own files, out/items/items.jsonl and the eval logs as they are now
python -m http.server -d view/dist    serves them on http://localhost:8000

One Inspect Viz HtmlPage per page: one in-browser DuckDB and one selection for every widget, so the bar at the top filters every episode
view; our own runs have filters of their own. The data is inline; the scripts come from cdn.jsdelivr.net.
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

from styx import items
from view import tidy

OUT = "view/dist"
CLI = dict(color_domain=["claude", "codex", "opencode"], color_range=["#2a78d6", "#eb6834", "#1baf7a"])  # fixed order, by harness
SEEDS = ["#9a9a96", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#2a78d6", "#1baf7a"]
SIDE = 440
WIDE = 620  # a plot titled with its job, as wide as its longest title
WHO = {"agent": "agent", "episode": "episode", "run": "run"}
BESIDE = dict(y_label_anchor="center", margin_left=70)  # a titled plot's y label runs up its axis: at the top it would sit under the title
BAR = ["task", "harness", "model", "condition", "gpu", "job", "run", "asked"]  # what the bar filters on: every table it filters has them
KEY = ["task", "asked", "harness", "model", "condition", "gpu", "job"]  # an episode's cell of what was run
EPISODE = ["task", "run", "harness", "model", "agent", "episode", "condition", "gpu", "job", "asked", "answered", "rank",
           "said_runtime_s", "said_runtime_s_low", "said_runtime_s_high", "said_request_s", "ours_seconds", "runtime_ln", "runtime_ratio", "margin",
           "said_test_score", "said_test_score_low", "said_test_score_high", "ours_test_score", "score_ln",
           "first_gpu", "first_state", "first_limit_s", "first_seconds", "jobs", "failed", "usd", "usd_ours", "usd_ratio", "correct",
           "own_self_seconds", "own_self_seconds_low", "own_self_seconds_high", "agent_s", "own_self_output_tokens", "output_tokens", "own_self_tool_calls", "tool_calls"]
OURS = ["line", "job", "metric", "gpu", "where", "variant", "seed_label", "budget", "budget_label", "state", "exit", "seconds", "score", "peak_gpu_mb", "pool_job", "truth", "full"]


def episodes(df):
    return Data.from_dataframe(df.reindex(columns=EPISODE).reset_index(drop=True))


def runs(df):
    return Data.from_dataframe(df[OURS].reset_index(drop=True))


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
    return dot(data, x=x, y=y, fill="harness", symbol=symbol, filter_by=filt, r=4, fill_opacity=0.65, channels=WHO | extra, **options)


def rows(data, filt, cols, width=980, height=SIDE + 80):
    return table(data, filter_by=filt, columns=cols, width=width, height=height, pagination=True)


def num(name, label=None, fmt=None, width=85):
    return column(name, label=label or name, format=fmt, width=width)


WHO_COLS = [num("agent", width=190), num("episode", width=125), num("run", width=125)]


def bar(ep, filt):
    """The filters of every episode view, on the columns of BAR."""
    return [select(ep, column="task", label="Task", target=filt, filter_by=filt),
            checkbox_group(ep, column="harness", label="CLI", target=filt, filter_by=filt),
            select(ep, column="model", label="Model", target=filt, filter_by=filt, multiple=True),
            checkbox_group(ep, column="condition", label="Condition", target=filt, filter_by=filt),
            checkbox_group(ep, column="gpu", label="GPU", target=filt, filter_by=filt),
            select(ep, column="job", label="Job", target=filt, filter_by=filt, multiple=True),
            select(ep, column="run", label="Run", target=filt, filter_by=filt, multiple=True),
            select(ep, column="asked", label="Prompt asked", target=filt, filter_by=filt, multiple=True)]


def matrix(eps, filt):
    """Every episode in one cell of KEY: a plot of counts per task and prompt, and the same cells as a table."""
    assert eps.groupby(KEY).size().sum() == len(eps), "an episode outside every cell"

    def counts(d):
        task, asked, models, jobs = d["task"].iloc[0], d["asked"].iloc[0], tidy.ordered(d), sorted(d["job"].unique())
        name = f"{task} · {asked}"
        facets, rows_ = (dict(fx="gpu", fy="condition"), d["condition"].nunique()) if task == "forecast" else ({}, 1)
        data = Data.from_dataframe(compact(d[BAR + ["episode"]]))
        return plot(cell(data, x="job", y="model", fill=count(), filter_by=filt, inset=0.5, **facets),
                    text(data, x="job", y="model", text=count(), filter_by=filt, fill="#0b0b0b", **facets),
                    title=title(name, margin_top=40), x_domain=jobs, y_domain=models, x_label=None, y_label=None, fx_label=None, fy_label=None,
                    x_tick_rotate=-30 if task == "post" else 0, margin_left=250, margin_bottom=70 if task == "post" else 30, margin_right=60,
                    width=max(310 + 30 * len(jobs) * d["gpu"].nunique(), fits(name, 250, 60)), height=18 * len(models) * rows_ + 120, color_scale="log", color_range=["#e3edfa", "#6f9fe0"])

    every = Data.from_dataframe(compact(eps[BAR + ["episode"]].assign(answered=eps["answered"].astype(float))))  # DuckDB sums no booleans
    cols = [num(c, width=w) for c, w in zip(KEY, (75, 300, 80, 250, 80, 65, 120))] + [column({"count": "episode"}, label="scored", width=75), column({"sum": "answered"}, label="answered", width=100)]
    return ("M", "What was run: an episode's task, the keys its prompt asked, CLI, model, condition, GPU and job", "", [],
            [([counts(d)], []) for _, d in sorted(eps.groupby(["task", "asked"]), key=lambda g: (g[0][0], g[1]["run"].min()))]
            + [([rows(every, filt, cols, width=1220)], ["fill, text, scored = count(*)", "answered = sum(answered)", *tidy.defined(*KEY, "answered")])])


def at_budgets(eps, filt):
    """Each forecast episode at each budget its prompt asked: per job, what it said against epochs, our runs as black dots, and the rows."""
    said = [f"said_{q}_e{x}" for q in ("runtime_s", "test_score") for x in ("", "_low", "_high")]
    b = tidy.budgets(eps).assign(uid=lambda d: d["run"] + "/" + d["agent"] + "/" + d["episode"]).sort_values(["uid", "epochs"])
    cols, seeds = BAR + ["agent", "episode", "uid", "epochs", "kind", *said, "ours_median_s", "ours_median_score"], tidy.seeds(b)

    def per(j, data, f, s, y, ours, label):
        g, o = positive(f, y), s[s[ours] > 0]
        epochs, dom = sorted(set(f["epochs"]) | set(o["epochs"])), span(g[y], o[ours])
        return plot(line(data, x="epochs", y=y, z="uid", fx="gpu", stroke="harness", stroke_opacity=0.35, sort={"channel": "x"}, filter_by=filt),
                    dot(data, x="epochs", y=y, fx="gpu", fill="harness", r=2, fill_opacity=0.5, filter_by=filt, channels=WHO | {"epochs": "epochs", "said": y}),
                    dot(Data.from_dataframe(compact(o[["gpu", "epochs", ours]])), x="epochs", y=ours, fx="gpu", fill="black", r=2.5),
                    title=named(j), x_scale="log", x_domain=[epochs[0] / 1.3, epochs[-1] * 1.3], x_ticks=epochs, x_tick_format="~g", x_label="epochs",
                    y_scale="log", y_domain=dom, y_ticks=ticks(dom), y_tick_format="~g", y_label=label, fx_label=None, **BESIDE, grid=True, width=WIDE, height=320,
                    legend=legend("color", frame_anchor="bottom"), **CLI)

    plots = []
    for j in sorted(b["job"].unique()):
        f, s = b[b["job"] == j], seeds[seeds["job"] == j]
        data = Data.from_dataframe(compact(f[cols]))
        plots += [per(j, data, f, s, "said_test_score_e", "test_score", f"said_test_score_e ({items.TASKS[items.JOBS[j][0]]['metric']})"),
                  per(j, data, f, s, "said_runtime_s_e", "seconds", "said_runtime_s_e (seconds)")]
    table_cols = WHO_COLS + [num("condition"), num("gpu", width=65), num("job", width=55), num("epochs", width=85)] + [
        num(c, fmt=".4~f" if "score" in c else None, width=190) for c in said] + [num("ours_median_s", fmt=".1f", width=140), num("ours_median_score", fmt=".4~f", width=160)]
    return ("R", "Forecasts at each budget, as rows: what an episode said a job's run takes and scores at each number of epochs, beside the medians of ours", "", [],
            [(plots + [rows(Data.from_dataframe(compact(b[cols])), filt, table_cols, width=1220)],
              [*tidy.defined("epochs", *said, "ours_median_s", "ours_median_score"), f"black dots: {tidy.source('seeds', tidy.seeds)}"])])


def build():
    eps, own = tidy.episodes(), tidy.ours()
    page, filt, mine = HtmlPage(), Selection.crossfilter(), Selection.intersect()
    fc, po = (eps[eps["task"] == t].reset_index(drop=True) for t in ("forecast", "post"))
    ep, fc_all, t = episodes(eps), episodes(fc), runs(own)
    sections = [matrix(eps, filt), at_budgets(eps, filt)]

    def ranked(said):
        """The forecast episodes with a positive `said`, its LOW only where positive, each with its rank by `said` within its job and GPU."""
        f = positive(fc, said)
        return f.assign(**{f"{said}_low": f[f"{said}_low"].where(f[f"{said}_low"] > 0)}, rank=f.groupby(["job", "gpu"])[said].rank(method="first"))

    def against(j, d, ours, said, value, scored, label):
        """Per GPU, job j's forecasts `said` (bar: LOW to HIGH) by their median, beside the `value` of each of our runs `ours` (black lines)."""
        f, s = d[d["job"] == j], ours[ours["job"] == j]
        data, y = episodes(f), span(f[said], f[f"{said}_low"], f[f"{said}_high"], s[value])
        return plot(rule_y(runs(s), y=value, fx="gpu", stroke="black", stroke_opacity=0.7),
                    rule_x(data, x="rank", y1=f"{said}_low", y2=f"{said}_high", fx="gpu", stroke="harness", stroke_opacity=0.35, filter_by=filt),
                    dots(data, filt, "rank", said, "condition", {"said": said, "low": f"{said}_low", "high": f"{said}_high", "scored against": scored}, fx="gpu"),
                    title=named(j), x_domain=[0, f["rank"].max() + 1], x_ticks=[], x_label="forecasts, by their median →", fx_label=None,
                    y_scale="log", y_domain=y, y_ticks=ticks(y), y_tick_format="~g", y_label=label, **BESIDE, grid=True, width=WIDE, height=380,
                    legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **CLI)

    a, timed = ranked("said_runtime_s"), own[own["full"]]
    sections.append(("A", "Forecast runtime: said.RUNTIME_S (LOW to HIGH) beside the seconds of our runs of that job and GPU",
                     f"{len(a)} of {len(fc)} forecast episodes drawn, ordered by their median within each job and GPU; bar: LOW to HIGH; the other {len(fc) - len(a)} gave no runtime. "
                     f"Black lines: each of our {len(timed)} runs at a job's full budget (as published and in the sweep), what a forecast's runtime is now scored against; not filtered. "
                     "Colour: CLI. Symbol: condition. The table's 'ours s' is the truth an episode was scored against when it ran. Table: every forecast episode.", [],
                     [against(j, a, timed, "said_runtime_s", "seconds", "ours_seconds", "said.RUNTIME_S (seconds)") for j in sorted(a["job"].unique())] + [
                      rows(fc_all, filt, WHO_COLS + [num("condition"), num("gpu", width=65), num("job", width=55), num("said_runtime_s", "said s"), num("said_runtime_s_low", "low s"),
                                                     num("said_runtime_s_high", "high s"), num("ours_seconds", "ours s", ".1f"), num("runtime_ln", "ln(said/ours)", ".2f", 105), num("answered", width=90)])]))

    b = positive(fc, "margin", "runtime_ratio")
    xd, yd = span(b["runtime_ratio"]), span(b["margin"])
    sections.append(("B", "Forecast limit: said.REQUEST_S / said.RUNTIME_S against said.RUNTIME_S / ours.seconds",
                     f"{len(b)} of {len(fc)} forecast episodes drawn. Solid grey: 1 on each axis. Dotted: request = truth (y = 1/x); below it the requested limit is shorter than our run.", [],
                     [loglog(rule_x(x=[1], stroke="#999"), rule_y(y=[1], stroke="#999"), diagonal(xd, yd, power=-1, dash="1 3"),
                             dots(episodes(b), filt, "runtime_ratio", "margin", "condition", {"request s": "said_request_s", "said s": "said_runtime_s", "ours s": "ours_seconds"}),
                             x_label="said.RUNTIME_S / ours.seconds (forecast error) →", y_label="↑ said.REQUEST_S / said.RUNTIME_S (margin)", xdom=xd, ydom=yd, fmt="~g"),
                      rows(fc_all, filt, WHO_COLS + [num("condition"), num("gpu", width=65), num("job", width=55), num("said_request_s", "request s"), num("said_runtime_s", "said s"),
                                                     num("ours_seconds", "ours s", ".1f"), num("margin", "margin", ".2f"), num("runtime_ratio", "said/ours", ".2f")])]))

    seeds = ["none"] + sorted({s for s in own["seed_label"] if s != "none"}, key=int)
    by_seed = dict(color_domain=seeds, color_range=(SEEDS * len(seeds))[:len(seeds)])
    gpu_runs = own[own["where"].ne("cpu") & (own["score"] > 0)]

    def curve(j):
        g = gpu_runs[gpu_runs["job"] == j]
        epochs, y = sorted({e for e, _, _ in items.BUDGETS.get(j, [])} | set(g["budget"].dropna())), span(g["score"], pad=1.05)
        return plot(dot(runs(g), x="budget", y="score", fx="gpu", fill="seed_label", symbol="variant", r=4, fill_opacity=0.8, filter_by=mine,
                        channels={"line": "line", "variant": "variant", "seed": "seed_label", "epochs": "budget", "score": "score", "seconds": "seconds", "pool job": "pool_job"}),
                    title=named(j), x_scale="log", x_domain=[epochs[0] / 1.3, epochs[-1] * 1.3], x_ticks=epochs, x_tick_format="~g", x_label="epochs →",
                    y_scale="log", y_domain=y, y_ticks=ticks(y), y_tick_format="~g", y_label=items.TASKS[items.JOBS[j][0]]["metric"], fx_label=None, **BESIDE, grid=True, width=WIDE, height=320,
                    legend=[legend("color", label="seed", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **by_seed)

    labels = sorted({l for l in own["budget_label"] if l.isdigit()}, key=int) + sorted({l for l in own["budget_label"] if not l.isdigit()})
    sections.append(("C", "Our runs of the jobs (out/items/items.jsonl): score against epochs per job and GPU, one dot per seed; and seconds",
                     f"{len(own)} rows, {own['score'].notna().sum()} with a score. A score is comparable only within a job (its task's metric), so each job has its own y scale. "
                     "Epochs: a sweep row's budget; a run of the script as published sits at the job's last budget. CPU runs (job a's proxies) are only in the seconds plot and the table. "
                     "A title says whether a smaller budget cuts the job's run off or rescales it, and marks a job whose score does not move with its budget. "
                     "In the table, truth marks the seeds a forecast's score is scored against, full the runs its runtime and memory are. "
                     "These filters act on this section only.",
                     [checkbox_group(t, column=c, label=c.capitalize(), target=mine) for c in ("job", "gpu", "variant", "state")],
                     [curve(j) for j in sorted(gpu_runs["job"].unique())] + [
                      plot(dot(t, x="budget_label", y="seconds", fx="gpu", fy="job", fill="seed_label", symbol="state", r=4, fill_opacity=0.8, filter_by=mine,
                               channels={"line": "line", "variant": "variant", "seed": "seed_label", "epochs": "budget_label", "state": "state", "seconds": "seconds", "pool job": "pool_job"}),
                           y_scale="log", y_label="↑ seconds", x_domain=labels, x_tick_rotate=-40, margin_bottom=70, x_label="epochs (or variant) →", fx_label=None, fy_label=None, grid=True,
                           width=900, height=90 * own["job"].nunique() + 110, legend=[legend("color", label="seed", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **by_seed),
                      rows(t, mine, [num("line", width=60), num("job", width=55), num("metric", width=85), num("gpu", width=65), num("variant", width=85), num("seed_label", "seed", width=65),
                                     num("budget", "epochs", width=75), num("state", width=105), num("exit", width=60), num("seconds", width=80), num("score", fmt=".5~f", width=85),
                                     num("peak_gpu_mb", "peak MB"), num("pool_job", "pool job"), num("truth", width=70), num("full", width=60)], height=440)]))

    d, truth = ranked("said_test_score"), own[own["truth"]]

    sections.append(("D", "Forecast score: said.TEST_SCORE (LOW to HIGH) beside the seeds of our runs of that job and GPU",
                     f"{len(d)} of {len(fc)} forecast episodes drawn, ordered by their median within each job and GPU; bar: LOW to HIGH. Black lines: each of our {len(truth)} seeds at a job's full budget "
                     "(the sweep's at its last budget, else the runs as published), what a forecast is now scored against; not filtered. "
                     "The table's 'scored against' is the truth an episode was scored against when it ran.", [],
                     [against(j, d, truth, "said_test_score", "score", "ours_test_score", f"said.TEST_SCORE ({items.TASKS[items.JOBS[j][0]]['metric']})") for j in sorted(d["job"].unique())] + [
                      rows(fc_all, filt, WHO_COLS + [num("condition"), num("job", width=55), num("gpu", width=65), num("said_test_score", "said", ".4~f"), num("said_test_score_low", "low", ".4~f"),
                                                     num("said_test_score_high", "high", ".4~f"), num("ours_test_score", "scored against", ".4~f", 120), num("score_ln", "ln(said/ours)", ".2f", 105)])]))

    e1, e2 = positive(po, "first_limit_s", "first_seconds"), positive(po, "usd_ratio")
    xd, yd = span(e1["first_seconds"]), span(e1["first_limit_s"])
    sections.append(("E", "Post: first job's limit against its seconds; and usd / usd_ours",
                     f"Left: {len(e1)} of {len(po)} post episodes posted a first job. Symbol: its state. Right: {len(e2)} of {len(po)} spent money (usd > 0); dashed: same cost as ours. Colour: CLI.", [],
                     [loglog(diagonal(xd, yd), dots(episodes(e1), filt, "first_seconds", "first_limit_s", "first_state", {"gpu": "first_gpu", "state": "first_state", "limit s": "first_limit_s", "seconds": "first_seconds"}),
                             x_label="first.seconds →", y_label="↑ first.limit_s", xdom=xd, ydom=yd),
                      plot(rule_x(x=[1], stroke="#777", stroke_dasharray="4 4"),
                           dot(episodes(e2), x="usd_ratio", y="job", fill="harness", filter_by=filt, r=4, fill_opacity=0.65, channels=WHO | {"usd": "usd", "usd_ours": "usd_ours", "jobs": "jobs", "correct": "correct"}),
                           x_scale="log", x_tick_format="~g", x_label="usd / usd_ours →", y_label=None, grid=True, width=SIDE, height=300, margin_left=110, legend=legend("color", frame_anchor="bottom"), **CLI),
                      rows(episodes(po), filt, WHO_COLS + [num("job", width=120), num("first_gpu", "gpu", width=65), num("first_state", "state", width=105), num("first_limit_s", "limit s"),
                                                           num("first_seconds", "seconds"), num("jobs", width=60), num("failed", width=65), num("usd", fmt=".4~f", width=75),
                                                           num("usd_ours", fmt=".4~f"), num("usd_ratio", "usd/ours", ".2f"), num("correct", width=80)])]))

    f1, f2 = positive(eps, "own_self_seconds", "agent_s"), positive(eps, "own_self_output_tokens", "output_tokens")
    x1, y1, x2, y2 = span(f1["agent_s"]), span(f1["own_self_seconds"]), span(f2["output_tokens"]), span(f2["own_self_output_tokens"])
    sections.append(("F", "Own run: own.SELF_SECONDS against agent_s; own.SELF_OUTPUT_TOKENS against output_tokens",
                     f"Left: {len(f1)} of {len(eps)} episodes; right: {len(f2)} of {len(eps)} (earlier runs did not ask, and zeros cannot sit on a log axis). Symbol: task. Table: every episode.", [],
                     [loglog(diagonal(x1, y1), dots(episodes(f1), filt, "agent_s", "own_self_seconds", "task", {"said s": "own_self_seconds", "low": "own_self_seconds_low", "high": "own_self_seconds_high", "agent s": "agent_s"}),
                             x_label="agent_s →", y_label="↑ own.SELF_SECONDS", xdom=x1, ydom=y1),
                      loglog(diagonal(x2, y2), dots(episodes(f2), filt, "output_tokens", "own_self_output_tokens", "task", {"said": "own_self_output_tokens", "actual": "output_tokens"}),
                             x_label="output_tokens →", y_label="↑ own.SELF_OUTPUT_TOKENS", xdom=x2, ydom=y2),
                      rows(ep, filt, WHO_COLS + [num("task", width=75), num("own_self_seconds", "said s"), num("own_self_seconds_low", "low s"), num("own_self_seconds_high", "high s"),
                                                 num("agent_s", "agent s"), num("own_self_output_tokens", "said tok", width=90), num("output_tokens", "out tok", width=90),
                                                 num("own_self_tool_calls", "said calls", width=90), num("tool_calls", "calls", width=70)])]))

    sections.append(prebuilt())
    return render(page, bar(ep, filt), sections, "styx episodes", '<a href="questions.html">questions</a> — the filters above act on M, R, A, B, D, E and F; C has its own. '
                  f'{len(eps)} episodes in {eps["run"].nunique()} runs; {len(own)} runs of ours. Hover a point for its episode; tables sort and filter by column.')


def prebuilt():
    """The library's own heatmaps of the eval logs, by the real model."""
    ev, sa = tidy.evals(), tidy.samples()
    ev = ev[(ev["status"] == "success") & (ev["task_name"] == "post")]
    keep = ["id", "model", "s_correct", "s_runtime_in"]
    post, fc = (Data.from_dataframe(sa.loc[sa["task_name"] == t].reindex(columns=keep).reset_index(drop=True)) for t in ("post", "forecast"))
    return ("G", "Inspect Viz's prebuilt heatmaps of the .eval logs",
            f"{len(ev)} post logs with status success, {len(sa)} samples; not filtered by the bar above. Models are the logs' agent and model args (opencode names keep their slash). "
            "A cell averages every log or sample in it; white is a share of 0 or no sample at all.", [],
            [scores_heatmap(Data.from_dataframe(ev[["run", "model", "score_correct_mean"]].reset_index(drop=True)), task_name="run", model_name="model", score_value="score_correct_mean",
                            title="scores_heatmap: post, correct share per log, run × model", width=900, height=520, margin_left=280, x_tick_rotate=-30, margin_bottom=90),
             sample_heatmap(post, id="id", model_name="model", score_value="s_correct", title="sample_heatmap: post, correct share per sample id", width=900, height=520, margin_left=280),
             sample_heatmap(fc, id="id", model_name="model", score_value="s_runtime_in", title="sample_heatmap: forecast, share with runtime in LOW..HIGH per sample id",
                            width=900, height=560, margin_left=280, x_tick_rotate=-45, margin_bottom=90)])


def render(page, bar, sections, name, nav, keyed=True):
    """The page: the bar, links to its sections (by key, or by title where not `keyed`) and `nav`, and each section. A component, or
    a list of them, given with lines is a block of its own row: the components side by side, and under them the lines, as the repo
    defines what they draw."""
    inputs = lambda comps, cls: f'<details class="{cls}" open><summary>Filters</summary><div>' + "".join(f"<div>{page.fragment(c)}</div>" for c in comps) + "</div></details>"
    shown = lambda c: f"<div>{page.fragment(c)}</div>"
    block = lambda cs, lines: ('<div class="block"><div class="row">' + "".join(map(shown, cs if isinstance(cs, list) else [cs])) + "</div>"
                               + ('<ul class="defs">' + "".join(f"<li>{html.escape(l)}</li>" for l in lines) + "</ul>" if lines else "") + "</div>")
    parts = [inputs(bar, "bar"), f'<p class="nav">{" · ".join(f"<a href=#{k}>{k if keyed else html.escape(t)}</a>" for k, t, *_ in sections)} — {nav}</p>']
    for key, title, note, ins, comps in sections:
        parts.append(f'<section id="{key}"><h2>{f"{key}. " if keyed else ""}{html.escape(title)}</h2>' + (f'<p class="note">{html.escape(note)}</p>' if note else "")
                     + (inputs(ins, "filters") if ins else "") + '<div class="row">' + "".join(block(*c) if isinstance(c, tuple) else shown(c) for c in comps) + "</div></section>")
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
