"""Each row as it is: what did an agent forecast a job would take, need and score, and what limit would it ask? What did its first
posted job ask and take, and what did its jobs cost against ours? What did it say its own run would take? And what do our own runs
of the jobs take and score, seed by seed and budget by budget?

python -m view.build                  writes view/dist/index.html from the episodes table, out/items/items.jsonl and the eval logs as they are now
python -m http.server -d view/dist    serves it on http://localhost:8000

One Inspect Viz HtmlPage: one in-browser DuckDB and one selection for every widget, so the bar at the top filters every episode
view; our own runs have filters of their own. The data is inline; the scripts come from cdn.jsdelivr.net.
"""

import html
import os

from inspect_viz import Data, Selection
from inspect_viz.input import checkbox_group, select
from inspect_viz.mark import dot, line, rule_x, rule_y, title
from inspect_viz.plot import HtmlPage, legend, plot
from inspect_viz.table import column, table
from inspect_viz.view import sample_heatmap, scores_heatmap

from styx import items
from view import tidy

OUT = "view/dist"
CLI = dict(color_domain=["claude", "codex", "opencode"], color_range=["#2a78d6", "#eb6834", "#1baf7a"])  # fixed order, by harness
SEEDS = ["#9a9a96", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#2a78d6", "#1baf7a"]
SIDE = 440
WHO = {"agent": "agent", "episode": "episode", "run": "run"}
EPISODE = ["task", "run", "harness", "model", "agent", "episode", "condition", "gpu", "job", "answered", "rank",
           "said_runtime_s", "said_runtime_s_low", "said_runtime_s_high", "said_request_s", "ours_seconds", "runtime_ln", "runtime_ratio", "margin",
           "said_test_score", "said_test_score_low", "said_test_score_high", "ours_test_score", "score_ln",
           "first_gpu", "first_state", "first_limit_s", "first_seconds", "jobs", "failed", "usd", "usd_ours", "usd_ratio", "correct",
           "own_self_seconds", "own_self_seconds_low", "own_self_seconds_high", "agent_s", "own_self_output_tokens", "output_tokens", "own_self_tool_calls", "tool_calls"]
OURS = ["line", "job", "metric", "gpu", "where", "variant", "seed_label", "budget", "budget_label", "state", "exit", "seconds", "score", "peak_gpu_mb", "pool_job", "truth"]


def episodes(df):
    return Data.from_dataframe(df.reindex(columns=EPISODE).reset_index(drop=True))


def runs(df):
    return Data.from_dataframe(df[OURS].reset_index(drop=True))


def positive(df, *cols):
    """The rows a log axis can draw."""
    return df[(df[list(cols)] > 0).all(axis=1)].reset_index(drop=True)


def span(*series, pad=1.3):
    return [min(s[s > 0].min() for s in series) / pad, max(s.max() for s in series) * pad]


def ticks(domain):
    """1, 1.5, 2, 3, 5 and 7 times the powers of ten inside a log axis's domain: its own ticks leave narrow ones bare."""
    return [m * 10.0**k for k in range(-4, 6) for m in (1, 1.5, 2, 3, 5, 7) if domain[0] <= m * 10.0**k <= domain[1]]


def named(j):
    return title(f"job {j} · {items.JOBS[j][0]}", margin_top=40)


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


def build():
    eps, own = tidy.episodes(), tidy.ours()
    page, filt, mine = HtmlPage(), Selection.crossfilter(), Selection.intersect()
    fc, po = (eps[eps["task"] == t].reset_index(drop=True) for t in ("forecast", "post"))
    ep, fc_all, t = episodes(eps), episodes(fc), runs(own)
    bar = [select(ep, column="task", label="Task", target=filt, filter_by=filt),
           checkbox_group(ep, column="harness", label="CLI", target=filt, filter_by=filt),
           select(ep, column="model", label="Model", target=filt, filter_by=filt, multiple=True),
           checkbox_group(ep, column="condition", label="Condition", target=filt, filter_by=filt),
           checkbox_group(ep, column="gpu", label="GPU", target=filt, filter_by=filt),
           select(ep, column="job", label="Job", target=filt, filter_by=filt, multiple=True),
           select(ep, column="run", label="Run", target=filt, filter_by=filt, multiple=True)]
    sections = []

    a = positive(fc, "said_runtime_s", "ours_seconds")
    xd, yd = span(a["ours_seconds"]), span(a["said_runtime_s"])
    sections.append(("A", "Forecast runtime: said.RUNTIME_S against ours.seconds",
                     f"{len(a)} of {len(fc)} forecast episodes drawn; the other {len(fc) - len(a)} gave no runtime. Dashed: forecast = truth. "
                     "Colour: CLI. Symbol: condition. Truth is one value per job and GPU, so dots stack in columns. Table: every forecast episode.", [],
                     [loglog(diagonal(xd, yd), dots(episodes(a), filt, "ours_seconds", "said_runtime_s", "condition", {"said s": "said_runtime_s", "ours s": "ours_seconds"}),
                             x_label="ours.seconds (truth) →", y_label="↑ said.RUNTIME_S (forecast)", xdom=xd, ydom=yd),
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
                    y_scale="log", y_domain=y, y_ticks=ticks(y), y_tick_format="~g", y_label=f"↑ {items.TASKS[items.JOBS[j][0]]['metric']}", fx_label=None, grid=True, width=SIDE, height=320,
                    legend=[legend("color", label="seed", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **by_seed)

    labels = sorted({l for l in own["budget_label"] if l.isdigit()}, key=int) + sorted({l for l in own["budget_label"] if not l.isdigit()})
    sections.append(("C", "Our runs of the jobs (out/items/items.jsonl): score against epochs per job and GPU, one dot per seed; and seconds",
                     f"{len(own)} rows, {own['score'].notna().sum()} with a score. A score is comparable only within a job (its task's metric), so each job has its own y scale. "
                     "Epochs: a sweep row's budget; a run of the script as published sits at the job's last budget. CPU runs (job a's proxies) are only in the seconds plot and the table. "
                     "These filters act on this section only.",
                     [checkbox_group(t, column=c, label=c.capitalize(), target=mine) for c in ("job", "gpu", "variant", "state")],
                     [curve(j) for j in sorted(gpu_runs["job"].unique())] + [
                      plot(dot(t, x="budget_label", y="seconds", fx="gpu", fy="job", fill="seed_label", symbol="state", r=4, fill_opacity=0.8, filter_by=mine,
                               channels={"line": "line", "variant": "variant", "seed": "seed_label", "epochs": "budget_label", "state": "state", "seconds": "seconds", "pool job": "pool_job"}),
                           y_scale="log", y_label="↑ seconds", x_domain=labels, x_tick_rotate=-40, margin_bottom=70, x_label="epochs (or variant) →", fx_label=None, fy_label=None, grid=True,
                           width=900, height=90 * own["job"].nunique() + 110, legend=[legend("color", label="seed", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **by_seed),
                      rows(t, mine, [num("line", width=60), num("job", width=55), num("metric", width=85), num("gpu", width=65), num("variant", width=85), num("seed_label", "seed", width=65),
                                     num("budget", "epochs", width=75), num("state", width=105), num("exit", width=60), num("seconds", width=80), num("score", fmt=".5~f", width=85),
                                     num("peak_gpu_mb", "peak MB"), num("pool_job", "pool job"), num("truth", width=70)], height=440)]))

    d = positive(fc, "said_test_score")
    d = d.assign(said_test_score_low=d["said_test_score_low"].where(d["said_test_score_low"] > 0), rank=d.groupby(["job", "gpu"])["said_test_score"].rank(method="first"))
    truth = own[own["truth"]]

    def scored(j):
        f, s = d[d["job"] == j], truth[truth["job"] == j]
        data, y = episodes(f), span(f["said_test_score"], f["said_test_score_low"], f["said_test_score_high"], s["score"])
        return plot(rule_y(runs(s), y="score", fx="gpu", stroke="black", stroke_opacity=0.7),
                    rule_x(data, x="rank", y1="said_test_score_low", y2="said_test_score_high", fx="gpu", stroke="harness", stroke_opacity=0.35, filter_by=filt),
                    dots(data, filt, "rank", "said_test_score", "condition", {"said": "said_test_score", "low": "said_test_score_low", "high": "said_test_score_high", "scored against": "ours_test_score"}, fx="gpu"),
                    title=named(j), x_domain=[0, f["rank"].max() + 1], x_ticks=[], x_label="forecasts, by their median →", fx_label=None,
                    y_scale="log", y_domain=y, y_ticks=ticks(y), y_tick_format="~g",
                    y_label=f"↑ said.TEST_SCORE ({items.TASKS[items.JOBS[j][0]]['metric']})", grid=True, width=560, height=380,
                    legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **CLI)

    sections.append(("D", "Forecast score: said.TEST_SCORE (LOW to HIGH) beside the seeds of our runs of that job and GPU",
                     f"{len(d)} of {len(fc)} forecast episodes drawn, ordered by their median within each job and GPU; bar: LOW to HIGH. Black lines: each of our {len(truth)} seeds at a job's full budget "
                     "(the sweep's at its last budget, else the runs as published), what a forecast is now scored against; not filtered. "
                     "The table's 'scored against' is the truth an episode was scored against when it ran.", [],
                     [scored(j) for j in sorted(d["job"].unique())] + [
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
    return render(page, bar, sections, eps, own)


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


def render(page, bar, sections, eps, own):
    inputs = lambda comps, cls: f'<div class="{cls}">' + "".join(f"<div>{page.fragment(c)}</div>" for c in comps) + "</div>"
    parts = [inputs(bar, "bar"), f'<p class="nav">{" · ".join(f"<a href=#{k}>{k}</a>" for k, *_ in sections)} — the filters above act on A, B, D, E and F; C has its own. '
             f'{len(eps)} episodes in {eps["run"].nunique()} runs; {len(own)} runs of ours. Hover a point for its episode; tables sort and filter by column.</p>']
    for key, title, note, ins, comps in sections:
        parts.append(f'<section id="{key}"><h2>{key}. {html.escape(title)}</h2><p class="note">{html.escape(note)}</p>'
                     + (inputs(ins, "filters") if ins else "") + '<div class="row">' + "".join(f"<div>{page.fragment(c)}</div>" for c in comps) + "</div></section>")
    return PAGE.format(head=page.head(), body="\n".join(parts))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>styx episodes</title>
<style>
body {{ margin: 0; padding: 0 16px 40px; background: #fcfcfb; color: #0b0b0b; font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", Arial, sans-serif; }}
.bar, .filters {{ display: flex; flex-wrap: wrap; gap: 4px 28px; align-items: flex-end; padding: 2px 0 8px; }}
.bar {{ position: sticky; top: 0; z-index: 10; background: #fcfcfb; border-bottom: 1px solid #ccc; }}
.bar .mosaic-widget, .filters .mosaic-widget {{ margin: 0 !important; }}
.nav {{ font-size: 13px; color: #52514e; }}
section {{ margin-top: 28px; scroll-margin-top: 110px; }}
h2 {{ font-size: 17px; margin: 0 0 4px; }}
.note {{ color: #52514e; font-size: 13px; margin: 0 0 8px; max-width: 1100px; }}
.row {{ display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }}
</style>
{head}
</head><body>
{body}
</body></html>
"""


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/index.html", "w") as f:
        f.write(build())
    print(f"{OUT}/index.html", os.path.getsize(f"{OUT}/index.html"), "bytes")
