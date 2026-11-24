"""1. An agent reads a published training script and is told which GPU will run it. It may not run it. How many times the
actual seconds does it forecast, and how often do the actual runs fall inside the range it gives?

2. It then requests a time limit, told that a run killed at its limit runs again in full and that the part of a limit a run does
not use costs {HOLD} of the price. How many times its own forecast does it request, and how many times the actual seconds? Does it
request more where its range is wider, where the run is shorter, or where its forecast is further off?

3. For the same script it forecasts the test score. How far off is it, and is an agent that is near on the seconds also near on
the score? Is its forecast the score that was published with the script?

4. Across the nine jobs, does an agent put them in the order of their actual seconds? Between the A10 and the A100, does it say
that the faster one is faster?

5. Given {threads} CPU threads and {MINUTES} minutes, or given those and told to find a proxy, does an agent compute, and do its
forecasts move toward the actual?

6. Asked for the same script at fewer epochs, does an agent's forecast of the score, and of the seconds, change as much as the
actual does? Does that grow with the model's capability?

7. Before it starts, an agent forecasts its own seconds, output tokens and tool calls. How many times the actual are they, and is
an agent that is near on its own run also near on the job's?

8. Handed a workload and a pool of GPUs it reaches only by posting jobs, which GPU does its first job ask for, what limit does it
ask against the seconds the job takes, and what do its jobs cost next to the cheapest posting we know?
"""

import contextlib
import inspect
import io
import re

from inspect_viz import Data, Selection
from inspect_viz.mark import cell, dot, line, link, rule_x, rule_y, text, tick_x, title
from inspect_viz.plot import HtmlPage, legend, plot
from inspect_viz.transform import count, sql

from styx import agents, items, pool
from tasks import forecast, post
from tasks.forecast import MEANS
from view import tidy
from view.build import BAR, BESIDE, CLI, bar, compact, positive, render, span, told

LEFT, COLS, SIDE = 250, 6, 210  # a model's name; panels to a row, and a panel's side
# The rows a plot of the budgets draws, as DataFrame.query reads them.
SMALL, CUT = "not full", "not full and kind == 'cut' and answered"
CONDITION = dict(color_domain=list(MEANS), color_range=["#2a78d6", "#eb6834", "#1baf7a"])
TASK = dict(color_domain=["forecast", "post"], color_range=["#2a78d6", "#eb6834"])
QUESTIONS = [" ".join(q.split()) for q in re.findall(r"^\d\. (.+?)(?:\n\n|\Z)", __doc__, re.M | re.S)]
HALO = dict(fill="#0b0b0b", stroke="#fcfcfb", stroke_width=3, paint_order="stroke")
GPU = lambda c, g: f"median({c}) FILTER (WHERE gpu = '{g}')"  # noqa: E731


def charged(hold):
    """The first line `styx info` prints at a pool's $STYX_HOLD of `hold`."""
    pool.HOLD, was, out = hold, pool.HOLD, io.StringIO()
    with contextlib.redirect_stdout(out):
        pool.info([])
    pool.HOLD = was
    return out.getvalue().splitlines()[0]


def build():
    eps = tidy.episodes()
    b = tidy.budgets(eps)
    page, filt = HtmlPage(), Selection.crossfilter()
    fc, po = (eps[eps["task"] == t].reset_index(drop=True) for t in ("forecast", "post"))
    rows = lambda d, *cols: Data.from_dataframe(compact(d[list(dict.fromkeys(BAR + ["episode", *cols]))]))
    who = {"model": "model", "id": "episode", "run": "run", "job": "job", "gpu": "gpu"}
    named = lambda x: f"{x} = {tidy.DERIVED[x]}" if x in tidy.DERIVED else x
    E = tidy.AGGREGATE

    def strip(d, x, fx="condition", mark="median", scale="log", **axis):
        """Each agent's x by model, a facet per value of fx, with mark(x) of each model as a black tick; on a log axis a line at 1."""
        d = positive(d, x) if scale == "log" else d[d[x].notna()].reset_index(drop=True)
        data, models, facets = rows(d, x), tidy.ordered(d), [v for v in (*MEANS, "forecast", "post") if v in set(d[fx])]
        return (f"{named(x)} · by model and {fx} · black tick = {mark}({x})",
                [plot(*([rule_x(x=[1], stroke="#999")] if scale == "log" else []),
                      dot(data, x=x, y="model", fx=fx, fill="cli", r=3, fill_opacity=0.4, filter_by=filt, channels=who | {x: x}),
                      tick_x(data, x=sql(f"{mark}({x})"), y="model", fx=fx, stroke="black", stroke_width=2, filter_by=filt),
                      x_scale=scale, y_domain=models, fx_domain=facets, x_label=x, y_label=None, fx_label=None, grid=True, margin_left=LEFT, margin_right=20,
                      width=LEFT + 260 * len(facets), height=20 * len(models) + 90, legend=legend("color", frame_anchor="bottom"), **CLI, **axis)],
                tidy.defined(x) + [f"black tick = {mark}({x})"])

    def grid(d, least):
        """The rows with the place of their model's panel, COLS to a row in model order: a panel for each model with `least` rows or more
        and one for the rest; the top of a panel is kept for its label."""
        n = d["model"].value_counts()
        own = [m for m in tidy.ordered(d) if n.get(m, 0) >= least]
        place = {m: own.index(m) if m in own else len(own) for m in n.index}
        k = len(own) + (len(place) > len(own))
        d = d.assign(px=d["model"].map(place) % COLS, py=d["model"].map(place) // COLS)
        return d, dict(fx_domain=list(range(min(COLS, k))), fy_domain=list(range((k + COLS - 1) // COLS)), fx_axis=False, fy_axis=False, fx_label=None, fy_label=None,
                       y_inset_top=30, width=70 + SIDE * min(COLS, k), height=70 + (SIDE + 10) * ((k + COLS - 1) // COLS)), least

    def label(data, r=None):
        """The panel's model (or 'other models'), with AGGREGATE[r] and its agents under it, at its top left."""
        name = "CASE WHEN count(DISTINCT model) = 1 THEN min(model) ELSE 'other models' END"
        return text(data, fx="px", fy="py", frame_anchor="top-left", dx=4, dy=4, filter_by=filt, styles=dict(font_size=10, line_height=1.25), **HALO,
                    text=sql(f"printf('%s\n{r} {'%d of' if E[r].startswith('count') else '%.2f ·'} %d', {name}, {E[r]}, {E['agents']})" if r else f"printf('%s · %d', {name}, {E['agents']})"))

    def versus(d, x, y, colour, least, r=None, bars=None, diagonal=True, scale="log", extra=()):
        """Per model, a dot per agent of y against x, coloured by `colour`, with y = x; with `bars`, a thin bar from its LOW to its HIGH;
        with `r`, AGGREGATE[r] in the panel."""
        d, panels, least = grid(positive(d, x, y) if scale == "log" else d.dropna(subset=[x, y]), least)
        cols = [x, y, colour, "px", "py", *(bars or ()), *extra]
        data, (lo, hi) = rows(d, *cols), span(d[x], d[y])
        domains = dict(x_domain=[lo, hi], y_domain=[lo, hi]) if diagonal else dict(x_domain=span(d[x]), y_domain=span(d[y])) if scale == "log" else {}
        colours = dict(condition=CONDITION, task=TASK, cli=CLI).get(colour, dict(color_scheme="tableau10"))
        return (f"{named(y)} against {named(x)} · per model" + (" · dashed: y = x" if diagonal else "") + (f" · bar: {bars[0]} to {bars[1]}" if bars else "") + (f" · {r} = {E[r]}" if r else ""),
                [plot(*([line(x=[lo, hi], y=[lo, hi], stroke="#777", stroke_dasharray="4 4")] if diagonal else []),
                      *([rule_x(data, x=x, y1=bars[0], y2=bars[1], fx="px", fy="py", stroke=colour, stroke_opacity=0.25, clip=True, filter_by=filt)] if bars else []),
                      dot(data, x=x, y=y, fx="px", fy="py", fill=colour, r=2.5, fill_opacity=0.55, filter_by=filt, channels=who | {x: x, y: y}), label(data, r),
                      x_scale=scale, y_scale=scale, **domains, x_tick_format="~g", y_tick_format="~g", x_label=x, y_label=y, grid=True,
                      legend=legend("color", frame_anchor="bottom"), **panels, **colours)],
                tidy.defined(*([r] if r else []), x, y, *(bars or ()), *([colour] if colour in ("condition", "metric", "cli") else [])) + [f"panels: a model with {least} agents or more, the rest under other models"])

    def order():
        """Per model, the median forecast seconds of each job on each GPU against the actual, on log axes with y = x."""
        d, panels, least = grid(positive(fc, "forecast_runtime_s", "actual_seconds"), 50)
        m = d.groupby(["model", "job", "gpu"])[["forecast_runtime_s", "actual_seconds"]].median()
        data, (lo, hi) = rows(d, "forecast_runtime_s", "actual_seconds", "px", "py"), span(m["forecast_runtime_s"], m["actual_seconds"])
        return ("median(forecast_runtime_s) against median(actual_seconds) of each job on each GPU · per model · dashed: y = x · order_r = " + E["order_r"],
                [plot(line(x=[lo, hi], y=[lo, hi], stroke="#777", stroke_dasharray="4 4"),
                      dot(data, x=sql("median(actual_seconds)"), y=sql("median(forecast_runtime_s)"), fx="px", fy="py", fill="job", symbol="gpu", r=3.5, filter_by=filt),
                      label(data, "order_r"), x_scale="log", y_scale="log", x_domain=[lo, hi], y_domain=[lo, hi], x_tick_format="~g", y_tick_format="~g", x_label="median(actual_seconds)",
                      y_label="median(forecast_runtime_s)", grid=True, color_scheme="tableau10", legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **panels)],
                tidy.defined("order_r", "job", "gpu") + [f"panels: a model with {least} agents or more, the rest under other models"])

    def faster():
        """Per model, each job's seconds on the A100 against those on the A10: the actual in black, the median forecast in its CLI's colour, joined."""
        d = positive(fc, "forecast_runtime_s", "actual_seconds")
        d, panels, least = grid(d[d.groupby(["model", "job"])["gpu"].transform("nunique") > 1], 50)
        data = rows(d, "forecast_runtime_s", "actual_seconds", "px", "py")
        a10, a100, f10, f100 = (sql(GPU(c, g)) for c in ("actual_seconds", "forecast_runtime_s") for g in ("A10", "A100"))
        lo, hi = span(d["actual_seconds"], d.groupby(["model", "job", "gpu"])["forecast_runtime_s"].median())
        return ("seconds on the A100 against seconds on the A10 of each job · per model · black: actual, colour: forecast · dashed: y = x",
                [plot(line(x=[lo, hi], y=[lo, hi], stroke="#777", stroke_dasharray="4 4"),
                      link(data, x1=a10, y1=a100, x2=f10, y2=f100, z="job", fx="px", fy="py", stroke="#aaa", filter_by=filt),
                      dot(data, x=a10, y=a100, z="job", fx="px", fy="py", fill="black", r=2.5, filter_by=filt),
                      text(data, x=a10, y=a100, text="job", fx="px", fy="py", dx=7, filter_by=filt, styles=dict(font_size=9), **HALO),
                      dot(data, x=f10, y=f100, z="job", fx="px", fy="py", fill="cli", r=3, filter_by=filt), label(data),
                      x_scale="log", y_scale="log", x_domain=[lo, hi], y_domain=[lo, hi], x_tick_format="~g", y_tick_format="~g", x_label="seconds on the A10", y_label="seconds on the A100",
                      grid=True, legend=legend("color", frame_anchor="bottom"), **panels, **CLI)],
                [f"black dot = ({GPU('actual_seconds', 'A10')}, {GPU('actual_seconds', 'A100')})", f"coloured dot = ({GPU('forecast_runtime_s', 'A10')}, {GPU('forecast_runtime_s', 'A100')})",
                 *tidy.defined("forecast_runtime_s", "actual_seconds", "job"), f"panels: a model with {least} agents or more, the rest under other models; jobs with agents on both GPUs"])

    def conditions(y, actual, label):
        """Per job and GPU, each model's median y in each condition, joined, beside the median actual as a black line; of the models that ran a job on a GPU in all three."""
        d = positive(fc, y, actual)
        d = d[d.groupby(["model", "job", "gpu"])["condition"].transform("nunique") == len(MEANS)]
        data, models = rows(d, y, actual), tidy.ordered(d)
        return (f"median({y}) against condition, a line per model, per job and gpu · black line: median({actual})",
                [plot(rule_y(data, y=sql(f"median({actual})"), fx="job", fy="gpu", stroke="black", filter_by=filt),
                      line(data, x="condition", y=sql(f"median({y})"), z="model", fx="job", fy="gpu", stroke="model", sort={"channel": "x"}, filter_by=filt),
                      dot(data, x="condition", y=sql(f"median({y})"), fx="job", fy="gpu", fill="model", r=3, filter_by=filt),
                      x_domain=list(MEANS), x_tick_rotate=-45, y_scale="log", y_tick_format="~g", x_label=None, y_label=label, fx_label=None, fy_label=None, grid=True,
                      width=130 + 120 * d["job"].nunique(), height=420, margin_bottom=60, margin_right=60, color_domain=models, color_scheme="tableau10",
                      legend=legend("color", frame_anchor="bottom"))],
                tidy.defined(y, actual, "condition") + [f"rows: a model's agents on a job and GPU it ran in all {len(MEANS)} conditions"])

    def curves(y, actual, label):
        """Per job and GPU, each model's median forecast against epochs, and the median actual as a black line."""
        models, out = tidy.ordered(b), []
        for i, j in enumerate(sorted(b["job"].unique())):
            f = positive(b[b["job"] == j], y)
            data, epochs = rows(f, y, actual, "epochs"), sorted(map(int, f["epochs"].unique()))
            out.append(plot(line(data, x="epochs", y=sql(f"median({actual})"), fx="gpu", stroke="black", sort={"channel": "x"}, filter_by=filt),
                            dot(data, x="epochs", y=sql(f"median({actual})"), fx="gpu", fill="black", r=2.5, filter_by=filt),
                            line(data, x="epochs", y=sql(f"median({y})"), z="model", fx="gpu", stroke="model", stroke_opacity=0.8, sort={"channel": "x"}, filter_by=filt),
                            title=title(f"job {j}", margin_top=36, font_size=12), x_scale="log", x_ticks=epochs, x_tick_format="~g", x_label="epochs",
                            y_scale="log", y_tick_format="~g", y_label=label(j), fx_label=None, x_inset=10, **BESIDE, grid=True, width=300, height=225, color_domain=models, color_scheme="tableau10",
                            name=None if i else f"curves_{y}"))
        return (f"median({y}) of each model and median({actual}) (black) against epochs, per job and gpu", [legend("color", for_plot=f"curves_{y}"), *out],
                tidy.defined(y, actual, "epochs"))

    def valued(query, name, n):
        """Per model and condition, AGGREGATE[name] of the rows of budgets the query keeps as a dot, and as text beside AGGREGATE[n];
        a line at 1. A model with fewer than 3 rows whose every column the two read is positive is left out."""
        d = b.query(query)
        cols = [c for c in dict.fromkeys(re.findall(r"[a-z_][a-z0-9_]*", E[name] + E[n])) if c in d.columns]
        d = positive(d, *cols)
        d = d[d.groupby("model")["model"].transform("size") >= 3]
        models, facets, data = tidy.ordered(d), [c for c in MEANS if c in set(d["condition"])], rows(d, *cols)
        return (f"{name} = {E[name]} · by model and condition · text = {name} · {n}",
                [plot(rule_x(x=[1], stroke="#999"),
                      dot(data, x=sql(E[name]), y="model", fx="condition", fill="cli", r=4, filter_by=filt),
                      text(data, x=sql(E[name]), y="model", fx="condition", text=sql(f"printf('%.2f · %d', {E[name]}, {E[n]})"), dx=8, filter_by=filt,
                           styles=dict(text_anchor="start", font_size=11), **HALO),
                      y_domain=models, fx_domain=facets, x_label=name, x_inset=12, y_label=None, fx_label=None, grid=True, margin_left=LEFT, margin_right=70,
                      width=LEFT + 260 * len(facets), height=20 * len(models) + 90, legend=legend("color", frame_anchor="bottom"), **CLI)],
                tidy.defined(name, n) + [f"text = {name} · {n}", f"rows: budgets.query({query!r})", f"models: {n} >= 3"])

    def gains(q):
        """Per model, each agent's forecast at a smaller budget over its forecast at the full one, against the same of the actual, on log axes with y = x."""
        x, y = f"{q}_gain_actual", f"{q}_gain_forecast"
        d, panels, least = grid(positive(b.query(SMALL), x, y), 30)
        data, (lo, hi) = rows(d, x, y, "kind", "epochs", "condition", "px", "py"), span(d[x], d[y])
        return (f"{named(y)} against {named(x)} · per model · dashed: y = x",
                [plot(line(x=[lo, hi], y=[lo, hi], stroke="#777", stroke_dasharray="4 4"),
                      dot(data, x=x, y=y, fx="px", fy="py", fill="condition", symbol="kind", r=2.5, fill_opacity=0.6, filter_by=filt, channels=who | {"epochs": "epochs", "kind": "kind"}),
                      label(data), x_scale="log", y_scale="log", x_domain=[lo, hi], y_domain=[lo, hi], x_tick_format="~g", y_tick_format="~g", x_label=x, y_label=y, grid=True,
                      symbol_domain=["cut", "flat", "rescaled"], legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **panels, **CONDITION)],
                tidy.defined(x, y, "kind") + [f"rows: budgets.query({SMALL!r})", f"panels: a model with {least} rows or more, the rest under other models"])

    def flat():
        """Per model, its agents' forecast test score of the jobs of items.FLAT at each budget, beside the median actual on each GPU."""
        d, panels, least = grid(positive(b[b["job"].isin(items.FLAT)], "forecast_test_score_e"), 10)
        data = rows(d, "forecast_test_score_e", "actual_test_score_e", "epochs", "condition", "px", "py")
        return (f"forecast_test_score_e and median(actual_test_score_e) (black, a symbol per gpu) against epochs, jobs {sorted(items.FLAT)} · per model",
                [plot(dot(data, x="epochs", y="forecast_test_score_e", fx="px", fy="py", fill="condition", r=2.5, fill_opacity=0.6, filter_by=filt, channels=who),
                      line(data, x="epochs", y=sql("median(actual_test_score_e)"), z="gpu", fx="px", fy="py", stroke="black", sort={"channel": "x"}, filter_by=filt),
                      dot(data, x="epochs", y=sql("median(actual_test_score_e)"), fx="px", fy="py", symbol="gpu", fill="black", r=3.5, filter_by=filt), label(data),
                      x_scale="log", x_ticks=sorted(map(int, d["epochs"].unique())), x_tick_format="~g", y_scale="log", y_tick_format="~g", x_label="epochs",
                      y_label=f"forecast_test_score_e ({items.TASKS[items.JOBS[sorted(items.FLAT)[0]][0]]['metric']})", x_inset=10, grid=True,
                      legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **panels, **CONDITION)],
                tidy.defined("forecast_test_score_e", "actual_test_score_e", "epochs") + [f"panels: a model with {least} agents or more, the rest under other models"])

    def counted(d):
        """Per model, how many agents of the post task posted their first job on each GPU."""
        d = d[d["first_gpu"].notna()]
        models, data = tidy.ordered(d), rows(d, "first_gpu")
        return ("count(*) by model and first_gpu",
                [plot(cell(data, x="first_gpu", y="model", fill=count(), filter_by=filt, inset=0.5),
                      text(data, x="first_gpu", y="model", text=count(), filter_by=filt, fill="#0b0b0b"),
                      y_domain=models, x_label="first_gpu", y_label=None, margin_left=LEFT, width=LEFT + 200, height=20 * len(models) + 70,
                      color_scale="log", color_range=["#e3edfa", "#6f9fe0"])],
                tidy.defined("first_gpu") + ["fill, text = count(*)"])

    threads = int(re.search(r"(\d+) CPU threads", forecast.LOCAL)[1])
    keys, phrases = tidy.prompts(forecast.PROMPT, inspect.getsource(forecast.forecast)), {e: phrase for e, phrase, _ in items.BUDGETS["c"] if phrase}
    metric = lambda j: items.TASKS[items.JOBS[j][0]]["metric"]
    blocks = [
        [told("tasks/forecast.py PROMPT", forecast.PROMPT, open=True),
         versus(fc, "actual_seconds", "forecast_runtime_s", "condition", 50, bars=("forecast_runtime_s_low", "forecast_runtime_s_high")),
         strip(fc, "runtime_in", mark="avg", scale="linear", x_tick_format="~g"), strip(fc, "runtime_ratio", x_tick_format="~g")],
        [told(f"tasks/forecast.py PROMPT REQUEST_S, hold = HOLD = {forecast.HOLD}", keys["REQUEST_S"].format(hold=forecast.HOLD)),
         versus(fc, "actual_seconds", "request_s", "condition", 50), versus(fc, "forecast_runtime_s", "request_s", "condition", 50),
         versus(fc, "forecast_runtime_s_high", "request_s", "condition", 50),
         versus(fc, "width_s", "padding_s", "condition", 50, "width_r", diagonal=False), versus(fc, "forecast_runtime_s", "padding_s", "condition", 50, "size_r", diagonal=False),
         versus(fc, "miss_s", "padding_s", "condition", 50, "miss_r", diagonal=False)],
        [versus(fc, "actual_test_score", "forecast_test_score", "metric", 50), strip(fc, "score_in", mark="avg", scale="linear", x_tick_format="~g"),
         versus(fc, "published_test_score", "forecast_test_score", "metric", 50, "published_near", extra=("published_off",)),
         versus(fc, "runtime_pct", "score_pct", "condition", 50, "near_r", diagonal=False)],
        [order(), faster()],
        [*[told(f'tasks/forecast.py MEANS["{c}"]', text) for c, (text, _) in MEANS.items()], strip(fc, "cpu_s", scale="symlog", x_ticks=[0, 10, 100, 1000], x_tick_format="d"),
         conditions("forecast_runtime_s", "actual_seconds", "seconds"), conditions("forecast_test_score", "actual_test_score", "test score")],
        [told("tasks/forecast.py forecast() lines", "\n".join(v for k, v in keys.items() if "{e}" in k)),
         told('styx/items.py BUDGETS["c"]', "\n".join(f"{e}: {phrase}" for e, phrase in phrases.items())),
         curves("forecast_test_score_e", "actual_test_score_e", metric), curves("forecast_runtime_s_e", "actual_seconds_e", lambda j: "seconds"),
         valued(CUT, "score_slope", "score_points"), valued(CUT, "seconds_slope", "seconds_points"), flat(), gains("score"), gains("seconds")],
        [told("styx/agents.py BEFORE", agents.BEFORE), versus(eps, "agent_s", "forecast_self_seconds", "task", 60), versus(eps, "output_tokens", "forecast_self_output_tokens", "task", 60),
         versus(eps, "tool_calls", "forecast_self_tool_calls", "task", 60), versus(fc, "runtime_pct", "self_seconds_pct", "condition", 50, "self_r", diagonal=False),
         versus(fc, "output_tokens", "runtime_pct", "condition", 50, "tokens_r", diagonal=False)],
        [told("tasks/post.py PROMPT", post.PROMPT), told("styx/pool.py info() at tasks/post.py HOLDS", "\n".join(f"{h}: {charged(h)}" for h in post.HOLDS)),
         versus(po, "first_seconds", "first_limit_s", "cli", 20), versus(po, "usd_reference", "usd", "cli", 20), counted(po)],
    ]
    sections = [(str(i), q.format(HOLD=forecast.HOLD, MINUTES=forecast.MINUTES, threads=threads), "", [], bs) for i, (q, bs) in enumerate(zip(QUESTIONS, blocks), 1)]
    return render(page, bar(rows(eps), filt), sections, "styx questions", '<a href="index.html">every agent</a>')
