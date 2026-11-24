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
from inspect_viz.mark import cell, dot, line, regression_y, rule_x, rule_y, text, tick_x, title
from inspect_viz.plot import HtmlPage, legend, plot
from inspect_viz.transform import count, sql

from styx import agents, items, pool
from tasks import forecast, post
from tasks.forecast import MEANS
from view import tidy
from view.build import BAR, BESIDE, CLI, bar, compact, fits, positive, render, span, told

LEFT, COLS = 250, 4  # a model's name; panels to a row
# The rows a plot of the budgets draws, and those of the level of a ratio, as DataFrame.query reads them.
SMALL, CUT, LEVEL = "not full", "not full and kind == 'cut' and answered", "task == 'forecast' and answered and {} > 0"
CONDITION = dict(color_domain=list(MEANS), color_range=["#2a78d6", "#eb6834", "#1baf7a"])
QUESTIONS = [" ".join(q.split()) for q in re.findall(r"^\d\. (.+?)(?:\n\n|\Z)", __doc__, re.M | re.S)]
HALO = dict(fill="#0b0b0b", stroke="#fcfcfb", stroke_width=3, paint_order="stroke")


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
    reads = lambda expr, d: [c for c in dict.fromkeys(re.findall(r"[a-z_][a-z0-9_]*", expr)) if c in d.columns]  # the columns of d an expression reads
    E = tidy.AGGREGATE

    def strip(d, x, fx="condition", mark="median", scale="log", near=None, **axis):
        """Each agent's x by model, a facet per value of fx, with mark(x) of each model as a black tick; on a log axis a line at 1; with
        `near`, AGGREGATE[near] of agents beside each model."""
        d = positive(d, x) if scale == "log" else d[d[x].notna()].reset_index(drop=True)
        data, models, facets = rows(d, x, *(reads(E[near], d) if near else [])), tidy.ordered(d), [v for v in (*MEANS, "forecast", "post") if v in set(d[fx])] if fx else [None]
        on = dict(fx=fx) if fx else {}
        marks = [dot(data, x=x, y="model", fill="cli", r=3, fill_opacity=0.4, filter_by=filt, channels=who | {x: x}, **on),
                 tick_x(data, x=sql(f"{mark}({x})"), y="model", stroke="black", stroke_width=2, filter_by=filt, **on)]
        if near:
            marks.append(text(data, y="model", text=sql(f"printf('%d of %d', {E[near]}, {E['agents']})"), frame_anchor="right", dx=8, filter_by=filt, styles=dict(text_anchor="start"), **on, **HALO))
        return (f"{named(x)} · by model{' and ' + fx if fx else ''} · black tick = {mark}({x})" + (f" · text = {near} of agents" if near else ""),
                [plot(*([rule_x(x=[1], stroke="#999")] if scale == "log" else []), *marks,
                      x_scale=scale, y_domain=models, x_label=x, y_label=None, fx_label=None, **(dict(fx_domain=facets) if fx else {}),
                      grid=True, margin_left=LEFT, margin_right=70 if near else 20, width=LEFT + 260 * len(facets) + (50 if near else 0), height=20 * len(models) + 90,
                      legend=legend("color", frame_anchor="bottom"), **CLI, **{"x_tick_format": "~g", **axis})],
                tidy.defined(x, *([near] if near else [])) + [f"black tick = {mark}({x})"])

    def grid(d, least):
        """The rows of the models with at least `least` of them, each with the place of its model's panel, COLS to a row, in model order;
        the top of a panel is kept for its label."""
        n = d["model"].value_counts()
        place = {m: i for i, m in enumerate(m for m in tidy.ordered(d) if n.get(m, 0) >= least)}
        d = d[d["model"].isin(place)].assign(px=lambda f: f["model"].map(place).astype(int) % COLS, py=lambda f: f["model"].map(place).astype(int) // COLS)
        return d, dict(fx_domain=list(range(min(COLS, len(place)))), fy_domain=list(range((len(place) + COLS - 1) // COLS)), fx_axis=False, fy_axis=False,
                       fx_label=None, fy_label=None, y_inset_top=30, width=60 + 240 * min(COLS, len(place)), height=60 + 220 * ((len(place) + COLS - 1) // COLS))

    def label(data, r):
        """The panel's model, and under it AGGREGATE[r] and its agents, at its top left."""
        return text(data, fx="px", fy="py", frame_anchor="top-left", dx=4, dy=4, text=sql(f"printf('%s\n{r} %.2f · %d', min(model), {E[r]}, {E['agents']})"), filter_by=filt,
                    styles=dict(font_size=10, line_height=1.25), **HALO)

    def relation(d, x, y, r):
        """Per model, a dot per agent of y against x, its least squares line and AGGREGATE[r]."""
        d, panels = grid(d.dropna(subset=[x, y]), 3)
        data = rows(d, x, y, "px", "py")
        return (f"{named(y)} against {named(x)} · per model · {r} = {E[r]}",
                [plot(dot(data, x=x, y=y, fx="px", fy="py", fill="cli", r=2.5, fill_opacity=0.5, filter_by=filt, channels=who),
                      regression_y(data, x=x, y=y, fx="px", fy="py", stroke="black", ci=0, filter_by=filt), label(data, r),
                      x_label=x, y_label=y, x_inset=8, y_inset=6, grid=True, legend=legend("color", frame_anchor="bottom"), **panels, **CLI)],
                tidy.defined(r) + [f"black line = regression_y(x={x}, y={y})", "panels: models with 3 agents or more"])

    def order():
        """Per model, the median forecast seconds of each job on each GPU against the actual, on log axes with y = x."""
        d = positive(fc, "forecast_runtime_s", "actual_seconds")
        cells = d.drop_duplicates(["model", "job", "gpu"])["model"].value_counts()
        d, panels = grid(d[d["model"].map(cells) >= 3], 1)
        m = d.groupby(["model", "job", "gpu"])[["forecast_runtime_s", "actual_seconds"]].median()
        data, (lo, hi) = rows(d, "forecast_runtime_s", "actual_seconds", "ln_forecast_runtime_s", "ln_actual_seconds", "px", "py"), span(m["forecast_runtime_s"], m["actual_seconds"])
        return ("median(forecast_runtime_s) against median(actual_seconds) of each job on each GPU · per model · order_r = " + E["order_r"],
                [plot(line(x=[lo, hi], y=[lo, hi], stroke="#777", stroke_dasharray="4 4"),
                      dot(data, x=sql("median(actual_seconds)"), y=sql("median(forecast_runtime_s)"), fx="px", fy="py", fill="job", symbol="gpu", r=3.5, filter_by=filt),
                      label(data, "order_r"), x_scale="log", y_scale="log", x_domain=[lo, hi], y_domain=[lo, hi], x_tick_format="~g", y_tick_format="~g", x_label="median(actual_seconds)",
                      y_label="median(forecast_runtime_s)", grid=True, color_scheme="tableau10", legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **panels)],
                tidy.defined("order_r", "job", "gpu") + ["dashed: y = x", "panels: models with 3 jobs on a GPU or more"])

    def faster():
        """Per model, each job's median forecast seconds on the A100 over those on the A10 against the actual's, lines at 1."""
        d = positive(fc, "forecast_runtime_s", "actual_seconds")
        d = d[d.groupby(["model", "job"])["gpu"].transform("nunique") > 1]
        d, panels = grid(d[d.groupby("model")["job"].transform("nunique") >= 3], 1)
        data = rows(d, "forecast_runtime_s", "actual_seconds", "px", "py")
        return (f"gpu_ratio_forecast against gpu_ratio_actual of each job · per model",
                [plot(rule_x(x=[1], stroke="#999"), rule_y(y=[1], stroke="#999"),
                      dot(data, x=sql(E["gpu_ratio_actual"]), y=sql(E["gpu_ratio_forecast"]), fx="px", fy="py", fill="job", r=4, filter_by=filt),
                      text(data, fx="px", fy="py", frame_anchor="top-left", dx=4, dy=4, text=sql("min(model)"), filter_by=filt, styles=dict(font_size=10, line_height=1.25), **HALO),
                      x_label="gpu_ratio_actual", y_label="gpu_ratio_forecast", x_inset=10, y_inset=10, grid=True, color_scheme="tableau10", legend=legend("color", frame_anchor="bottom"), **panels)],
                tidy.defined("gpu_ratio_forecast", "gpu_ratio_actual") + ["panels: models with agents on both GPUs of 3 jobs or more"])

    def valued(frame, query, name, n, least=1):
        """Per model and condition, AGGREGATE[name] of the rows of the frame the query keeps as a dot, and as text beside AGGREGATE[n];
        a line at 1. A model with fewer than `least` rows that have every column the two read is left out."""
        d = (b if frame == "budgets" else eps).query(query)
        cols = reads(E[name] + E[n], d)
        d = d.dropna(subset=cols)
        d = d[d.groupby("model")["model"].transform("size") >= least]
        models, facets, data = tidy.ordered(d), [c for c in MEANS if c in set(d["condition"])], rows(d, *cols)
        return (f"{name} = {E[name]} · by model and condition · text = {name} · {n}",
                [plot(rule_x(x=[1], stroke="#999"),
                      dot(data, x=sql(E[name]), y="model", fx="condition", fill="cli", r=4, filter_by=filt),
                      text(data, x=sql(E[name]), y="model", fx="condition", text=sql(f"printf('%.2f · %d', {E[name]}, {E[n]})"), dx=8, filter_by=filt,
                           styles=dict(text_anchor="start", font_size=11), **HALO),
                      y_domain=models, fx_domain=facets, x_label=name, x_inset=12, y_label=None, fx_label=None, grid=True, margin_left=LEFT, margin_right=70,
                      width=LEFT + 260 * len(facets), height=20 * len(models) + 90, legend=legend("color", frame_anchor="bottom"), **CLI)],
                tidy.defined(name, n) + [f"text = {name} · {n}", f"rows: {frame}.query({query!r})"] + ([f"models: {n} >= {least}"] if least > 1 else []))

    def across(ratio):
        """Per model, the median ratio of each job on each GPU in each condition, joined across them, of the jobs on a GPU it ran in more than one."""
        d, out = positive(fc, ratio), []
        d = d[d.groupby(["model", "gpu", "job"])["condition"].transform("nunique") > 1]
        dom, jobs = span(d.groupby(["model", "condition", "gpu", "job"])[ratio].median()), sorted(d["job"].unique())
        for m in tidy.ordered(d):
            f = d[d["model"] == m]
            data = rows(f, ratio)
            out.append(plot(rule_y(y=[1], stroke="#999"),
                            line(data, x="condition", y=sql(f"median({ratio})"), z="job", fx="gpu", stroke="job", sort={"channel": "x"}, filter_by=filt),
                            dot(data, x="condition", y=sql(f"median({ratio})"), fx="gpu", fill="job", r=3, filter_by=filt),
                            title=title(m, margin_top=40), x_domain=list(MEANS), fx_domain=sorted(f["gpu"].unique()), y_scale="log", y_domain=dom, y_tick_format="~g", y_label=f"median({ratio})",
                            x_label=None, fx_label=None, **BESIDE, grid=True, width=max(360, fits(m, 70, 20)), height=300, color_domain=jobs, color_scheme="tableau10",
                            legend=legend("color", frame_anchor="bottom")))
        return (f"median({ratio}) of each job on each GPU, by condition, joined · per model · {named(ratio)}", out,
                tidy.defined(ratio) + [f"line, dot = median({ratio}) of a job on a GPU in a condition", "rows: of a model, GPU and job with condition.nunique() > 1"])

    def gains(q):
        """Per model, each agent's ln(forecast at a smaller budget / at full) against the actual's, y = x, and the least squares line over CUT."""
        x, y = f"{q}_gain_actual", f"{q}_gain_forecast"
        d = b.query(SMALL).dropna(subset=[x, y])
        lo, hi, fitted, out = min(d[x].min(), d[y].min()) - 0.1, max(d[x].max(), d[y].max()) + 0.1, d.query(CUT)["model"].value_counts(), []
        for m in [m for m in tidy.ordered(d) if fitted.get(m, 0) >= 3]:
            f = d[d["model"] == m]
            out.append(plot(line(x=[lo, hi], y=[lo, hi], stroke="#777", stroke_dasharray="4 4"),
                            dot(rows(f, x, y, "kind", "epochs"), x=x, y=y, fill="condition", symbol="kind", r=3.5, fill_opacity=0.6, filter_by=filt,
                                channels=who | {"epochs": "epochs", "kind": "kind"}),
                            regression_y(rows(f.query(CUT), x, y), x=x, y=y, stroke="black", ci=0, filter_by=filt),
                            title=title(m, margin_top=40), x_domain=[lo, hi], y_domain=[lo, hi], x_label=x, y_label=y, **BESIDE, grid=True, width=max(340, fits(m, 70, 20)), height=320,
                            symbol_domain=["cut", "flat", "rescaled"], legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **CONDITION))
        return (f"{named(y)} against {named(x)} · per model · black line = least squares", out,
                tidy.defined(x, y, "kind", f"{q}_points") + [f"dots: budgets.query({SMALL!r})", "dashed: y = x",
                                                            f"black line = regression_y(x={x}, y={y}) over budgets.query({CUT!r})", f"models: {q}_points >= 3"])

    def flat():
        """Per model, job c's forecast gain in score at each smaller budget, beside the median of the actual's on each GPU."""
        d, panels = grid(b.query(f"job in {sorted(items.FLAT)} and {SMALL}").dropna(subset=["score_gain_forecast"]), 3)
        data = rows(d, "score_gain_forecast", "score_gain_actual", "epochs", "px", "py")
        return ("score_gain_forecast and median(score_gain_actual) against epochs, jobs of items.FLAT · per model",
                [plot(rule_y(y=[0], stroke="#999"),
                      dot(data, x="epochs", y="score_gain_forecast", fx="px", fy="py", fill="condition", r=3, fill_opacity=0.6, filter_by=filt, channels=who),
                      dot(data, x="epochs", y=sql("median(score_gain_actual)"), fx="px", fy="py", symbol="gpu", fill="black", r=4, filter_by=filt),
                      text(data, fx="px", fy="py", frame_anchor="top-left", dx=4, dy=4, text=sql("min(model)"), filter_by=filt, styles=dict(font_size=10, line_height=1.25), **HALO),
                      x_scale="log", x_ticks=sorted(map(int, d["epochs"].unique())), x_tick_format="~g", x_label="epochs", y_label="score_gain", x_inset=12, y_inset=8, grid=True,
                      legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **panels, **CONDITION)],
                tidy.defined("score_gain_forecast", "score_gain_actual", "epochs") + ["black = median(score_gain_actual) on a GPU", f"rows: budgets.query(\"job in {sorted(items.FLAT)} and {SMALL}\")"])

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
    blocks = [
        [told("tasks/forecast.py PROMPT", forecast.PROMPT, open=True), strip(fc, "runtime_ratio"), strip(fc, "runtime_in", mark="avg", scale="linear")],
        [told(f"tasks/forecast.py PROMPT REQUEST_S, hold = HOLD = {forecast.HOLD}", keys["REQUEST_S"].format(hold=forecast.HOLD)), strip(fc, "margin"), strip(fc, "request_ratio"), relation(fc, "ln_runtime_width", "ln_margin", "width_r"), relation(fc, "ln_forecast_runtime_s", "ln_margin", "size_r"),
         relation(fc, "runtime_off", "ln_margin", "miss_r"), strip(fc, "request_high")],
        [strip(fc, "score_ratio"), strip(fc, "score_in", mark="avg", scale="linear"), relation(fc, "runtime_off", "score_off", "score_r"), strip(fc, "published_ratio", None, near="published_near")],
        [order(), faster()],
        [*[told(f'tasks/forecast.py MEANS["{c}"]', text) for c, (text, _) in MEANS.items()], strip(fc, "cpu_s", scale="symlog", x_ticks=[0, 10, 100, 1000], x_tick_format="d"), across("runtime_ratio"), across("score_ratio")],
        [told("tasks/forecast.py forecast() lines", "\n".join(v for k, v in keys.items() if "{e}" in k)),
         told('styx/items.py BUDGETS["c"]', "\n".join(f"{e}: {phrase}" for e, phrase in phrases.items())), gains("score"), valued("budgets", CUT, "score_slope", "score_points", 3), gains("seconds"), valued("budgets", CUT, "seconds_slope", "seconds_points", 3), flat(),
         *[valued("agents", LEVEL.format(r), name, "agents") for r, name in (("runtime_ratio", "runtime_miss"), ("score_ratio", "score_miss"), ("margin", "margin_mean"))]],
        [told("styx/agents.py BEFORE", agents.BEFORE), strip(eps, "self_seconds_ratio", "task"), strip(eps, "self_tokens_ratio", "task"), strip(eps, "self_calls_ratio", "task"),
         relation(fc, "runtime_off", "self_seconds_off", "self_r"), relation(fc, "ln_output_tokens", "runtime_off", "tokens_r")],
        [told("tasks/post.py PROMPT", post.PROMPT), told("styx/pool.py info() at tasks/post.py HOLDS", "\n".join(f"{h}: {charged(h)}" for h in post.HOLDS)),
         strip(po, "usd_ratio", None), counted(po), strip(po, "first_limit_ratio", None)],
    ]
    sections = [(str(i), q.format(HOLD=forecast.HOLD, MINUTES=forecast.MINUTES, threads=threads), "", [], bs) for i, (q, bs) in enumerate(zip(QUESTIONS, blocks), 1)]
    return render(page, bar(rows(eps), filt), sections, "styx questions", '<a href="index.html">every agent</a>')
