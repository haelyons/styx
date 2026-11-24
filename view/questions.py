"""don't know how long to run? don't know how long to think? can't (or don't?) pick reliable proxies? IS THERE A SCALING LAW?
what about when presented with GPU infrastructure?

python -m view.build writes it beside index.html as questions.html: every plot by model, under it what the repo says it draws.
"""

import re

from inspect_viz import Data, Selection
from inspect_viz.mark import cell, dot, line, regression_y, rule_x, rule_y, text, tick_x, title
from inspect_viz.plot import HtmlPage, legend, plot
from inspect_viz.transform import count, sql

from tasks.forecast import MEANS
from view import tidy
from view.build import BAR, BESIDE, CLI, bar, compact, fits, positive, render, span

LEFT = 250  # a model's name
# The rows each scaling plot draws, as DataFrame.query reads them: of budgets, then of episodes for the level of each ratio.
SMALL, CUT, LEVEL = "not full", "not full and kind == 'cut' and answered", "task == 'forecast' and answered and {} > 0"
CONDITION = dict(color_domain=list(MEANS), color_range=["#2a78d6", "#eb6834", "#1baf7a"])


def build():
    eps = tidy.episodes()
    b = tidy.budgets(eps)
    page, filt = HtmlPage(), Selection.crossfilter()
    fc, po = (eps[eps["task"] == t].reset_index(drop=True) for t in ("forecast", "post"))
    rows = lambda d, *cols: Data.from_dataframe(compact(d[list(dict.fromkeys(BAR + ["agent", "episode", *cols]))]))
    who = {"agent": "agent", "episode": "episode", "run": "run", "job": "job", "gpu": "gpu"}

    def strip(d, x, fx="condition", mark="median", scale="log", **axis):
        """Each episode's x by model, a facet per value of fx, with mark(x) of each model as a black tick; on a log axis a line at 1."""
        d = positive(d, x) if scale == "log" else d[d[x].notna()].reset_index(drop=True)
        data, models, facets = rows(d, x), tidy.ordered(d), [v for v in (*MEANS, "forecast", "post") if v in set(d[fx])] if fx else [None]
        on = dict(fx=fx) if fx else {}
        return (plot(*([rule_x(x=[1], stroke="#999")] if scale == "log" else []),
                     dot(data, x=x, y="model", fill="harness", r=3, fill_opacity=0.4, filter_by=filt, channels=who | {x: x}, **on),
                     tick_x(data, x=sql(f"{mark}({x})"), y="model", stroke="black", stroke_width=2, filter_by=filt, **on),
                     x_scale=scale, y_domain=models, x_label=x, y_label=None, fx_label=None, **(dict(fx_domain=facets) if fx else {}),
                     grid=True, margin_left=LEFT, margin_right=20, width=LEFT + 260 * len(facets), height=20 * len(models) + 90,
                     legend=legend("color", frame_anchor="bottom"), **CLI, **{"x_tick_format": "~g", **axis}),
                tidy.defined(x) + [f"black tick = {mark}({x})"])

    def valued(frame, query, name, n, least=1):
        """Per model and condition, AGGREGATE[name] of the rows of the frame the query keeps as a dot, and as text beside AGGREGATE[n];
        a line at 1. A model with fewer than `least` rows that have every column the two read is left out."""
        e, d = tidy.AGGREGATE, (b if frame == "budgets" else eps).query(query)
        cols = [c for c in re.findall(r"[a-z_][a-z0-9_]*", e[name] + e[n]) if c in d.columns]
        d = d.dropna(subset=cols)
        d = d[d.groupby("model")["model"].transform("size") >= least]
        models, facets, data = tidy.ordered(d), [c for c in MEANS if c in set(d["condition"])], rows(d, *cols)
        return (plot(rule_x(x=[1], stroke="#999"),
                     dot(data, x=sql(e[name]), y="model", fx="condition", fill="harness", r=4, filter_by=filt),
                     text(data, x=sql(e[name]), y="model", fx="condition", text=sql(f"printf('%.2f · %d', {e[name]}, {e[n]})"), dx=8, filter_by=filt,
                          fill="#0b0b0b", stroke="#fcfcfb", stroke_width=3, paint_order="stroke", styles=dict(text_anchor="start", font_size=11)),
                     y_domain=models, fx_domain=facets, x_label=name, x_inset=12, y_label=None, fx_label=None, grid=True, margin_left=LEFT, margin_right=70,
                     width=LEFT + 260 * len(facets), height=20 * len(models) + 90, legend=legend("color", frame_anchor="bottom"), **CLI),
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
        return out, tidy.defined(ratio) + [f"line, dot = median({ratio}) of a job on a GPU in a condition", "rows: of a model, GPU and job with condition.nunique() > 1"]

    def gains(q):
        """Per model, each episode's ln(said at a smaller budget / at full) against ours, y = x, and the least squares line over CUT."""
        x, y = f"{q}_gain_ours", f"{q}_gain_said"
        d = b.query(SMALL).dropna(subset=[x, y])
        lo, hi, fitted, out = min(d[x].min(), d[y].min()) - 0.1, max(d[x].max(), d[y].max()) + 0.1, d.query(CUT)["model"].value_counts(), []
        for m in [m for m in tidy.ordered(d) if fitted.get(m, 0) >= 3]:
            f = d[d["model"] == m]
            c = f.query(CUT)
            out.append(plot(line(x=[lo, hi], y=[lo, hi], stroke="#777", stroke_dasharray="4 4"),
                            dot(rows(f, x, y, "kind", "epochs"), x=x, y=y, fill="condition", symbol="kind", r=3.5, fill_opacity=0.6, filter_by=filt,
                                channels=who | {"epochs": "epochs", "kind": "kind"}),
                            regression_y(rows(c, x, y), x=x, y=y, stroke="black", ci=0, filter_by=filt),
                            title=title(m, margin_top=40), x_domain=[lo, hi], y_domain=[lo, hi], x_label=x, y_label=y, **BESIDE, grid=True, width=max(340, fits(m, 70, 20)), height=320,
                            symbol_domain=["cut", "flat", "rescaled"], legend=[legend("color", frame_anchor="bottom"), legend("symbol", frame_anchor="bottom")], **CONDITION))
        return out, tidy.defined(x, y, "kind", f"{q}_points") + [f"dots: budgets.query({SMALL!r})", "dashed: y = x",
                                                                 f"black line = regression_y(x={x}, y={y}) over budgets.query({CUT!r})", f"models: {q}_points >= 3"]

    def counted(d):
        """Per model, how many post episodes posted their first job on each GPU."""
        d, models = d[d["first_gpu"].notna()], tidy.ordered(d[d["first_gpu"].notna()])
        data = rows(d, "first_gpu")
        return (plot(cell(data, x="first_gpu", y="model", fill=count(), filter_by=filt, inset=0.5),
                     text(data, x="first_gpu", y="model", text=count(), filter_by=filt, fill="#0b0b0b"),
                     y_domain=models, x_label="first_gpu", y_label=None, margin_left=LEFT, width=LEFT + 200, height=20 * len(models) + 70,
                     color_scale="log", color_range=["#e3edfa", "#6f9fe0"]),
                tidy.defined("first_gpu") + ["fill, text = count(*)"])

    sections = [
        ("run", "don't know how long to run", "", [],
         [strip(fc, "runtime_ratio"), strip(fc, "request_ratio"), strip(fc, "margin"), strip(fc, "runtime_in", mark="avg", scale="linear")]),
        ("think", "don't know how long to think", "", [],
         [strip(eps, "self_seconds_ratio", "task"), strip(eps, "self_tokens_ratio", "task"), strip(eps, "self_rate_ratio", "task")]),
        ("proxy", "can't (or don't?) pick reliable proxies", "", [],
         [strip(fc, "cpu_s", scale="symlog", x_ticks=[0, 10, 100, 1000], x_tick_format="d"), across("runtime_ratio"), across("score_ratio")]),
        ("scaling", "IS THERE A SCALING LAW?", "", [],
         [gains("score"), valued("budgets", CUT, "score_slope", "score_points", 3), gains("seconds"), valued("budgets", CUT, "seconds_slope", "seconds_points", 3),
          *[valued("episodes", LEVEL.format(r), name, "episodes") for r, name in (("runtime_ratio", "runtime_miss"), ("score_ratio", "score_miss"), ("margin", "margin_mean"))]]),
        ("gpu", "what about when presented with GPU infrastructure?", "", [],
         [strip(po, "usd_ratio", None), counted(po), strip(po, "first_limit_ratio", None)]),
    ]
    return render(page, bar(rows(eps), filt), sections, "styx questions", '<a href="index.html">every episode</a>', keyed=False)
