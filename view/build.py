"""Build Results (index.html) and Analysis (analysis.html) from one frozen input snapshot.

python -m view.build
python -m http.server -d view/dist

--root reads published episode directories or table/episodes.jsonl. --items and
--reference select reference JSONL files; --snapshot supplies HF revision metadata.
No evaluations, jobs, or network requests are made by this builder.
"""

import argparse
import ast
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from plotly.offline import get_plotlyjs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def flat(value, prefix=""):
    out = {}
    for k, v in value.items():
        key = prefix + k
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif not isinstance(v, list):
            out[key] = v
    return out


def load_lines(path, sources, warnings):
    if not path.exists():
        warnings.append(f"Not available: {path.name}")
        return []
    blob = path.read_bytes()
    sources.append(dict(file=str(path), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)))
    rows = []
    for n, line in enumerate(blob.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append((n, clean(json.loads(line))))
        except (ValueError, UnicodeDecodeError):
            warnings.append(f"{path.name}, line {n}: invalid or incomplete JSON; not included")
    return rows


def catalogue():
    """Read descriptive literals without importing task runners or their dependencies."""
    tree = ast.parse((REPO / "styx/items.py").read_text())
    names, jobs, metrics = {}, {}, {}
    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        try:
            value = ast.literal_eval(n.value)
            for target in n.targets:
                if isinstance(target, ast.Tuple):
                    names.update({t.id: v for t, v in zip(target.elts, value)})
                elif isinstance(target, ast.Name):
                    names[target.id] = value
        except (ValueError, TypeError):
            pass
        if any(isinstance(t, ast.Name) and t.id == "TASKS" for t in n.targets):
            for key, value in zip(n.value.keys, n.value.values):
                metrics[key.id] = next(ast.literal_eval(k.value) for k in value.keywords if k.arg == "metric")
        if any(isinstance(t, ast.Name) and t.id == "JOBS" for t in n.targets):
            for k in n.value.keywords:
                task, script = k.value.elts
                jobs[k.arg] = dict(task=names.get(task.id, task.id), script=ast.literal_eval(script), metric=metrics.get(task.id))
    return jobs


def snapshot(args):
    sources, warnings, episodes, jobs = [], [], {}, []
    root = args.root
    meta = json.loads(args.snapshot.read_text()) if args.snapshot else {}
    revision = meta.get("sha", "main")
    hub = f"https://huggingface.co/datasets/hal2k/styx-data/blob/{revision}/"
    # Prefer original records; use the flattened table only for episodes absent locally.
    paths = sorted(root.glob("*/*/*/*/episode.json"))
    records = []
    for p in paths:
        blob = p.read_bytes()
        sources.append(dict(file=str(p.relative_to(root)), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)))
        try:
            value = clean(json.loads(blob))
        except ValueError:
            warnings.append(f"{p.relative_to(root)}: incomplete episode record")
            continue
        task, run, agent, episode = p.parts[-5:-1]
        records.append((dict(task=task, run=run, agent=agent, episode=episode), value, p.parent))
    table = root / "table/episodes.jsonl"
    if table.exists():
        for _, row in load_lines(table, sources, warnings):
            identity = {k: row[k] for k in ("task", "run", "agent", "episode")}
            records.append((identity, row, None))
    for identity, raw, directory in records:
        key = "/".join(identity.values())
        if key in episodes:
            # The table can add timing measurements absent from the episode JSON.
            supplemental = flat(raw)
            episodes[key]["fields"] = {**supplemental, **episodes[key]["fields"]}
            episodes[key]["table_record"] = raw
            continue
        f = flat(raw)
        task, run, agent, episode = identity.values()
        cell = episode.rsplit("-", 1)[0]
        parts = cell.split("-")
        condition, gpu, workload = (parts[0], parts[1], parts[2] if len(parts) > 2 else "a") if task == "forecast" else ("n/a", "n/a", str(raw.get("cell", cell)).split("@")[0])
        # Canonical aliases are for display only; the original record stays intact.
        for suffix in ("", "_LOW", "_HIGH"):
            if f"said.TEST_SCORE{suffix}" not in f:
                f[f"said.TEST_SCORE{suffix}"] = f.get(f"said.TEST_RMSE{suffix}")
        if "ours.test_score" not in f:
            f["ours.test_score"] = f.get("ours.test_rmse")
        prompt = (directory / "prompt.txt").read_text() if directory and (directory / "prompt.txt").exists() else None
        status = "timed out" if raw.get("late") else "agent error" if raw.get("exit") not in (None, 0) else "incomplete answer" if raw.get("answered") is False else "recorded"
        e = dict(identity, id=key, condition=condition, gpu=gpu, workload=workload,
                 harness=agent.partition("-")[0], model=agent.partition("-")[2] or "not recorded",
                 status=status, fields=f, raw=raw, prompt=prompt, url=hub+key+"/episode.json",
                 transcript=hub+key+"/transcript.jsonl", prompt_url=hub+key+"/prompt.txt")
        episodes[key] = e
        posted = raw.get("posted")
        if isinstance(posted, list):
            for i, job in enumerate(posted):
                jobs.append(dict(id=f"{key}/job/{i+1}", episode_id=key, agent=agent, run=run, workload=workload,
                                 condition=condition, gpu=job.get("gpu", "not recorded"), order=i+1,
                                 fields=flat(job), raw=job, complete=True))
        elif any(k.startswith("first.") for k in f):
            job = {k[6:]: v for k, v in f.items() if k.startswith("first.")}
            jobs.append(dict(id=key+"/job/1", episode_id=key, agent=agent, run=run, workload=workload,
                             condition=condition, gpu=job.get("gpu", "not recorded"), order=1, fields=job, raw=job, complete=False))
    refs = []
    for kind, path in (("training", args.items), ("inference", args.reference)):
        for line, raw in load_lines(path, sources, warnings):
            refs.append(dict(id=f"{kind}/{line}", agent="reference", run="reference", condition="reference",
                             workload=raw.get("script", "a") if kind == "training" else raw.get("cell", "unknown"),
                             gpu=raw.get("gpu") or raw.get("where", "unknown"), kind=kind,
                             fields=flat(raw), raw=raw))
    if not episodes:
        raise SystemExit("No episode records found. Supply --root pointing to a downloaded dataset or the collection machine's cache.")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return clean(dict(episodes=list(episodes.values()), jobs=jobs, references=refs, workloads=catalogue(),
                     meta=dict(built=datetime.now(timezone.utc).isoformat(), code=commit,
                               view_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "build.py", HERE / "app.js", HERE / "style.css")},
                               revision=meta.get("sha"), published=meta.get("lastModified"), sources=sources, warnings=warnings)))


def build(args):
    data = snapshot(args)
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    css, js = (HERE / "style.css").read_text(), (HERE / "app.js").read_text()
    args.out.mkdir(parents=True, exist_ok=True)
    for mode, name in (("results", "index.html"), ("analysis", "analysis.html")):
        title = mode.capitalize()
        document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Styx · {title}</title><style>{css}</style><script>{get_plotlyjs()}</script></head>
<body data-page="{mode}"><header><a class="brand" href="index.html">Styx</a><nav aria-label="Pages"><a href="index.html" {'aria-current="page"' if mode == 'results' else ''}>Results</a><a href="analysis.html" {'aria-current="page"' if mode == 'analysis' else ''}>Analysis</a></nav><button id="download">Download snapshot</button></header>
<main><div id="intro"></div><div id="filters" aria-label="Filters"></div><p id="counts" aria-live="polite"></p><div id="content"></div></main>
<dialog id="detail"><button id="close">Close</button><div id="detail-content"></div></dialog>
<script id="snapshot" type="application/json">{payload}</script><script>{js}</script></body></html>'''
        (args.out / name).write_text(document)
        print(f"{args.out / name}: {len(data['episodes'])} recorded episodes, {len(data['jobs'])} job records, {len(data['references'])} reference executions")
    (args.out / "snapshot.json").write_text(json.dumps(data, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".cache/styx")
    parser.add_argument("--items", type=Path, default=REPO / "out/items/items.jsonl")
    parser.add_argument("--reference", type=Path, default=REPO / "out/reference.jsonl")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "dist")
    build(parser.parse_args())
