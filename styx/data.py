"""Results go to the HF dataset as <task>/<run>/. No value of .env leaves this machine: wherever
one occurs in a file, the copy that is uploaded has its name instead.
"""

import fnmatch
import glob
import os
import tempfile
import time

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

DATASET = "hal2k/styx-data"
ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
MAX_BYTES = 50e6
ROOT = os.path.expanduser("~/.cache/styx")
# What of an episode is published: what it said and did, and the text files it wrote.
KEEP = ("episode.json", "ledger.jsonl", "transcript.jsonl", "clock.txt", "stderr.txt", "answer.txt", "before/transcript.jsonl", "before/clock.txt",
        "before/answer.txt", "work/*.py", "work/*.sh", "work/*.txt", "work/*.json", "work/*.log", "work/*.md")


def env():
    rows = (l.strip().split("=", 1) for l in open(ENV) if "=" in l and not l.startswith("#"))
    return {k: v.strip("\"'") for k, v in rows if v.strip("\"'")}


def stage(src, tmp, keep):
    """Copies the files of `src` whose relative path matches one of `keep` into `tmp`, each value of
    .env replaced by its name. Hidden directories, __pycache__ and files over 50 MB stay behind."""
    names = {v.encode(): f"<{k}>".encode() for k, v in env().items()}
    for root, _, files in os.walk(src):
        for rel in (os.path.relpath(f"{root}/{f}", src) for f in files):
            hidden = "/." in f"/{os.path.dirname(rel)}" or "__pycache__" in rel
            if hidden or os.path.getsize(f"{src}/{rel}") > MAX_BYTES or not any(fnmatch.fnmatch(rel, k) for k in keep):
                continue
            b = open(f"{src}/{rel}", "rb").read()
            for value, name in names.items():
                b = b.replace(value, name)
            os.makedirs(os.path.dirname(f"{tmp}/{rel}"), exist_ok=True)
            with open(f"{tmp}/{rel}", "wb") as f:
                f.write(b)


def upload(src, dst, keep):
    """Sends the files of directory `src` matching `keep` to <dataset>/<dst>, in one commit."""
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(src))) as tmp:  # /tmp has a quota here
        stage(src, tmp, keep)
        HfApi(token=env()["HF_W_KEY_ONE"]).upload_folder(folder_path=tmp, path_in_repo=dst, repo_type="dataset", repo_id=DATASET)


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


if __name__ == "__main__":
    for run in glob.glob(f"{ROOT}/post/*-*") + glob.glob(f"{ROOT}/forecast/*-*"):
        print(run, flush=True)
        push(run, 0)
