"""Results go to the HF dataset as <task>/<run>/. No value of .env leaves this machine: wherever
one occurs in a file, the copy that is uploaded has its name instead.
"""

import fnmatch
import os
import tempfile

from huggingface_hub import HfApi

DATASET = "hal2k/styx-data"
ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
MAX_BYTES = 50e6


def env():
    rows = (l.strip().split("=", 1) for l in open(ENV) if "=" in l and not l.startswith("#"))
    return {k: v.strip("\"'") for k, v in rows if v.strip("\"'")}


def upload(src, dst, keep):
    """Sends the files of directory `src` whose relative path matches one of `keep` to <dataset>/<dst>.
    Hidden directories, __pycache__ and files over 50 MB stay behind."""
    names = {v.encode(): f"<{k}>".encode() for k, v in env().items()}
    with tempfile.TemporaryDirectory() as tmp:
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
        HfApi(token=env()["HF_W_KEY_ONE"]).upload_folder(folder_path=tmp, path_in_repo=dst, repo_id=DATASET, repo_type="dataset")
