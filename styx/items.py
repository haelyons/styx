"""Items handed to agents that are built from published sources instead of files of this repo.

denoise: the denoising-dirty-documents solution of run c4121e82... in the HF dataset
zjunlp/PredictBeforeExecute, with the task's public data as MLE-bench prepares it. Needs
~/.cache/styx/data/data/denoising-dirty-documents, that folder of the dataset's tasks/data.zip.
"""

import os
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = os.path.expanduser("~/.cache/styx")
HUB = "zjunlp/PredictBeforeExecute"
SCRIPT = ("solutions_subset_15/denoising-dirty-documents/code/solution_Automind_wo_knowledge_denoising-dirty-documents_"
          "13a296eb-c605-4737-bea2-edd07d49161a_run_c4121e82042c4af2b8755e35432adc9d.py")
DATA = f"{ROOT}/data/data/denoising-dirty-documents/prepared"
# ArchPilot's proxy: the same code for one epoch on a tenth of the training data. The script trains
# at three resolutions, so the epoch is taken at its first and at its last.
MARK = "    # Progressive resizing training\n"
TENTH = [
    (MARK, "    train_files, train_clean_files = train_files[: len(train_files) // 10], train_clean_files[: len(train_files) // 10]\n" + MARK),
    ("epochs_per_resolution = [5, 5, 10]", "epochs_per_resolution = [1]"),
    ("criterion, epochs=10\n", "criterion, epochs=0\n"),
]
ALL = "resolutions = [(128, 128), (256, 256), (420, 540)]"
VARIANTS = dict(full=[], epoch128=TENTH + [(ALL, "resolutions = [(128, 128)]")], epoch420=TENTH + [(ALL, "resolutions = [(420, 540)]")])


def denoise(variant="full"):
    """A directory holding solution.py, changed as the variant says, description.md and input/."""
    d = f"{ROOT}/denoise/{variant}"
    shutil.rmtree(d, ignore_errors=True)
    code = open(hf_hub_download(HUB, SCRIPT, repo_type="dataset", token=False)).read()
    for old, new in VARIANTS[variant]:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    for sub in ("train", "train_cleaned", "test"):
        shutil.copytree(f"{DATA}/public/{sub}", f"{d}/input/{sub}")
    shutil.copy(f"{DATA}/public/description.md", d)
    with open(f"{d}/solution.py", "w") as f:
        f.write(code)
    return d


def rmse(submission):
    """MLE-bench's grade for the task: RMSE over the pixels, rows matched by id."""
    s, a = (pd.read_csv(f).sort_values("id") for f in (submission, f"{DATA}/private/answers.csv"))
    assert (s["id"].values == a["id"].values).all()
    return round(float(((s["value"].values - a["value"].values) ** 2).mean() ** 0.5), 5)
