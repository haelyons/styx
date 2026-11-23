"""Items handed to agents that are built from published sources instead of files of this repo.

denoise: five denoising-dirty-documents solutions from the HF dataset zjunlp/PredictBeforeExecute,
with the task's public data as MLE-bench prepares it. Needs
~/.cache/styx/data/data/denoising-dirty-documents, that folder of the dataset's tasks/data.zip.
"""

import os
import re
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = os.path.expanduser("~/.cache/styx")
HUB = "zjunlp/PredictBeforeExecute"
CODE = "solutions_subset_15/denoising-dirty-documents/code/solution_"
# Published there on an RTX 3090: seconds, test RMSE. The letters keep the run ids out of episodes.
JOBS = dict(
    a="Automind_wo_knowledge_denoising-dirty-documents_13a296eb-c605-4737-bea2-edd07d49161a_run_c4121e82042c4af2b8755e35432adc9d",  # 324.3, 0.03719
    b="Automind_wo_knowledge_denoising-dirty-documents_13a296eb-c605-4737-bea2-edd07d49161a_run_ceec4550855d47b18397376b17dc4e3a",  # 375.8, 0.31935; stops at once on torch 2.11
    c="Aide_v3_denoising-dirty-documents_1a0c30a7-6820-4c1f-b0eb-09d6233500ed_run_61772fcd13da488aa84353a615f35047",  # 977.1, 0.55767
    d="Aide_o3mini_denoising-dirty-documents_535b5136-3a35-4d8e-97fd-8d21db8f6c38_run_86da71a0b3d74ebfaa74c1f6b4bc9137",  # 1265.9, 0.10309
    e="Automind_wo_knowledge_denoising-dirty-documents_58d3c580-d6ab-4fea-b7a5-c8d115efda09_run_5d34b54f165f4cd1b3c91bede4b52f13",  # 2642.7, 0.30242
)
DATA = f"{ROOT}/data/data/denoising-dirty-documents/prepared"
# ArchPilot's proxy for job a: the same code for one epoch on a tenth of the training data. The
# script trains at three resolutions, so the epoch is taken at its first and at its last.
MARK = "    # Progressive resizing training\n"
TENTH = [
    (MARK, "    train_files, train_clean_files = train_files[: len(train_files) // 10], train_clean_files[: len(train_files) // 10]\n" + MARK),
    ("epochs_per_resolution = [5, 5, 10]", "epochs_per_resolution = [1]"),
    ("criterion, epochs=10\n", "criterion, epochs=0\n"),
]
ALL = "resolutions = [(128, 128), (256, 256), (420, 540)]"
VARIANTS = dict(full=[], epoch128=TENTH + [(ALL, "resolutions = [(128, 128)]")], epoch420=TENTH + [(ALL, "resolutions = [(420, 540)]")])

# Each job's budgets: (epochs trained, what the script changes to train that many, replacements), the last being the script as published.
# Job a trains 5, 5 and 10 epochs at three resolutions and then 10 at full size; a stage given 0 epochs trains nothing and leaves best_model.pth to the stages before it.
STAGES = "epochs_per_resolution = [5, 5, 10]"
FINAL = "criterion, epochs=10\n"
EPOCHS = "num_epochs = 10"
BUDGETS = dict(
    a=[(5, "`epochs_per_resolution = [5, 5, 10]` is changed to `epochs_per_resolution = [5, 0, 0]` and the final stage's `epochs=10` to `epochs=0`", [(STAGES, "epochs_per_resolution = [5, 0, 0]"), (FINAL, "criterion, epochs=0\n")]),
       (10, "`epochs_per_resolution = [5, 5, 10]` is changed to `epochs_per_resolution = [5, 5, 0]` and the final stage's `epochs=10` to `epochs=0`", [(STAGES, "epochs_per_resolution = [5, 5, 0]"), (FINAL, "criterion, epochs=0\n")]),
       (20, "the final stage's `epochs=10` is changed to `epochs=0`", [(FINAL, "criterion, epochs=0\n")]),
       (30, None, [])],
    **{j: [(n, f"`num_epochs = 10` is changed to `num_epochs = {n}`", [(EPOCHS, f"num_epochs = {n}")]) for n in (2, 5)] + [(10, None, [])] for j in "cd"},
)


def denoise(job="a", variant="full", seed=None, epochs=None):
    """A directory holding solution.py, changed as the variant says, with every fixed seed 42 replaced by `seed` if given and the epochs
    cut to the budget `epochs` of BUDGETS if given, description.md and input/."""
    d = f"{ROOT}/denoise/{job}-{variant}-s{seed}-e{epochs}-{os.getpid()}"  # one per process: evals run side by side
    shutil.rmtree(d, ignore_errors=True)
    code = open(hf_hub_download(HUB, f"{CODE}{JOBS[job]}.py", repo_type="dataset", token=False)).read()
    for old, new in VARIANTS[variant] + ([] if epochs is None else next(r for e, _, r in BUDGETS[job] if e == epochs)):
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    if seed is not None:
        code, n = re.subn(r"(seed\(|random_state=)42\b", rf"\g<1>{seed}", code)
        assert n and not re.search(r"\b42\b", code), n
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
