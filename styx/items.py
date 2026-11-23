"""Items handed to agents that are built from published sources instead of files of this repo.

Eleven training scripts, jobs a to k, from the HF dataset zjunlp/PredictBeforeExecute, for four of its Kaggle tasks
(TASKS), each with the task's public data as MLE-bench prepares it. fetch reads one task's prepared data, public and
private, from the dataset's tasks/data.zip into ~/.cache/styx/data/data/<task>/prepared, by HTTP range and without the
33 GB zip; item fetches it if it is not there.
"""

import functools
import json
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

ROOT = os.path.expanduser("~/.cache/styx")
HUB = "zjunlp/PredictBeforeExecute"
DENOISE, VENTILATOR, STATOIL, COVID = "denoising-dirty-documents", "ventilator-pressure-prediction", "statoil-iceberg-classifier-challenge", "stanford-covid-vaccine"


def joined(s, a, key):
    """The answers `a` with the submission `s` beside them, rows matched by `key` and sorted by it; the columns of `s` that `a` has too end in _s."""
    m = a.merge(s, on=key, suffixes=("", "_s"), validate="1:1").sort_values(key)
    assert len(m) == len(a) == len(s), "the submission and the answers do not hold the same ids"
    return m


def denoise(s, a):
    m = joined(s, a, "id")
    return ((m["value"] - m["value_s"]) ** 2).mean() ** 0.5


def ventilator(s, a):
    m = joined(s, a, "id")
    m = m[m["u_out"] == 0]  # the expiratory phase is not scored
    return (m["pressure"] - m["pressure_s"]).abs().mean()


def statoil(s, a):
    m = joined(s, a, "id")
    y, p = m["is_iceberg"], m["is_iceberg_s"].clip(1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def covid(s, a):
    m = joined(s, a, "id_seqpos")
    m = m[m["keep"]]
    return np.mean([((m[c] - m[f"{c}_s"]) ** 2).mean() ** 0.5 for c in ("reactivity", "deg_Mg_pH10", "deg_Mg_50C")])


# Per task: the metric as a forecaster reads it, its grade of a submission against the private answers (the number the
# task's MLE-bench grade.py gives), the private file of the answers and what of the public files goes into input/.
TASKS = {
    DENOISE: dict(metric="RMSE", grade=denoise, answers="answers.csv", inputs=("train", "train_cleaned", "test")),
    VENTILATOR: dict(metric="MAE", grade=ventilator, answers="test.csv", inputs=("train.csv", "test.csv", "sample_submission.csv")),
    STATOIL: dict(metric="log loss", grade=statoil, answers="test.csv", inputs=("train.json", "test.json", "sample_submission.csv")),
    COVID: dict(metric="MCRMSE", grade=covid, answers="test.csv", inputs=("train.json", "test.json", "sample_submission.csv")),
}
# Each job's task and solution. The letters keep the run ids out of episodes.
JOBS = dict(
    a=(DENOISE, "Automind_wo_knowledge_denoising-dirty-documents_13a296eb-c605-4737-bea2-edd07d49161a_run_c4121e82042c4af2b8755e35432adc9d"),
    b=(DENOISE, "Automind_wo_knowledge_denoising-dirty-documents_13a296eb-c605-4737-bea2-edd07d49161a_run_ceec4550855d47b18397376b17dc4e3a"),  # stops at once on torch 2.11
    c=(DENOISE, "Aide_v3_denoising-dirty-documents_1a0c30a7-6820-4c1f-b0eb-09d6233500ed_run_61772fcd13da488aa84353a615f35047"),
    d=(DENOISE, "Aide_o3mini_denoising-dirty-documents_535b5136-3a35-4d8e-97fd-8d21db8f6c38_run_86da71a0b3d74ebfaa74c1f6b4bc9137"),
    e=(DENOISE, "Automind_wo_knowledge_denoising-dirty-documents_58d3c580-d6ab-4fea-b7a5-c8d115efda09_run_5d34b54f165f4cd1b3c91bede4b52f13"),
    f=(VENTILATOR, "Automind_o3_mini_ventilator-pressure-prediction_f7a4402c-291d-48f8-a551-6df1f6fe0e8f_run_c024cdf8cb5a422bbace3c75c98a6a09"),
    g=(VENTILATOR, "Automind_o3_mini_ventilator-pressure-prediction_f7a4402c-291d-48f8-a551-6df1f6fe0e8f_run_51db934843a445d9adb2da80248e12f3"),
    h=(STATOIL, "Aide_v3_statoil-iceberg-classifier-challenge_fff61ac5-30df-4c7e-81f4-c689c99240b3_run_12d441195e804b34836464d9fa31630a"),
    i=(STATOIL, "Automind_v3_statoil-iceberg-classifier-challenge_aa954edb-2de6-4889-82a5-2dcd398eef4c_run_01a567f3baaf4b2990284fc172a8e272"),
    j=(COVID, "Automind_o3_mini_stanford-covid-vaccine_2bb6734d-60b2-463f-af9d-509ad3e7fb2f_run_5c980d4a05f048daac78c91cfe1ffb94"),
    k=(COVID, "Automind_o3_mini_stanford-covid-vaccine_2bb6734d-60b2-463f-af9d-509ad3e7fb2f_run_9344002ac0d942bc9a155ac4e0d685a4"),
)
# What each job's solution took and scored as published there, on an RTX 3090: seconds, test score.
PUBLISHED = dict(a=(324.3, 0.03719), b=(375.8, 0.31935), c=(977.1, 0.55767), d=(1265.9, 0.10309), e=(2642.7, 0.30242), f=(93, 1.215), g=(96, 1.494),
                 h=(27, 0.317), i=(101, 0.262), j=(62, 0.295), k=(51, 0.308))
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


def cut(old, new, ns):
    """The budgets `ns` of a script that trains `old` as published, the last: each of the others has `old` replaced by `new` with its budget in it."""
    return [(n, f"`{old}` is changed to `{new.format(n)}`", [(old, new.format(n))]) for n in ns[:-1]] + [(ns[-1], None, [])]


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
    **{j: cut(EPOCHS, "num_epochs = {}", (2, 5, 10)) for j in "cd"},
    f=cut("epochs = 5", "epochs = {}", (1, 3, 5)),
    g=cut("num_epochs = 5", "num_epochs = {}", (1, 3, 5)),
    h=cut("for epoch in range(20):", "for epoch in range({}):", (5, 10, 20)),
    i=cut("for epoch in range(50):", "for epoch in range({}):", (5, 15, 50)),
    **{j: cut("num_epochs = 20", "num_epochs = {}", (5, 10, 20)) for j in "jk"},
)
# What a smaller budget does to each job's run. cut: the same run stops early, as nothing in the script is keyed to its epochs (job a's stages
# get 0 epochs under one ReduceLROnPlateau). rescaled: the whole run changes, as job j's OneCycleLR has total_steps = len(train_loader) * num_epochs.
BUDGET_KIND = dict(a="cut", c="cut", d="cut", f="cut", g="cut", h="cut", i="cut", j="rescaled", k="cut")
# Jobs whose test score does not move with their budget: job c's five seeds scored 0.49 to 0.61 at 2 epochs and 0.54 to 0.56 at 10, on both GPUs.
FLAT = {"c"}
# A fixed seed of a script: a seed(...) call, as in torch.cuda.manual_seed_all(42), or a random_state.
SEED = r"(seed(?:_all)?\(|random_state=)42\b"


def prepared(task):
    return f"{ROOT}/data/data/{task}/prepared"


def fetch(task):
    """Writes the members of the dataset's tasks/data.zip under data/<task>/prepared/ that are not yet in ~/.cache/styx/data with their size."""
    prefix = f"data/{task}/prepared/"
    with HfFileSystem(token=False).open(f"datasets/{HUB}/tasks/data.zip", "rb", block_size=2**22) as f, zipfile.ZipFile(f) as z:
        for m in z.infolist():
            path = f"{ROOT}/data/{m.filename}"
            if m.filename.startswith(prefix) and not m.is_dir() and not (os.path.exists(path) and os.path.getsize(path) == m.file_size):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with z.open(m) as src, open(f"{path}.part", "wb") as dst:
                    shutil.copyfileobj(src, dst, 2**22)
                os.replace(f"{path}.part", path)


def item(job="a", variant="full", seed=None, epochs=None):
    """A directory holding solution.py, changed as the variant says, with every fixed seed 42 replaced by `seed` if given and the epochs
    cut to the budget `epochs` of BUDGETS if given, the task's description.md, input/ with its public files and the empty working/ and submission/ of MLE-bench."""
    task, solution = JOBS[job]
    d = f"{ROOT}/items/{job}-{variant}-s{seed}-e{epochs}-{os.getpid()}"  # one per process: evals run side by side
    shutil.rmtree(d, ignore_errors=True)
    code = open(hf_hub_download(HUB, f"solutions_subset_15/{task}/code/solution_{solution}.py", repo_type="dataset", token=False)).read()
    for old, new in VARIANTS[variant] + ([] if epochs is None else next(r for e, _, r in BUDGETS[job] if e == epochs)):
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    if seed is not None:
        assert re.search(SEED, code) and not re.search(r"\b42\b", re.sub(SEED, "", code))  # no 42 that is not a seed
        code = re.sub(SEED, rf"\g<1>{seed}", code)
    public, t = f"{prepared(task)}/public", TASKS[task]
    if not all(os.path.exists(f) for f in (f"{public}/description.md", f"{prepared(task)}/private/{t['answers']}", *(f"{public}/{n}" for n in t["inputs"]))):
        fetch(task)
    os.makedirs(f"{d}/input")
    for n in t["inputs"]:
        (shutil.copytree if os.path.isdir(f"{public}/{n}") else shutil.copy)(f"{public}/{n}", f"{d}/input/{n}")
    shutil.copy(f"{public}/description.md", d)
    for n in ("working", "submission"):
        os.makedirs(f"{d}/{n}")
    with open(f"{d}/solution.py", "w") as f:
        f.write(code)
    return d


def graded(task, submission):
    """MLE-bench's grade of a submission file for the task, rounded to five places."""
    t = TASKS[task]
    return round(float(t["grade"](pd.read_csv(submission), pd.read_csv(f"{prepared(task)}/private/{t['answers']}"))), 5)


def score(job, submission):
    """MLE-bench's grade for the job's task, rounded to five places."""
    return graded(JOBS[job][0], submission)


@functools.cache
def sample(task):
    """The grade of the task's sample submission, public/sample_submission.csv of its prepared data, which knows the task and nothing of a script."""
    return graded(task, f"{prepared(task)}/public/sample_submission.csv")


def got(row):
    """The test score of a row of the results file; rows from before there were many tasks hold it as test_rmse."""
    return row["test_score"] if "test_score" in row else row["test_rmse"]


def lines(path):
    """The rows of a JSON-lines file; a line that is not JSON, as the last of out/items/items.jsonl may be while it is appended to, is skipped."""
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def truth(rows, gpus):
    """Per job of BUDGETS and GPU of `gpus` with a run of the script as published that succeeded, its rows of the results file that succeeded:
    `runs` those runs, `seeds` the sweep's at each of the job's budgets it has, `truth` the seeds a forecast's test score is scored
    against, the sweep's at the job's full budget if it has any, else the runs as published, and `full` every run at the job's full
    budget, as published and in the sweep, which a forecast's seconds and peak GPU memory are scored against."""
    ok, ours = [r for r in rows if r.get("state") == "SUCCEEDED"], {}
    for j in BUDGETS:
        for g in gpus:
            runs = [r for r in ok if (r["script"], r["gpu"], r.get("variant")) == (j, g, "full")]
            sweep = {e: [r for r in ok if (r["script"], r["gpu"], r.get("variant"), r.get("epochs")) == (j, g, "sweep", e)] for e, _, _ in BUDGETS[j]}
            if runs:
                last = sweep[BUDGETS[j][-1][0]]
                ours[j, g] = dict(runs=runs, seeds={e: s for e, s in sweep.items() if s}, truth=last or runs, full=runs + last)
    return ours
