# styx

Are agents bad at posting GPU jobs? An agent on this machine is given an ML workload and a pool of
GPU nodes it can only reach through `styx post`. Every job it posts is recorded and compared with
our own posting of the same workload.

## Output
- Code and data only. No specs, plans, reports, figures or tables.
- The repo describes only what exists. No TODOs or planned work.
- No smoke tests or test suites. Run the real thing and fix what breaks.

## Code
- `styx/` is the core. `jobs/` holds the workloads handed to agents. `tasks/` holds Inspect tasks.
- A task's docstring is its question, nothing else. Its constants stay in it.
- Used twice, it moves to the core. Anything episodes are compared on has one implementation, in the core.
- Follow /andrej-karpathy-skills:karpathy-guidelines.

## Runs
- Launch only from a clean committed tree. Name the run first:
  `git log -1 --format=%cd-%h --date=format:%Y%m%d > COMMIT`
- Agents reach the pool only through `styx`. They are never given `sky`.
- The agent's working directory is outside this repo. `out/` holds the reference answers.
- Every job records seconds, exit code, peak GPU memory and GPU busy share on the node.
- The next commit message carries the run's evidence and names the commit that ran.

## Data
- Results go to the private HF dataset as `<task>/<date>-<commit>/`. No prose.

## Limits
- Never print credentials.
- Spend limit: $500 without asking. Pool nodes are launched with `-i 60 --down`.
