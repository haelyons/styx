"""Runs a coding agent installed on this machine, headless and under its own subscription, in an
episode directory outside this repo.

<episode>/work             the agent's directory, holding the task's files
<episode>/spool            where its styx calls and the directories it posts are left
<episode>/bin/styx         styx/client.py; PATH is this and the system's
<episode>/ledger.jsonl     every styx call, written by the broker
<episode>/transcript.jsonl the agent's own event stream
"""

import asyncio
import json
import os
import shutil
import signal
import sqlite3
import sys
import time

ROOT = os.path.expanduser("~/.cache/styx")
HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
ACCOUNT = os.environ.get("CLAUDE_CONFIG_DIR", f"{HOME}/.claude")
LOGIN = f"{HOME}/.codex/auth.json"
# What an agent allowed to compute here gets: 4 threads, 8 GB, and a Python 3.12 with the CPU wheels
# of torch 2.11.0 and torchvision 0.26.0, scikit-learn, pandas and pillow.
# $STYX_CORES names other threads, so that evals of different agents can run side by side.
CAGE = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=8G", "-p", "MemorySwapMax=0", "taskset", "-c", os.environ.get("STYX_CORES", "0-3")]
CPU_ENV = f"{ROOT}/cpu-env"


# What of the home directory each jailed CLI needs: its install, and where it keeps its account.
OPEN = dict(
    claude=lambda ep: ["--ro-bind", f"{HOME}/.local/share/claude", f"{HOME}/.local/share/claude", "--bind", ACCOUNT, ACCOUNT,
                       "--bind-try", f"{HOME}/.claude.json", f"{HOME}/.claude.json", "--tmpfs", f"{ACCOUNT}/projects"],
    opencode=lambda ep: ["--ro-bind", f"{HOME}/.opencode", f"{HOME}/.opencode", "--bind", f"{ep}/home", f"{HOME}/.local/share/opencode"],
)


def jail(agent, ep):
    """bubblewrap for Claude Code, whose own sandbox does not start where the kernel restricts
    nested user namespaces, and for OpenCode, which has none: the system read-only, the home
    directory empty but for the CLI, its account and the episode. The network stays. Codex runs
    under its own sandbox: no network, writes only in the episode, everything readable."""
    return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", HOME, *OPEN[agent](ep),
            "--ro-bind-try", CPU_ENV, CPU_ENV, "--ro-bind-try", f"{HOME}/.local/share/uv/python", f"{HOME}/.local/share/uv/python",
            "--bind", ep, ep, "--die-with-parent"]


def account(path):
    """OpenCode's database with its login and none of the user's sessions."""
    src = sqlite3.connect(f"file:{HOME}/.local/share/opencode/opencode.db?mode=ro", uri=True)
    db = sqlite3.connect(path)
    src.backup(db)
    for (table,) in db.execute("select name from sqlite_master where type = 'table'").fetchall():
        if table not in ("account", "account_state", "migration", "data_migration"):
            db.execute(f'delete from "{table}"')
    db.commit()
    db.execute("vacuum")
    db.close()


def claude(model, prompt, ep, shell):
    cmd = ["-p", prompt, "--output-format", "stream-json", "--verbose", "--permission-mode", "dontAsk",
           "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
           "--add-dir", f"{ep}/spool", "--allowedTools", "Read", "Glob", "Grep"] + (["Write", "Edit", "Bash"] if shell else [])
    return cmd + (["--model", model] if model else [])


def codex(model, prompt, ep, shell):
    """Without a shell Codex still runs commands, but can write nothing."""
    cmd = ["exec", "--json", "--skip-git-repo-check", "--ephemeral", "-c", "model_reasoning_effort=medium", "-s", "workspace-write" if shell else "read-only",
           "--add-dir", f"{ep}/spool", "-o", f"{ep}/answer.txt"]
    return cmd + (["-m", model] if model else []) + [prompt]


def opencode(model, prompt, ep, shell):
    """Permissions come from $OPENCODE_CONFIG, which run() writes: everything, or no shell and no edits."""
    return ["run", "--format", "json", "--auto", "-m", model, prompt]


def answer(agent, ep):
    """The agent's last message, and what its CLI reported of its own usage."""
    events = [json.loads(l) for l in open(f"{ep}/transcript.jsonl") if l.startswith("{")]
    failed = [e for e in events if e.get("type") == ("error" if agent == "opencode" else "turn.failed") or e.get("is_error")]
    if failed:  # a usage limit or a provider error is no answer to score
        raise RuntimeError(f"{agent} did not run: {json.dumps(failed[-1])[:300]}")
    if agent == "claude":
        last = next((e for e in reversed(events) if e.get("type") == "result"), {})
        return last.get("result") or "", {k: last.get(k) for k in ("num_turns", "total_cost_usd", "usage")}
    if agent == "opencode":
        said = [e["part"] for e in events if e.get("type") == "text"]
        steps = [e["part"] for e in events if e.get("type") == "step_finish"]
        text = "".join(p["text"] for p in said if p["messageID"] == said[-1]["messageID"])
        return text, dict(num_turns=len(steps), total_cost_usd=sum(p.get("cost", 0) for p in steps), usage=[p.get("tokens") for p in steps])
    text = open(f"{ep}/answer.txt").read() if os.path.exists(f"{ep}/answer.txt") else ""
    return text, dict(usage=[e["usage"] for e in events if e.get("type") == "turn.completed"])


def episode(task, run, agent, model, sample, epoch):
    return f"{ROOT}/{task}/{run}/{agent}{'-' + model.replace('/', '_') if model else ''}/{sample}-{epoch}"


def names(ep, words):
    """Whether the agent's transcript names any of `words`, which say where the answers are."""
    text = open(f"{ep}/transcript.jsonl").read()
    return any(w in text for w in words)


async def burnt(p):
    """CPU seconds used in the CAGE scope that process `p` runs in, read until it ends."""
    await asyncio.sleep(1)
    stat, seconds = "/sys/fs/cgroup" + open(f"/proc/{p.pid}/cgroup").read().strip().split("::")[1] + "/cpu.stat", 0
    assert stat.endswith(".scope/cpu.stat"), stat
    while p.returncode is None:
        try:
            seconds = int(open(stat).read().split()[1]) / 1e6
        except FileNotFoundError:
            break
        await asyncio.sleep(2)
    return round(seconds, 1)


async def run(agent, model, prompt, files, ep, minutes, tools="pool"):
    """One episode. `tools` is what the agent has besides reading its files: "pool" a shell with
    styx, "cpu" a shell inside CAGE with CPU_ENV's python, "none" neither. Returns the agent's last
    message, its seconds, the CPU seconds used inside CAGE and whether it ran out of time."""
    for d in ("work", "spool", "bin"):
        os.makedirs(f"{ep}/{d}")
    for f in files:
        shutil.copytree(f, f"{ep}/work/{os.path.basename(f)}") if os.path.isdir(f) else shutil.copy(f, f"{ep}/work/")
    if tools == "pool":
        shutil.copy(f"{HERE}/client.py", f"{ep}/bin/styx")
        os.chmod(f"{ep}/bin/styx", 0o755)
        broker = await asyncio.create_subprocess_exec(sys.executable, "-m", "styx.pool", "serve", f"{ep}/spool", f"{ep}/ledger.jsonl")
    if tools == "cpu":
        for name in ("python", "python3"):  # a link would lose the environment's packages
            with open(f"{ep}/bin/{name}", "w") as f:
                f.write(f'#!/bin/sh\nexec {CPU_ENV}/bin/python "$@"\n')
            os.chmod(f"{ep}/bin/{name}", 0o755)
    exe = os.path.realpath(shutil.which(agent))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE", "VIRTUAL_ENV"))}
    env.update(PATH=f"{ep}/bin:/usr/bin:/bin", STYX_SPOOL=f"{ep}/spool")
    if agent == "codex":  # a home with its login and nothing else of the user's: no AGENTS.md, config or skills
        os.makedirs(f"{ep}/home/.codex")
        shutil.copy(LOGIN, f"{ep}/home/.codex/auth.json")
        env.update(HOME=f"{ep}/home")
    if agent == "opencode":  # its account, and a config of ours in place of the user's
        os.makedirs(f"{ep}/home")
        account(f"{ep}/home/opencode.db")
        allow = {"*": "allow"} if tools != "none" else {"*": "allow", "bash": "deny", "edit": "deny"}
        with open(f"{ep}/opencode.json", "w") as f:
            json.dump(dict(autoupdate=False, share="disabled", permission=allow), f)
        env.update(OPENCODE_CONFIG=f"{ep}/opencode.json")
    cmd = (CAGE if tools == "cpu" else []) + (jail(agent, ep) if agent in OPEN else []) + [exe, *globals()[agent](model, prompt, ep, tools != "none")]
    t = time.time()
    with open(f"{ep}/transcript.jsonl", "w") as out, open(f"{ep}/stderr.txt", "w") as err:
        p = await asyncio.create_subprocess_exec(
            *cmd, cwd=f"{ep}/work", env=env,
            stdin=asyncio.subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True,
        )
        meter = asyncio.create_task(burnt(p)) if tools == "cpu" else None
        try:
            await asyncio.wait_for(p.wait(), minutes * 60)
            late = False
        except asyncio.TimeoutError:
            os.killpg(p.pid, signal.SIGKILL)
            late = True
    if tools == "pool":
        broker.terminate()
    if agent == "codex" and open(f"{ep}/home/.codex/auth.json").read() != open(LOGIN).read():
        shutil.copy(f"{ep}/home/.codex/auth.json", LOGIN)  # a login Codex renewed goes back
    shutil.rmtree(f"{ep}/home", ignore_errors=True)  # no login stays in the episode
    text, usage = answer(agent, ep)
    return dict(answer=text, agent_s=round(time.time() - t, 1), cpu_s=await meter if meter else None, late=late, exit=p.returncode, **usage)
