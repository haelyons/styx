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
import sys
import time

ROOT = os.path.expanduser("~/.cache/styx")
HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = json.dumps(dict(sandbox=dict(enabled=True, autoAllowBashIfSandboxed=True, allowUnsandboxedCommands=False)))


def claude(model, prompt, ep):
    cmd = ["-p", prompt, "--output-format", "stream-json", "--verbose", "--permission-mode", "dontAsk",
           "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "", "--settings", SANDBOX,
           "--add-dir", f"{ep}/spool", "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", "Bash"]
    return cmd + (["--model", model] if model else [])


def codex(model, prompt, ep):
    cmd = ["exec", "--json", "--skip-git-repo-check", "--ephemeral", "-s", "workspace-write",
           "--add-dir", f"{ep}/spool", "-o", f"{ep}/answer.txt"]
    return cmd + (["-m", model] if model else []) + [prompt]


def answer(agent, ep):
    """The agent's last message, and what its CLI reported of its own usage."""
    events = [json.loads(l) for l in open(f"{ep}/transcript.jsonl") if l.startswith("{")]
    if agent == "claude":
        last = next((e for e in reversed(events) if e.get("type") == "result"), {})
        return last.get("result") or "", {k: last.get(k) for k in ("num_turns", "total_cost_usd", "usage")}
    text = open(f"{ep}/answer.txt").read() if os.path.exists(f"{ep}/answer.txt") else ""
    return text, dict(usage=[e["usage"] for e in events if e.get("type") == "turn.completed"])


async def run(agent, model, prompt, files, ep, minutes):
    """One episode. Returns the agent's last message, its seconds, and whether it ran out of time."""
    for d in ("work", "spool", "bin"):
        os.makedirs(f"{ep}/{d}")
    for f in files:
        shutil.copy(f, f"{ep}/work/")
    shutil.copy(f"{HERE}/client.py", f"{ep}/bin/styx")
    os.chmod(f"{ep}/bin/styx", 0o755)
    exe = shutil.which(agent)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE", "VIRTUAL_ENV"))}
    env.update(PATH=f"{ep}/bin:/usr/bin:/bin", STYX_SPOOL=f"{ep}/spool")
    broker = await asyncio.create_subprocess_exec(sys.executable, "-m", "styx.pool", "serve", f"{ep}/spool", f"{ep}/ledger.jsonl")
    t = time.time()
    with open(f"{ep}/transcript.jsonl", "w") as out, open(f"{ep}/stderr.txt", "w") as err:
        p = await asyncio.create_subprocess_exec(
            exe, *globals()[agent](model, prompt, ep), cwd=f"{ep}/work", env=env,
            stdin=asyncio.subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True,
        )
        try:
            await asyncio.wait_for(p.wait(), minutes * 60)
            late = False
        except asyncio.TimeoutError:
            os.killpg(p.pid, signal.SIGKILL)
            late = True
    broker.terminate()
    text, usage = answer(agent, ep)
    return dict(answer=text, agent_s=round(time.time() - t, 1), late=late, exit=p.returncode, **usage)
