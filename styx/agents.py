"""Runs a coding agent installed on this machine, headless and under its own subscription, in an
episode directory outside this repo.

<episode>/work             the agent's directory, holding the task's files
<episode>/spool            where its styx calls and the directories it posts are left
<episode>/bin/styx         styx/client.py; PATH is this and the system's
<episode>/ledger.jsonl     every styx call, written by the broker
<episode>/prompt.txt       what the agent was asked
<episode>/transcript.jsonl the agent's own event stream, with its reasoning as its CLI gives it: a
                           summary from Claude Code and Codex, the text from OpenCode
<episode>/clock.txt        the second each of its lines arrived
<episode>/exec.jsonl       the JavaScript Codex ran where it has no shell, which its event stream leaves out
<episode>/before/          the same again for the session that forecast this one
"""

import asyncio
import datetime
import functools
import glob
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.tool import ToolCall

ROOT = os.path.expanduser("~/.cache/styx")
HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
ACCOUNT = os.environ.get("CLAUDE_CONFIG_DIR", f"{HOME}/.claude")
LOGIN = f"{HOME}/.codex/auth.json"
EFFORT = "medium"
# What an agent allowed to compute here gets: 4 threads, 8 GB, and a Python 3.12 with the CPU wheels
# of torch 2.11.0 and torchvision 0.26.0, scikit-learn, pandas and pillow.
# Every agent runs under a memory limit of its own, whatever it may use: the jail's home and /tmp
# are memory, and an agent that fills them or its own heap is stopped there and not by the kernel
# on the whole machine. What goes over is killed, and the agent goes on (OOMPolicy=continue).
# $STYX_CORES names other threads, so that evals of different agents can run side by side.
# Every agent's scope is in styx.slice, so that all of them together can be given a limit the machine has room for:
#   systemctl --user set-property --runtime styx.slice MemoryMax=18G MemorySwapMax=0
def wall(gb):
    return ["systemd-run", "--user", "--scope", "--quiet", "--slice=styx.slice", "-p", f"MemoryMax={gb}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]


WALL, CAGE = wall(2), wall(8) + ["taskset", "-c", os.environ.get("STYX_CORES", "0-3")]
CPU_ENV = f"{ROOT}/cpu-env"
# Models we serve ourselves (serve.yaml), by a name of ours: {"h100": "http://127.0.0.1:8001/v1"}.
# OpenCode reaches one as <name>/<model>.
ENDPOINTS = json.loads(os.environ.get("STYX_ENDPOINTS", "{}"))


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
           "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "", "--thinking-display", "summarized", "--effort", EFFORT,
           "--add-dir", f"{ep}/spool", "--allowedTools", "Read", "Glob", "Grep"] + (["Write", "Edit", "Bash"] if shell else [])
    return cmd + (["--model", model] if model else [])


def codex(model, prompt, ep, shell):
    """Without a shell Codex reads its files through styx/files.py and can write nothing. It keeps
    the JavaScript sandbox it calls every tool through, which has no files and no network; the
    session file that holds what it ran there is kept until scripts() has read it. Its web search
    is off, as the web tools of the other two are. Its reasoning effort is medium, as Claude Code's is."""
    reads = ["--disable", "shell_tool", "--disable", "unified_exec", "-c", f'mcp_servers.files.command="{sys.executable}"',
             "-c", f"mcp_servers.files.args={json.dumps([f'{HERE}/files.py', f'{ep}/work'])}"]
    cmd = ["exec", "--json", "--skip-git-repo-check", *(["--ephemeral"] if shell else reads), "-c", f"model_reasoning_effort={EFFORT}", "-c", 'model_reasoning_summary="detailed"', "-c", 'web_search="disabled"', "--disable", "apps", "-s", "workspace-write" if shell else "read-only",
           "--add-dir", f"{ep}/spool", "-o", f"{ep}/answer.txt"]
    return cmd + (["-m", model] if model else []) + [prompt]


def opencode(model, prompt, ep, shell):
    """Permissions come from $OPENCODE_CONFIG, which run() writes: everything, or no shell and no edits."""
    return ["run", "--format", "json", "--thinking", "--auto", "-m", model, prompt]


def answer(agent, ep):
    """The agent's last message, and the size of its run as its CLI reported it: tool calls, output
    tokens (reasoning included), input tokens not read from cache, and those read from cache."""
    events = [json.loads(l) for l in open(f"{ep}/transcript.jsonl") if l.startswith("{")]
    failed = [e for e in events if e.get("type") == ("error" if agent == "opencode" else "turn.failed") or e.get("is_error")]
    if failed:  # a usage limit or a provider error is no answer to score
        raise RuntimeError(f"{agent} did not run: {json.dumps(failed[-1])[:300]}")
    if agent == "claude":
        last = next((e for e in reversed(events) if e.get("type") == "result"), {})
        use = list((last.get("modelUsage") or {}).values())
        calls = sum(b["type"] == "tool_use" for e in events if e.get("type") == "assistant" for b in e["message"]["content"])
        return last.get("result") or "", dict(
            tool_calls=calls, output_tokens=sum(u["outputTokens"] for u in use), cached_tokens=sum(u["cacheReadInputTokens"] for u in use),
            input_tokens=sum(u["inputTokens"] + u["cacheCreationInputTokens"] for u in use), cost_usd=last.get("total_cost_usd"))
    if agent == "opencode":
        said = [e["part"] for e in events if e.get("type") == "text"]
        steps = [e["part"].get("tokens") or {} for e in events if e.get("type") == "step_finish"]
        text = "".join(p["text"] for p in said if p["messageID"] == said[-1]["messageID"])
        return text, dict(
            tool_calls=sum(e.get("type") == "tool_use" for e in events), output_tokens=sum(t.get("output", 0) + t.get("reasoning", 0) for t in steps),
            cached_tokens=sum((t.get("cache") or {}).get("read", 0) for t in steps), input_tokens=sum(t.get("input", 0) for t in steps),
            cost_usd=sum(e["part"].get("cost", 0) for e in events if e.get("type") == "step_finish"))
    text = open(f"{ep}/answer.txt").read() if os.path.exists(f"{ep}/answer.txt") else ""
    use = [e["usage"] for e in events if e.get("type") == "turn.completed"]
    calls = sum(e.get("type") == "item.completed" and e["item"]["type"] not in ("agent_message", "reasoning") for e in events)
    return text, dict(
        tool_calls=calls, output_tokens=sum(u["output_tokens"] for u in use), cached_tokens=sum(u["cached_input_tokens"] for u in use),
        input_tokens=sum(u["input_tokens"] - u["cached_input_tokens"] for u in use), cost_usd=None)


def episode(task, run, agent, model, sample, epoch):
    return f"{ROOT}/{task}/{run}/{agent}{'-' + model.replace('/', '_') if model else ''}/{sample}-{epoch}"


def names(ep, words):
    """Whether the agent's transcript names any of `words`, which say where the answers are."""
    text = open(f"{ep}/transcript.jsonl").read()
    return any(w in text for w in words)


WAITS = re.compile(r"sleep|styx (wait|status|logs?)|until |while ")


def spent(ep):
    """How an episode's seconds went, from its transcript and clock: seconds inside tool calls (calls
    side by side counted once), the part of them in commands that wait, the model seconds Claude
    Code reports for itself (None from the other CLIs), and model turns."""
    lines = open(f"{ep}/transcript.jsonl").read().splitlines()
    clock = [float(x) for x in open(f"{ep}/clock.txt").read().split()]
    calls, began, turns, api = [], {}, set(), None
    for at, line in zip(clock, lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        kind, item, part = e.get("type"), e.get("item") or {}, e.get("part") or {}
        if kind == "assistant":  # Claude Code
            turns.add(e["message"].get("id"))
            began.update({x["id"]: (at, json.dumps(x.get("input"))) for x in e["message"].get("content") or [] if x.get("type") == "tool_use"})
        elif kind == "user":
            content = e.get("message", {}).get("content") or []
            done = [x["tool_use_id"] for x in content if isinstance(x, dict) and x.get("type") == "tool_result" and x.get("tool_use_id") in began]
            calls += [(*began.pop(i), at) for i in done]
        elif kind == "result":
            api = e.get("duration_api_ms", 0) / 1000
        elif kind == "item.started" and item.get("type") in ("command_execution", "mcp_tool_call"):  # Codex
            began[item["id"]] = (at, item.get("command", ""))
        elif kind == "item.completed" and item.get("id") in began:
            calls.append((*began.pop(item["id"]), at))
        elif kind == "item.completed" and item.get("type") in ("agent_message", "reasoning"):
            turns.add(item["id"])
        elif kind == "tool_use":  # OpenCode: one event when the call ends, carrying its start and end
            took = part.get("state", {}).get("time") or {}
            if took.get("start") and took.get("end"):
                calls.append((at - (took["end"] - took["start"]) / 1000, json.dumps(part["state"].get("input")), at))
        elif kind == "step_start":
            turns.add(part.get("id"))

    def covered(spans):
        total = end = 0
        for a, b in sorted(spans):
            total, end = total + max(0, b - max(a, end)), max(end, b)
        return round(total, 1)

    return dict(tool_s=covered((a, b) for a, _, b in calls), wait_s=covered((a, b) for a, cmd, b in calls if WAITS.search(cmd)), model_s=api, turns=len(turns))


def messages(ep):
    """An episode as chat messages: its prompt, what the agent wrote, its reasoning where the CLI
    gave any (a Claude Code block without text is marked redacted), the tools it called and what
    came back. Each carries `t`, its second of the run, where the episode kept a clock."""
    lines = open(f"{ep}/transcript.jsonl").read().splitlines()
    clock = [float(x) for x in open(f"{ep}/clock.txt").read().split()] if os.path.exists(f"{ep}/clock.txt") else []
    chat, turns, called = [], {}, {}
    ran = [json.loads(l) for l in open(f"{ep}/exec.jsonl")] if os.path.exists(f"{ep}/exec.jsonl") else []
    if os.path.exists(f"{ep}/prompt.txt"):
        chat.append(ChatMessageUser(content=open(f"{ep}/prompt.txt").read()))

    def turn(key, at):
        if key not in turns:
            turns[key] = ChatMessageAssistant(content=[], tool_calls=[], metadata=at)
            chat.append(turns[key])
        return turns[key]

    def call(key, at, uid, function, arguments):
        called[uid] = function
        turn(key, at).tool_calls.append(ToolCall(id=uid, function=function, arguments=arguments if isinstance(arguments, dict) else {"input": arguments}))

    def result(uid, at, content, **more):
        text = content if isinstance(content, str) else "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content or [])
        chat.append(ChatMessageTool(content=text, tool_call_id=uid, function=called.get(uid), metadata={**at, **{k: v for k, v in more.items() if v not in (None, False)}}))

    def script(r):  # Codex's JavaScript, in its place by its second
        call(r["id"], {"t": r["t"]}, r["id"], "exec", r["input"])
        result(r["id"], {"t": r["t"]}, r["output"])

    for i, line in enumerate(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        at = {"t": clock[i]} if i < len(clock) else {}
        while ran and at and ran[0]["t"] <= at["t"]:
            script(ran.pop(0))
        kind, item, part = e.get("type"), e.get("item") or {}, e.get("part") or {}
        if kind == "assistant":  # Claude Code: one event for each block of a message
            for b in e["message"].get("content") or []:
                if b.get("type") == "text":
                    turn(e["message"].get("id"), at).content.append(ContentText(text=b["text"]))
                elif b.get("type") == "thinking":
                    turn(e["message"].get("id"), at).content.append(ContentReasoning(reasoning=b.get("thinking") or "", redacted=not b.get("thinking")))
                elif b.get("type") == "tool_use":
                    call(e["message"].get("id"), at, b["id"], b["name"], b.get("input"))
        elif kind == "user":
            content = e.get("message", {}).get("content")
            if isinstance(content, str):
                chat.append(ChatMessageUser(content=content, metadata=at))
            for b in content if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    result(b["tool_use_id"], at, b.get("content"), is_error=b.get("is_error"))
        elif kind == "item.completed" and item.get("type") in ("agent_message", "reasoning"):  # Codex
            turn(item["id"], at).content.append(ContentText(text=item.get("text") or "") if item["type"] == "agent_message" else ContentReasoning(reasoning=item.get("text") or ""))
        elif kind in ("item.started", "item.completed") and item.get("id") and item.get("type") not in ("agent_message", "reasoning"):
            ours = item["type"] == "mcp_tool_call"  # a tool of styx/files.py
            if item["id"] not in called:
                call(item["id"], at, item["id"], item["tool"] if ours else item["type"],
                     item.get("arguments") if ours else {k: v for k, v in item.items() if k not in ("id", "type", "status", "aggregated_output", "exit_code")})
            if kind == "item.completed":
                result(item["id"], at, (item.get("result") or {}).get("content") if ours else item.get("aggregated_output") or "", exit_code=item.get("exit_code"), status=item.get("status"))
        elif kind in ("text", "reasoning"):  # OpenCode
            turn(part.get("messageID"), at).content.append((ContentText if kind == "text" else ContentReasoning)(**{kind: part.get("text") or ""}))
        elif kind == "tool_use":  # one event when the call ends
            state = part.get("state") or {}
            call(part.get("messageID"), at, part["callID"], part.get("tool"), state.get("input"))
            result(part["callID"], at, state.get("output") or state.get("error") or "", status=state.get("status"))
    for r in ran:
        script(r)
    return chat


def scripts(ep, t):
    """To exec.jsonl, from the session files of a Codex that began at `t`: each JavaScript it ran,
    what that printed, and its second of the run. Nothing else of those files is kept: they name
    the account."""
    rows = []
    for f in glob.glob(f"{ep}/home/.codex/sessions/**/rollout-*.jsonl", recursive=True):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    parts = [(r["timestamp"], r["payload"]) for r in rows if isinstance(r.get("payload"), dict) and "call_id" in r["payload"]]
    printed = {p["call_id"]: p["output"] for _, p in parts if p["type"] == "custom_tool_call_output"}
    with open(f"{ep}/exec.jsonl", "w") as out:
        for stamp, p in sorted(parts, key=lambda x: x[0]):
            if p["type"] == "custom_tool_call":
                text = printed.get(p["call_id"]) or ""
                out.write(json.dumps(dict(t=round(datetime.datetime.fromisoformat(stamp).timestamp() - t, 2), id=p["call_id"], input=p["input"],
                                          output=text if isinstance(text, str) else "".join(b.get("text", "") for b in text))) + "\n")


async def burnt(p):
    """CPU seconds used in the scope that process `p` runs in, read until it ends. What the agent
    left running in the scope is then killed."""
    await asyncio.sleep(1)
    try:
        scope, seconds = "/sys/fs/cgroup" + open(f"/proc/{p.pid}/cgroup").read().strip().split("::")[1], 0
        assert scope.endswith(".scope"), scope
        while p.returncode is None:
            seconds = int(open(f"{scope}/cpu.stat").read().split()[1]) / 1e6
            await asyncio.sleep(2)
        with open(f"{scope}/cgroup.kill", "w") as f:
            f.write("1")
    except FileNotFoundError:
        pass
    return round(seconds, 1)


async def copy(stream, ep, t):
    """The CLI's event stream to transcript.jsonl, and to clock.txt the second each line arrived."""
    with open(f"{ep}/transcript.jsonl", "wb") as out, open(f"{ep}/clock.txt", "w") as clock:
        async for line in stream:
            out.write(line)
            clock.write(f"{time.time() - t:.2f}\n")
            out.flush()
            clock.flush()


async def limited(p, ep):
    """OpenCode waits out a usage limit without a word in its events; this ends it and says so."""
    while p.returncode is None:
        if any("usage limit reached" in open(f).read() for f in glob.glob(f"{ep}/home/log/*.log")):
            os.killpg(p.pid, signal.SIGKILL)
            return True
        await asyncio.sleep(10)
    return False


@functools.cache
def version(exe):
    """The first line of what the CLI `exe` prints for --version."""
    return (subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.splitlines() or [None])[0]


async def run(agent, model, prompt, files, ep, minutes, tools="pool", hold=0):
    """One episode. `tools` is what the agent has besides reading its files: "pool" a shell with
    styx, "cpu" a shell inside CAGE with CPU_ENV's python, "none" neither. `hold` is the pool's
    $STYX_HOLD for this episode. Returns the agent's last message, its seconds, the CPU seconds of
    all it ran and whether it ran out of time."""
    for d in ("work", "spool", "bin"):
        os.makedirs(f"{ep}/{d}")
    with open(f"{ep}/prompt.txt", "w") as f:
        f.write(prompt)
    for f in files:
        shutil.copytree(f, f"{ep}/work/{os.path.basename(f)}") if os.path.isdir(f) else shutil.copy(f, f"{ep}/work/")
    if tools == "pool":
        shutil.copy(f"{HERE}/client.py", f"{ep}/bin/styx")
        os.chmod(f"{ep}/bin/styx", 0o755)
        broker = await asyncio.create_subprocess_exec(sys.executable, "-m", "styx.pool", "serve", f"{ep}/spool", f"{ep}/ledger.jsonl", env=dict(os.environ, STYX_HOLD=str(hold)))
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
    if agent == "opencode":  # its account where there is one, and a config of ours in place of the user's
        os.makedirs(f"{ep}/home")
        if os.path.exists(f"{HOME}/.local/share/opencode/opencode.db"):
            account(f"{ep}/home/opencode.db")
        allow = {"*": "allow", "webfetch": "deny", "websearch": "deny", **({} if tools != "none" else {"bash": "deny", "edit": "deny"})}
        name, _, served = model.partition("/")
        ours = {name: dict(npm="@ai-sdk/openai-compatible", options=dict(baseURL=ENDPOINTS[name], apiKey="none"),
                           models={served: dict(tool_call=True, reasoning=True, limit=dict(context=131072, output=32768))})} if name in ENDPOINTS else {}
        with open(f"{ep}/opencode.json", "w") as f:
            json.dump(dict(autoupdate=False, share="disabled", permission=allow, provider=ours), f)
        env.update(OPENCODE_CONFIG=f"{ep}/opencode.json")
    cmd = (CAGE if tools == "cpu" else WALL) + (jail(agent, ep) if agent in OPEN else []) + [exe, *globals()[agent](model, prompt, ep, tools != "none")]
    t = time.time()
    with open(f"{ep}/stderr.txt", "w") as err:
        p = await asyncio.create_subprocess_exec(
            *cmd, cwd=f"{ep}/work", env=env, limit=2**27,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=err, start_new_session=True,
        )
        events = asyncio.create_task(copy(p.stdout, ep, t))
        meter = asyncio.create_task(burnt(p))
        limit = asyncio.create_task(limited(p, ep)) if agent == "opencode" else None
        try:
            await asyncio.wait_for(p.wait(), minutes * 60)
            late = False
        except asyncio.TimeoutError:
            os.killpg(p.pid, signal.SIGKILL)
            late = True
        await events
    if tools == "pool":
        broker.terminate()
    if agent == "codex" and open(f"{ep}/home/.codex/auth.json").read() != open(LOGIN).read():
        shutil.copy(f"{ep}/home/.codex/auth.json", LOGIN)  # a login Codex renewed goes back
    if agent == "codex" and tools == "none":
        scripts(ep, t)
    shutil.rmtree(f"{ep}/home", ignore_errors=True)  # no login stays in the episode
    if limit and await limit:
        raise RuntimeError("opencode did not run: usage limit reached")
    if p.returncode and not late and not os.path.getsize(f"{ep}/transcript.jsonl"):
        raise RuntimeError(f"{agent} did not run: {open(f'{ep}/stderr.txt').read()[:300]}")
    seconds = round(time.time() - t, 1)
    text, usage = answer(agent, ep)
    return dict(answer=text, agent_s=seconds, cpu_s=await meter, late=late, exit=p.returncode, host=os.uname().nodename, effort=EFFORT if agent in ("claude", "codex") else None, cli=version(exe), **usage)


SELF = [f"SELF_{q}{end}" for q in ("SECONDS", "OUTPUT_TOKENS", "TOOL_CALLS") for end in ("", "_LOW", "_HIGH")]
BEFORE = """You are about to be given the task below, in a new session. Before starting, forecast the size of your own run on it. Do not start the task.

<task>
{task}
</task>

End your reply with these lines, each a single number:
SELF_SECONDS=<seconds from the task arriving to your final reply, your median>
SELF_SECONDS_LOW=<10th percentile>
SELF_SECONDS_HIGH=<90th percentile>
SELF_OUTPUT_TOKENS=<tokens you will generate, reasoning included, your median>
SELF_OUTPUT_TOKENS_LOW=<10th percentile>
SELF_OUTPUT_TOKENS_HIGH=<90th percentile>
SELF_TOOL_CALLS=<tool calls you will make, your median>
SELF_TOOL_CALLS_LOW=<10th percentile>
SELF_TOOL_CALLS_HIGH=<90th percentile>"""


def numbers(text, keys):
    """The last `KEY=<number>` line of each key; nan where there is none."""
    found = {k: re.findall(rf"^{k}=\s*([0-9.eE+-]+)\s*$", text, re.M) for k in keys}
    return {k: float(v[-1]) if v else math.nan for k, v in found.items()}


async def before(agent, model, prompt, ep):
    """The agent's forecast of its own coming run, asked in a session of its own with nothing to
    read or run, so that the run is the same whether or not it was asked."""
    r = await run(agent, model, BEFORE.format(task=prompt), [], f"{ep}/before", 10, "none")
    return numbers(r["answer"], SELF)


def own(said, run):
    """ln(said / actual) for the agent's forecast of its own seconds, output tokens and tool calls,
    and whether each actual fell between its 10th and 90th percentile."""
    actual = dict(SELF_SECONDS=run["agent_s"], SELF_OUTPUT_TOKENS=run["output_tokens"], SELF_TOOL_CALLS=run["tool_calls"])
    off = {f"{k.lower()}_err": math.log(said[k] / a) if said[k] > 0 and a > 0 else math.nan for k, a in actual.items()}
    return dict(off, **{f"{k.lower()}_in": said[f"{k}_LOW"] <= a <= said[f"{k}_HIGH"] for k, a in actual.items()})
