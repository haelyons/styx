"""Before an agent's first job, what limit did it ask for, and what had it written and reasoned
until then: that the limit costs nothing, a margin, a runtime? What does a model reading the
episode say its reason for the limit was? Given a CPU to forecast with, how often did it run
python, and what had it written and reasoned before the first time? Where Codex has no shell, how
often did it compute in the JavaScript sandbox its CLI leaves it?

scout scan scripts/s_decide.py -T ~/.cache/styx/table/transcripts --scans ~/.cache/styx/scans --model anthropic/claude-haiku-5-5
scout view --scans ~/.cache/styx/scans
why_limit asks the model, and only of post episodes; it needs $ANTHROPIC_API_KEY.

The transcripts are styx.data.transcripts; a transcript of an earlier run holds its agent's
reasoning only where the CLI gave it unasked.
"""

import json
import re

from inspect_scout import Result, Scanner, Transcript, llm_scanner, scanner

UNIT = dict(s=1, m=60, h=3600)
WHY = [
    "The limit costs nothing: only the seconds a job runs are charged, so a long limit is free",
    "A margin over a runtime it estimated, with no mention of what a limit costs",
    "A limit named without an estimate of the runtime",
    "Nothing about the limit before the job is posted",
    "No job is posted",
]
PHRASES = dict(
    free=r"(limit|time)[^.]{0,80}(doesn'?t|does not|won'?t|no|isn'?t|not)[^.]{0,30}(cost|charg|bill|add)|(charged|billed|billing)[^.]{0,40}(per second|only for|actual|runtime|it runs)|only pay",
    margin=r"headroom|buffer|margin|to be safe|generous|plenty|cushion|comfortabl",
    runtime=r"(take|takes|run|runs|finish|last)[^.]{0,60}\b\d+(\.\d+)? ?(s|sec|seconds?|min|minutes?|m|h|hours?)\b|(only|a few|a couple of) (seconds|minutes)",
)


def until(t, command):
    """What the agent wrote and reasoned up to its first tool call matching `command`, that call's
    arguments (None if there is none), and the reasoning blocks that came without text."""
    written, withheld = [], 0
    for m in t.messages:
        if m.role != "assistant":
            continue
        for b in m.content if isinstance(m.content, list) else []:
            written.append(b.reasoning if b.type == "reasoning" else b.text if b.type == "text" else "")
            withheld += b.type == "reasoning" and not b.reasoning
        for c in m.tool_calls or []:
            if re.search(command, json.dumps(c.arguments)):
                return "\n".join(s for s in written if s.strip()), json.dumps(c.arguments), withheld
    return "\n".join(s for s in written if s.strip()), None, withheld


@scanner(messages="all")
def first_job() -> Scanner[Transcript]:
    """The limit of the first job in seconds, with all the agent had written and reasoned until then."""
    async def scan(t: Transcript) -> Result:
        written, call, withheld = until(t, r"styx post\b")
        limit = re.search(r"--time[ =]+(\d+)([smh]?)", call or "")
        gpu = re.search(r"--gpu[ =]+(\w+)", call or "")
        return Result(value=int(limit[1]) * UNIT[limit[2] or "s"] if limit else None, explanation=written[-3000:],
                      metadata=dict(gpu=gpu and gpu[1], chars=len(written), withheld=withheld))

    return scan


@scanner(messages="all")
def before_first_job() -> Scanner[Transcript]:
    """Whether, before its first job, the agent wrote that the limit costs nothing, named a margin, or named a runtime."""
    async def scan(t: Transcript) -> list[Result]:
        written, call, _ = until(t, r"styx post\b")
        found = {k: re.search(p, written, re.I) if call else None for k, p in PHRASES.items()}
        return [Result(label=k, value=bool(f) if call else None, explanation=written[max(0, f.start() - 200):f.end() + 200] if f else None) for k, f in found.items()]

    return scan


@scanner(messages="all")
def ran_python() -> Scanner[Transcript]:
    """How many tool calls ran python, with all the agent had written and reasoned before the first."""
    async def scan(t: Transcript) -> Result:
        written, _, withheld = until(t, r"\bpython3?\b")
        runs = sum(bool(re.search(r"\bpython3?\b", json.dumps(c.arguments))) for m in t.messages if m.role == "assistant" for c in m.tool_calls or [])
        return Result(value=runs, explanation=written[-3000:], metadata=dict(chars=len(written), withheld=withheld))

    return scan


@scanner(messages="all")
def ran_javascript() -> Scanner[Transcript]:
    """How many of Codex's JavaScript runs called no tool, of how many, with those scripts. Codex calls every tool from such a run."""
    async def scan(t: Transcript) -> Result:
        ran = [c.arguments["input"] for m in t.messages if m.role == "assistant" for c in m.tool_calls or [] if c.function == "exec"]
        alone = [js for js in ran if "tools." not in js]
        return Result(value=len(alone), explanation="\n---\n".join(alone)[-3000:], metadata=dict(runs=len(ran)))

    return scan


@scanner(messages="all")
def why_limit() -> Scanner[Transcript]:
    """A model's reading of why the agent asked for the limit of its first job."""
    ask = llm_scanner(
        question="The agent posts GPU jobs with `styx post --gpu <gpu> --time <limit> -- <command>`. Look only at what it wrote and reasoned "
                 "before its first such call. Which of these best describes the reason it gave for that limit?",
        answer=WHY)

    async def scan(t: Transcript) -> Result:
        return await ask(t) if t.task_set == "post" else Result(value=None)

    return scan

