"""Before an agent's first job, what limit did it ask for, and what had it written and reasoned
until then: that the limit costs nothing, a margin, a runtime? Given a CPU to forecast with, how
often did it run python, and what had it written and reasoned before the first time?

scout scan scripts/s_decide.py -T ~/.cache/styx/table/transcripts --scans ~/.cache/styx/scans
scout view --scans ~/.cache/styx/scans

The transcripts are styx.data.transcripts; a transcript of an earlier run holds its agent's
reasoning only where the CLI gave it unasked.
"""

import json
import re

from inspect_scout import Result, Scanner, Transcript, scanner

UNIT = dict(s=1, m=60, h=3600)
SAID = dict(
    free=r"(limit|time)[^.]{0,80}(doesn'?t|does not|won'?t|no|isn'?t|not)[^.]{0,30}(cost|charg|bill|add)|(charged|billed|billing)[^.]{0,40}(per second|only for|actual|runtime|it runs)|only pay",
    margin=r"headroom|buffer|margin|to be safe|generous|plenty|cushion|comfortabl",
    runtime=r"(take|takes|run|runs|finish|last)[^.]{0,60}\b\d+(\.\d+)? ?(s|sec|seconds?|min|minutes?|m|h|hours?)\b|(only|a few|a couple of) (seconds|minutes)",
)


def until(t, command):
    """What the agent wrote and reasoned up to its first tool call matching `command`, that call's
    arguments (None if there is none), and the reasoning blocks that came without text."""
    said, withheld = [], 0
    for m in t.messages:
        if m.role != "assistant":
            continue
        for b in m.content if isinstance(m.content, list) else []:
            said.append(b.reasoning if b.type == "reasoning" else b.text if b.type == "text" else "")
            withheld += b.type == "reasoning" and not b.reasoning
        for c in m.tool_calls or []:
            if re.search(command, json.dumps(c.arguments)):
                return "\n".join(s for s in said if s.strip()), json.dumps(c.arguments), withheld
    return "\n".join(s for s in said if s.strip()), None, withheld


@scanner(messages="all")
def first_job() -> Scanner[Transcript]:
    """The limit of the first job in seconds, with all the agent had written and reasoned until then."""
    async def scan(t: Transcript) -> Result:
        said, call, withheld = until(t, r"styx post\b")
        limit = re.search(r"--time[ =]+(\d+)([smh]?)", call or "")
        gpu = re.search(r"--gpu[ =]+(\w+)", call or "")
        return Result(value=int(limit[1]) * UNIT[limit[2] or "s"] if limit else None, explanation=said[-3000:],
                      metadata=dict(gpu=gpu and gpu[1], chars=len(said), withheld=withheld))

    return scan


@scanner(messages="all")
def before_first_job() -> Scanner[Transcript]:
    """Whether, before its first job, the agent said the limit costs nothing, named a margin, or named a runtime."""
    async def scan(t: Transcript) -> list[Result]:
        said, call, _ = until(t, r"styx post\b")
        found = {k: re.search(p, said, re.I) if call else None for k, p in SAID.items()}
        return [Result(label=k, value=bool(f) if call else None, explanation=said[max(0, f.start() - 200):f.end() + 200] if f else None) for k, f in found.items()]

    return scan


@scanner(messages="all")
def ran_python() -> Scanner[Transcript]:
    """How many tool calls ran python, with all the agent had written and reasoned before the first."""
    async def scan(t: Transcript) -> Result:
        said, _, withheld = until(t, r"\bpython3?\b")
        runs = sum(bool(re.search(r"\bpython3?\b", json.dumps(c.arguments))) for m in t.messages if m.role == "assistant" for c in m.tool_calls or [])
        return Result(value=runs, explanation=said[-3000:], metadata=dict(chars=len(said), withheld=withheld))

    return scan
