"""What Codex reads its files through where it has no shell: an MCP server on stdin and stdout with
read, glob and grep, which answer only under the directory it is started on.

python styx/files.py <directory>   (agents.codex has Codex start it)
"""

import glob
import itertools
import json
import os
import re
import sys

ROOT = os.path.realpath(sys.argv[1])


def schema(**types):
    return dict(type="object", properties={k: dict(type=t) for k, t in types.items()}, required=list(types)[:1])


TOOLS = dict(
    read=("Lines of a text file, numbered. offset is the first line (from 1), limit how many (2000 if not given).", schema(path="string", offset="integer", limit="integer")),
    glob=("Files whose path matches a glob pattern (** crosses directories), each with its size in bytes.", schema(pattern="string")),
    grep=("Lines that match a regular expression, in the text files a glob pattern names (every file if not given).", schema(pattern="string", glob="string")),
)


def inside(path):
    full = os.path.realpath(os.path.join(ROOT, path))
    assert full == ROOT or full.startswith(ROOT + os.sep), f"{path} is outside this directory"
    return full


def files(pattern):
    return sorted(f for f in glob.glob(pattern, root_dir=ROOT, recursive=True) if os.path.isfile(inside(f)))


def matches(pattern, names):
    for name in names:
        with open(inside(name), "rb") as f:
            if b"\0" in f.read(1024):  # not text
                continue
        with open(inside(name), errors="replace") as f:
            yield from (f"{name}:{n}:{line.rstrip()[:500]}" for n, line in enumerate(f, 1) if re.search(pattern, line))


def call(name, a):
    if name == "read":
        first = a.get("offset") or 1
        with open(inside(a["path"]), errors="replace") as f:
            return "\n".join(f"{n}\t{line.rstrip()[:2000]}" for n, line in enumerate(itertools.islice(f, first - 1, first - 1 + (a.get("limit") or 2000)), first))
    if name == "glob":
        found = files(a["pattern"])
        return "\n".join(f"{os.path.getsize(inside(f))}\t{f}" for f in found[:1000]) + f"\n({len(found)} files)"
    found = list(itertools.islice(matches(a["pattern"], files(a.get("glob") or "**")), 501))
    return "\n".join(found[:500]) + ("\n(more than 500 lines)" if len(found) > 500 else f"\n({len(found)} lines)")


if __name__ == "__main__":
    for line in sys.stdin:
        m = json.loads(line)
        if "id" not in m:  # a notification
            continue
        if m["method"] == "initialize":
            r = dict(protocolVersion=m["params"]["protocolVersion"], capabilities=dict(tools={}), serverInfo=dict(name="files", version="1"))
        elif m["method"] == "tools/list":
            r = dict(tools=[dict(name=n, description=d, inputSchema=s, annotations=dict(readOnlyHint=True)) for n, (d, s) in TOOLS.items()])
        elif m["method"] == "tools/call":
            try:
                r = dict(content=[dict(type="text", text=call(m["params"]["name"], m["params"].get("arguments") or {}))])
            except Exception as e:
                r = dict(content=[dict(type="text", text=f"{type(e).__name__}: {e}")], isError=True)
        else:
            r = {}
        print(json.dumps(dict(jsonrpc="2.0", id=m["id"], result=r)), flush=True)
