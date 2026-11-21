#!/usr/bin/python3
"""`styx` as an agent runs it. It holds no credentials: it leaves the call, and for `post` a copy of
the current directory, in $STYX_SPOOL and prints what the broker (styx.pool serve) answers.
"""

import json
import os
import shutil
import sys
import time
import uuid

SKIP = (".git", ".venv", "__pycache__")


def skip(d, names):
    big = lambda n: os.path.isfile(os.path.join(d, n)) and os.path.getsize(os.path.join(d, n)) > 50e6
    return [n for n in names if n in SKIP or big(n)]


def main():
    spool, rid = os.environ["STYX_SPOOL"], uuid.uuid4().hex[:10]
    d = os.path.join(spool, "new-" + rid)
    os.makedirs(d)
    if sys.argv[1:2] == ["post"]:
        shutil.copytree(".", os.path.join(d, "dir"), ignore=skip, symlinks=True)
    with open(os.path.join(d, "argv.json"), "w") as f:
        json.dump(sys.argv[1:], f)
    os.rename(d, os.path.join(spool, "in-" + rid))
    out = os.path.join(spool, f"out-{rid}.json")
    while not os.path.exists(out):
        time.sleep(0.2)
    r = json.load(open(out))
    sys.stdout.write(r["out"])
    sys.stderr.write(r["err"])
    sys.exit(r["code"])


if __name__ == "__main__":
    main()
