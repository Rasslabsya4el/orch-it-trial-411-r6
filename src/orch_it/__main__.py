"""`python -m orch_it ...`"""

import sys

from orch_it.cli import main

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # a note may hold any text
    raise SystemExit(main())
