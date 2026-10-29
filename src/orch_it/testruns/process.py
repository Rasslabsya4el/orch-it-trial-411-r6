"""One test command run under its bound (#394 decision 2, the `cut` row).

`run_bounded` starts the command, waits until it exits or its bound passes, and then kills its
whole process tree either way, so no process of the run is left; how a tree is started and
killed on each OS lives in `orch_it.proctree`. The bound passes when either the monotonic clock or the
wall clock says so: a wall clock stepping back never stretches it, and a machine that sleeps
through it still cuts the run. A jump of the wall clock between two polls longer than
`SLEEP_GAP_SECONDS` that carries the run past its bound marks it `slept` (a short sleep long
before the bound does not). The command's output is read through a pipe and only its last
`LOG_TAIL_BYTES` are kept. A caller may also set an `abort` event: the run is then killed like a
cut one (a machine place sets it when its caller is gone).
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from orch_it.proctree import KILL_WAIT_SECONDS, ProcessTree

POLL_SECONDS = 0.5
SLEEP_GAP_SECONDS = 60.0
LOG_TAIL_BYTES = 8192


@dataclass(frozen=True)
class Finished:
    rc: int | None  # None: killed at the bound
    cut: bool
    slept: bool
    seconds: float
    log_tail: str = ""


class CommandMissing(Exception):
    """The command could not be started (not found, not executable, not a valid argument)."""


def run_bounded(argv: list[str], *, cwd: Path, env: dict[str, str] | None, bound_seconds: float,
                wall=time.time, poll_seconds: float = POLL_SECONDS,
                abort: threading.Event | None = None) -> Finished:
    """Run `argv` until it exits, its bound passes, or `abort` is set (then it is killed like a
    cut run: a machine place sets it when its caller is gone)."""

    try:
        tree = ProcessTree.start(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
    except (OSError, ValueError) as e:
        raise CommandMissing(f"{argv[0] if argv else ''}: {e}") from None
    tail = _Tail(tree.process.stdout)
    started_wall = last = wall()
    started_mono = time.monotonic()
    deadline = started_wall + bound_seconds
    slept = cut = False
    try:
        while True:
            try:
                tree.process.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                pass
            if abort is not None and abort.is_set():
                cut = True
                break
            now = wall()
            if now - last > poll_seconds + SLEEP_GAP_SECONDS and last < deadline <= now:
                slept = True  # the machine slept across the bound
            last = now
            if now >= deadline or time.monotonic() - started_mono >= bound_seconds:
                cut = True
                break
    finally:
        tree.kill()
        tail.join()
    seconds = max(time.monotonic() - started_mono, wall() - started_wall, 0.0)
    return Finished(None if cut else tree.process.returncode, cut, slept, seconds, tail.text())


class _Tail:
    """Reads a pipe to its end on a thread of its own, keeping its last `LOG_TAIL_BYTES`."""

    def __init__(self, pipe):
        self.chunks: deque[bytes] = deque()
        self.size = 0
        self.pipe = pipe
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while chunk := self.pipe.read1(65536):
                self.chunks.append(chunk)
                self.size += len(chunk)
                while self.size - len(self.chunks[0]) >= LOG_TAIL_BYTES:
                    self.size -= len(self.chunks.popleft())
        except (OSError, ValueError):
            pass

    def join(self) -> None:
        self.thread.join(timeout=KILL_WAIT_SECONDS)
        self.pipe.close()

    def text(self) -> str:
        return b"".join(self.chunks)[-LOG_TAIL_BYTES:].decode("utf-8", "replace")
