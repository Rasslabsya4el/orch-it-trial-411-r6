"""A machine place (#394 decision 4): one run of the test command on another machine over ssh.

One `run` is one ssh connection to the machine. Its first stdin line is Python source that
unpacks a small bundle of this package's stdlib-only modules (see `remote`) and starts `remote.main`;
the second is the request. The far side takes a checkout slot, puts the commit there, runs the
command and reports on stdout in `ORCH-IT` lines; the caller's work tree is never touched. Stdin is
kept open until the run ends: closing it is how the far side learns the caller is gone.

How a run ends. The report arrives, and `run` returns it. `PlaceRefused` (with a step saying
where): the machine has no slot free (code `busy`); or, code `place_off`, ssh cannot be started,
cannot connect or ends before the command started, the far setup fails (clone, fetch, unknown
commit, a folder that is not the layer's own), or the command has not started within
`setup_seconds` of the connection. `PlaceVanished`, with ssh killed: ssh ends or the connection
drops after the command started and before its report arrived; the report does not arrive within
the run's bound plus `run_slack_seconds` of the START line; a REPORT or END line cannot be decoded
or a line is over `MAX_LINE_BYTES`. A line is used only when complete (ending in a newline), and
is found after whatever precedes its `ORCH-IT ` prefix on the line (a banner without a newline).
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import queue
import subprocess
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from orch_it import identity
from orch_it.testruns.config import Machine
from orch_it.testruns.places import (LOG_TAIL_CHARS, REFUSAL_STEPS, PlaceEnd, PlaceRefused, PlaceVanished,
                                     Request)
from orch_it.testruns.remote import PREFIX as PROTOCOL_PREFIX

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PREFIX = PROTOCOL_PREFIX.encode("ascii")
FOLDER = Path(__file__).parent
BUNDLE = ("environment.py", "places.py", "process.py", "remote.py", "pytest_plugin/__init__.py",
          "pytest_plugin/orch_it_ids.py")
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=4")
REMOTE_PROGRAM = "import sys;exec(sys.stdin.readline())"
SETUP_SECONDS = 600.0  # from the connection to the START line: connect, clone or fetch, checkout
RUN_SLACK_SECONDS = 120.0  # on top of the run's bound, from the START line to the report
MAX_LINE_BYTES = 256 * 1024 * 1024
STDERR_TAIL_BYTES = 2000
GRACE_SECONDS = 5.0  # ssh's time to go by itself once its stdin is closed, before it is killed
TOO_LONG = object()


def bootstrap_line() -> bytes:
    """The first stdin line: Python source that unpacks the bundle into a temp folder, puts it
    first on `sys.path` and calls `remote.main` with that folder."""

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("orch_it/__init__.py", "")
        bundle.writestr("orch_it/testruns/__init__.py", "")
        bundle.writestr("orch_it/proctree.py", (FOLDER.parent / "proctree.py").read_bytes())
        for name in BUNDLE:
            bundle.writestr(f"orch_it/testruns/{name}", (FOLDER / name).read_bytes())
    packed = base64.b64encode(archive.getvalue()).decode("ascii")
    return (f"import base64,io,sys,tempfile,zipfile;d=tempfile.mkdtemp(prefix='orch-it-');"
            f"zipfile.ZipFile(io.BytesIO(base64.b64decode('{packed}'))).extractall(d);"
            f"sys.path.insert(0,d);from orch_it.testruns.remote import main;main(d)\n").encode("ascii")


def without_credentials(url: str) -> str:
    """An http(s) url without its `user:password@`; any other url as it is."""

    parts = urlsplit(url)
    if parts.scheme in ("http", "https") and "@" in parts.netloc:
        return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))
    return url


def _identity_of(origin: str) -> dict | None:
    """The identity the far clone's commits carry; None when gh cannot tell it (the run goes on
    with git's own identity there)."""

    try:
        return identity.for_origin(origin)
    except identity.IdentityError:
        return None


@dataclass
class MachinePlace:
    machine: Machine
    origin_url: str | None = None  # the caller's origin; the machine's own `origin` setting wins
    ssh: tuple[str, ...] = ("ssh",)
    setup_seconds: float = SETUP_SECONDS
    run_slack_seconds: float = RUN_SLACK_SECONDS
    id: str = field(init=False)
    os: str = field(init=False)

    def __post_init__(self):
        self.id = self.machine.place
        self.os = self.machine.os

    def remote_command(self) -> str:
        """The one string the far shell gets. Double quotes around the program work in a POSIX
        shell and in Windows cmd.exe alike. cmd.exe, run as `cmd /c <string>`, takes off the
        outer pair of quotes of a string that starts with one, so the string of a Windows machine
        gets one more pair around it."""

        python = self.machine.python or ("python" if self.machine.os == "windows" else "python3")
        if any(c.isspace() for c in python):
            python = f'"{python}"'
        command = f'{python} -c "{REMOTE_PROGRAM}"'
        return f'"{command}"' if self.machine.os == "windows" else command

    def run(self, request: Request, scratch: Path) -> PlaceEnd:
        if not request.commit:
            raise PlaceRefused("place_off", "the work tree has no commit to check out on the machine",
                               "commit")
        origin = self.machine.origin or (without_credentials(self.origin_url) if self.origin_url else None)
        if not origin:
            raise PlaceRefused("place_off", "no origin url to clone the machine's checkout from", "clone")
        request_line = self._request_line(origin, request)  # built before ssh starts
        argv = [*self.ssh, *self.machine.ssh_options, *SSH_OPTIONS, "--", self.machine.host,
                self.remote_command()]
        started = time.monotonic()
        try:
            ssh = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, creationflags=NO_WINDOW)
        except (OSError, ValueError) as e:
            raise PlaceRefused("place_off", f"ssh could not be started: {e}", "connect") from None
        lines: queue.Queue = queue.Queue()
        stderr: deque[bytes] = deque()
        threading.Thread(target=_read_lines, args=(ssh.stdout, lines), daemon=True).start()
        threading.Thread(target=_keep_tail, args=(ssh.stderr, stderr), daemon=True).start()
        sent = threading.Event()
        threading.Thread(target=_send, args=(ssh.stdin, [bootstrap_line(), request_line], sent),
                         daemon=True).start()
        try:
            return self._read(lines, stderr, ssh, scratch, started, request.bound_seconds)
        finally:
            self._close(ssh, sent)

    def _request_line(self, origin: str, request: Request) -> bytes:
        return (json.dumps({
            "checkout": self.machine.checkout, "capacity": self.machine.capacity, "origin": origin,
            "commit": request.commit, "command": list(request.command), "test_env": request.test_env,
            "bound_seconds": request.bound_seconds,
            "stale_seconds": request.bound_seconds + self.setup_seconds,
            "ids": request.ids, "identity": _identity_of(origin)}) + "\n").encode("utf-8")

    def _read(self, lines, stderr, ssh, scratch, started, bound_seconds) -> PlaceEnd:
        host = self.machine.host
        start = end = None
        deadline = started + self.setup_seconds
        while True:
            try:
                line = lines.get(timeout=max(deadline - time.monotonic(), 0.0))
            except queue.Empty:
                if start is None:
                    raise PlaceRefused("place_off", f"{host}: the run did not start within "
                                       f"{self.setup_seconds:g} s{_stderr_note(stderr)}",
                                       "start_timeout") from None
                raise PlaceVanished(f"{host}: no report within the run's bound") from None
            if line is TOO_LONG:
                raise PlaceVanished(f"{host}: a line over {MAX_LINE_BYTES} bytes")
            if line is None:
                if start is None:
                    raise PlaceRefused("place_off", f"{host}: ssh ended ({_exit_of(ssh)}) before the run "
                                       f"started{_stderr_note(stderr)}", "connect")
                raise PlaceVanished(f"{host}: the connection ended mid-run{_stderr_note(stderr)}")
            kind, _, payload = line.partition(b" ")
            if kind == b"SETUP-FAILED":
                failed = _json_or_empty(payload)
                step = failed.get("step")
                message = f"{host}: setup failed at {step}: {failed.get('message')}"
                if step == "busy":
                    raise PlaceRefused("busy", message, "busy")
                raise PlaceRefused("place_off", message, step if step in REFUSAL_STEPS else "connect")
            if kind == b"START":
                start = time.monotonic() - started
                deadline = time.monotonic() + bound_seconds + self.run_slack_seconds
            elif kind == b"END" and start is not None:
                try:
                    end = json.loads(payload)
                except ValueError:
                    end = None
                if not isinstance(end, dict):
                    raise PlaceVanished(f"{host}: an END line that cannot be read")
            elif kind == b"REPORT" and end is not None:
                return self._place_end(payload, end, start, scratch)

    def _place_end(self, payload: bytes, end: dict, start: float, scratch: Path) -> PlaceEnd:
        report = None
        if payload.strip():
            try:
                packed = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError):
                raise PlaceVanished(f"{self.machine.host}: a REPORT line that cannot be read") from None
            report = scratch / "junit.xml"
            report.write_bytes(packed)
        return PlaceEnd(report, bool(end.get("cut")), bool(end.get("slept")), end.get("rc"), start,
                        float(end.get("seconds") or 0.0), str(end.get("log_tail") or "")[-LOG_TAIL_CHARS:])

    @staticmethod
    def _close(ssh: subprocess.Popen, sent: threading.Event) -> None:
        """End the connection: closing stdin tells a live far side to kill its test tree; ssh is
        killed when it does not go by itself. A sender still blocked on a pipe nobody reads holds
        the pipe's lock: then ssh is killed at once."""

        if sent.is_set():
            try:
                ssh.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                ssh.wait(timeout=GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        if ssh.poll() is None:
            ssh.kill()
        ssh.wait()


def _json_or_empty(payload: bytes) -> dict:
    try:
        value = json.loads(payload)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _exit_of(ssh: subprocess.Popen) -> int | None:
    try:
        return ssh.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return None


def _stderr_note(stderr: deque[bytes]) -> str:
    """What ssh said on stderr, as `": <text>"` to append to a message, or nothing."""

    time.sleep(0.2)  # let the stderr reader catch up with an ssh that has just ended
    text = b"".join(list(stderr))[-STDERR_TAIL_BYTES:].decode("utf-8", "replace").strip()
    return f": {text}" if text else ""


def _send(pipe, chunks: list[bytes], sent: threading.Event) -> None:
    try:
        for chunk in chunks:
            pipe.write(chunk)
            pipe.flush()
    except (OSError, ValueError):
        pass
    sent.set()


def _read_lines(pipe, lines: queue.Queue) -> None:
    """The protocol lines of ssh's stdout, up to and with their prefix cut off; `TOO_LONG` for a
    line over the cap; `None` at the end. A last line with no newline is dropped: it may be cut."""

    try:
        while raw := pipe.readline(MAX_LINE_BYTES + 1):
            if len(raw) > MAX_LINE_BYTES:
                lines.put(TOO_LONG)
                return
            if not raw.endswith(b"\n"):
                break
            at = raw.find(PREFIX)
            if at >= 0:
                lines.put(raw[at + len(PREFIX):].rstrip(b"\r\n"))
    except (OSError, ValueError):
        pass
    lines.put(None)


def _keep_tail(pipe, kept: deque[bytes]) -> None:
    """Read ssh's stderr to its end, keeping about the last `STDERR_TAIL_BYTES`."""

    size = 0
    try:
        while chunk := pipe.read1(4096):
            kept.append(chunk)
            size += len(chunk)
            while size - len(kept[0]) >= STDERR_TAIL_BYTES:
                size -= len(kept.popleft())
    except (OSError, ValueError):
        pass
