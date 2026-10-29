"""The far side of a machine place (#394 decision 4): what runs on the test machine.

The local side sends this module, `process` (and the `proctree` it starts commands through),
`environment`, `places` and the `orch_it_ids` plugin as one bundle on the first line of stdin, so
the far machine needs nothing but git, Python and the repo's test dependencies. The second line
is the request (JSON). `main` then takes a free checkout slot, puts its folder at the commit,
runs the command once under its bound in the clean environment of this OS, and prints protocol
lines on stdout, each starting `ORCH-IT `:

    START <json>         the command is starting
    END <json>           rc, cut, slept, seconds, log_tail
    REPORT <base64>      the report's bytes, empty when the command wrote none
    SETUP-FAILED <json>  step and message; the command never started

Slots: the machine's `checkout` setting is slot 0 and `<checkout>-<n>` is slot n, for n below its
capacity. A run holds a slot by creating `<slot>.orch-it-lock` (O_CREAT|O_EXCL, its pid and time
inside) and lets it go at its end; a lock older than the run's bound plus the setup time is stale
and broken. No free slot is `SETUP-FAILED` at step `busy`.

Only folders this module made are touched: a clone it makes gets the marker file
`.git/orch-it-checkout`. An existing folder with files and no marker is `SETUP-FAILED` at step
`not_ours` and is never checked out or cleaned. A marked folder that is no repository of its own
any more (git finds none, or the one around it) is removed and cloned again. In a slot the commit
is fetched (or the folder cloned), `checkout --force --detach` puts it there and `clean -fd` removes
untracked files; ignored files stay, since dependencies may live in ignored folders.

Stdin stays open for the whole run: it is a dead man's switch. When it reaches EOF (the caller
or the connection is gone) the test process tree is killed and `main` exits without a word.
Stdlib only, so it needs no install here.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from orch_it.testruns import places
from orch_it.testruns.places import LOG_TAIL_CHARS

PREFIX = "ORCH-IT "
SETUP_STEP_SECONDS = 600
LOCK_SUFFIX = ".orch-it-lock"
MARKER = "orch-it-checkout"  # in the clone's `.git`
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class SetupFailed(Exception):
    def __init__(self, step: str, message: str):
        self.step = step
        super().__init__(message)


def say(kind: str, payload: str) -> None:
    sys.stdout.write(f"{PREFIX}{kind} {payload}\n")
    sys.stdout.flush()


def main(workdir: str) -> None:
    """Serve one request read from stdin, then exit the process (never returns)."""

    gone = threading.Event()
    code = 0
    lock = None
    try:
        request = json.loads(sys.stdin.readline())
        threading.Thread(target=_watch_stdin, args=(gone,), daemon=True).start()
        try:
            slot, lock = _take_slot(request)
            checkout = _setup(request, slot, gone)
        except SetupFailed as e:
            say("SETUP-FAILED", json.dumps({"step": e.step, "message": str(e)[-LOG_TAIL_CHARS:]}))
        else:
            if not gone.is_set():
                _run(request, Path(workdir), checkout, gone)
    except BaseException:
        code = 1
        traceback.print_exc()
    finally:
        if lock:
            _unlink(lock)
        shutil.rmtree(workdir, ignore_errors=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # a thread blocked on stdin must not hold the interpreter's shutdown


def _watch_stdin(gone: threading.Event) -> None:
    try:
        while sys.stdin.readline():
            pass
    except (OSError, ValueError):
        pass
    gone.set()


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _take_slot(request: dict) -> tuple[str, str]:
    """(slot folder, its lock file) of the first slot nobody holds."""

    base = os.path.abspath(os.path.expanduser(request["checkout"]))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for n in range(request["capacity"]):
        slot = base if n == 0 else f"{base}-{n}"
        if _lock(slot + LOCK_SUFFIX, request["stale_seconds"]):
            return slot, slot + LOCK_SUFFIX
    raise SetupFailed("busy", f"all {request['capacity']} slot(s) of {base} are in use")


def _lock(path: str, stale_seconds: float) -> bool:
    for _ in range(2):  # the second try follows breaking a stale lock
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):  # Windows: a create racing a removal of the same name is refused
            try:
                if time.time() - os.path.getmtime(path) <= stale_seconds:
                    return False
            except OSError:
                continue  # it went while we looked
            _unlink(path)
            continue
        with os.fdopen(handle, "w") as lock:
            lock.write(json.dumps({"pid": os.getpid(), "at": time.time()}))
        return True
    return False


def _git(args: list[str], *, cwd: str | None, step: str) -> str:
    env = os.environ | {"GIT_TERMINAL_PROMPT": "0"}
    try:
        # stdin is ours (the dead man's switch): git gets none of it
        done = subprocess.run(["git", *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, encoding="utf-8", errors="replace",
                              timeout=SETUP_STEP_SECONDS, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SetupFailed(step, f"git {args[0]}: {e}") from None
    if done.returncode != 0:
        raise SetupFailed(step, f"git {args[0]} failed ({done.returncode}): "
                                f"{done.stderr.strip() or done.stdout.strip()}")
    return done.stdout.strip()


def _same_folder(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _own_repository(slot: str) -> bool:
    """Is `slot` a repository of its own (git finds this folder, not one around it)?"""

    try:
        return _same_folder(_git(["rev-parse", "--show-toplevel"], cwd=slot, step="fetch"), slot)
    except SetupFailed:
        return False


def _remove(slot: str) -> None:
    def writable(function, path, _error):  # read-only files of .git on Windows
        os.chmod(path, 0o700)
        function(path)

    if sys.version_info >= (3, 12):
        shutil.rmtree(slot, onexc=writable)
    else:
        shutil.rmtree(slot, onerror=writable)


def _clone(slot: str, origin: str, identity: dict | None = None) -> None:
    if not origin:
        raise SetupFailed("clone", "the folder is no clone and no origin url is known")
    _git(["clone", "--no-checkout", "--", origin, slot], cwd=None, step="clone")
    if identity:
        _set_identity(slot, identity, "clone")
    Path(slot, ".git", MARKER).write_text("made by orch-it\n", "utf-8")


def _set_identity(slot: str, identity: dict, step: str) -> None:
    """Set `user.name` and `user.email` in the checkout's own config only: the identity its commits
    carry, written on a fresh clone and on every fetch of a reused slot."""

    _git(["config", "--local", "user.name", identity["name"]], cwd=slot, step=step)
    _git(["config", "--local", "user.email", identity["email"]], cwd=slot, step=step)


def _setup(request: dict, slot: str, gone: threading.Event) -> str:
    """Put `slot` at the requested commit; return the folder."""

    commit = request["commit"]
    if not _COMMIT.fullmatch(commit):
        raise SetupFailed("commit", f"not a commit id: {commit!r}")
    if os.path.isdir(slot) and os.listdir(slot):
        if not os.path.isfile(os.path.join(slot, ".git", MARKER)):
            raise SetupFailed("not_ours", f"{slot} exists and is not a checkout orch-it made")
        if not _own_repository(slot):
            _remove(slot)
    if not os.path.isdir(slot) or not os.listdir(slot):
        _clone(slot, request.get("origin"), request.get("identity"))
    else:
        try:
            _git(["fetch", "origin"], cwd=slot, step="fetch")
        except SetupFailed as e:
            if "not a git repository" not in str(e):
                raise
            _remove(slot)
            _clone(slot, request.get("origin"), request.get("identity"))
        else:
            if request.get("identity"):  # the identity asked for now, not the one of the earlier run
                _set_identity(slot, request["identity"], "fetch")
    if gone.is_set():
        return slot
    try:
        _git(["cat-file", "-e", commit + "^{commit}"], cwd=slot, step="commit")
    except SetupFailed:
        _git(["fetch", "origin", commit], cwd=slot, step="commit")  # a commit no branch holds
        _git(["cat-file", "-e", commit + "^{commit}"], cwd=slot, step="commit")
    _git(["checkout", "--force", "--detach", commit], cwd=slot, step="checkout")
    _git(["clean", "-fd"], cwd=slot, step="clean")
    return slot


def _run(request: dict, workdir: Path, checkout: str, gone: threading.Event) -> None:
    asked = places.Request(Path(checkout), request["commit"], tuple(request["command"]),
                           request["test_env"], request["bound_seconds"], request.get("ids"))
    done = places.run_command(
        asked, Path(checkout), workdir, abort=gone,
        on_start=lambda: say("START", json.dumps({"checkout": checkout, "python": sys.executable})))
    if gone.is_set():
        return
    say("END", json.dumps({"rc": done.rc, "cut": done.cut, "slept": done.slept,
                           "seconds": done.run_seconds, "log_tail": done.log_tail}))
    say("REPORT", base64.b64encode(done.report.read_bytes()).decode("ascii") if done.report else "")
