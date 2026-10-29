"""Which processes a session owns, and stopping an own one with its whole tree (#395 decision 3).

A process is own when one of three facts proves it: its environment carries this session's id
(the setting `session_env_var`, default `CLAUDE_CODE_SESSION_ID`, which Claude Code gives every
shell and child of a session), or its parent chain — each parent older than its child, so a
reused pid never counts — reaches the session's agent process, or the store recorded a `process`
this session started with that pid and start time. Nothing proven is foreign.

The ownership rule is pure over a `Facts` reading of the process tree; the psutil reading is the
one adapter behind `Facts`, used on Windows, Linux and macOS (#378). Stopping ends an own
process and its whole tree, children first, so no wrapper is left holding a pipe.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_SESSION_ENV_VAR = "CLAUDE_CODE_SESSION_ID"
# a parent chain longer than this is not walked further (a cycle or a pathological tree)
MAX_CHAIN = 200
# a process that exists but whose facts cannot be read: ownership cannot be proven, so a stop of
# it is refused as unresolved rather than passed as "nothing to stop" (never confused with gone)
UNREADABLE = "unreadable"
# a name or command-line match is run under this wall-clock bound, so a crafted pattern cannot
# hang the hook on catastrophic backtracking; a match that overruns is unresolved
MATCH_SECONDS = 2.0
MAX_PATTERN = 1000


@dataclass(frozen=True)
class Proc:
    """One process as the ownership rule needs it: its pid, its start time (the clock the OS
    stamps it with), its parent's pid, and this session's id read from its environment (None
    when absent or unreadable)."""

    pid: int
    create_time: float
    ppid: int | None
    env_session: str | None


def is_own(pid: int, *, facts, session_id: str | None, agent_pid: int | None,
           recorded: set[tuple[int, float]]) -> bool:
    """Whether the live process `pid` belongs to this session. `facts.get(pid)` returns a
    `Proc` or None. `recorded` is the (pid, start time) of every `process` this session started,
    read from the store. A pid the facts cannot read is foreign (nothing proven)."""

    start = facts.get(pid)
    if start is None or start is UNREADABLE:
        return False  # gone, or its facts cannot be read: nothing proves it own
    if session_id and start.env_session == session_id:
        return True
    if (pid, start.create_time) in recorded:
        return True
    return _chain_reaches(start, facts, agent_pid)


def _chain_reaches(start: Proc, facts, agent_pid: int | None) -> bool:
    """Whether `start`'s parent chain reaches `agent_pid`, each parent older than its child so a
    reused pid breaks the link; stops on a cycle or a missing parent."""

    if agent_pid is None:
        return False
    current = start
    seen: set[int] = set()
    for _ in range(MAX_CHAIN):
        if current.pid == agent_pid:
            return True
        if current.ppid is None or current.pid in seen:
            return False
        seen.add(current.pid)
        parent = facts.get(current.ppid)
        if parent is None or parent is UNREADABLE or parent.create_time > current.create_time:
            return False  # no parent, unreadable, or a reused pid stamped later than its child
        current = parent
    return False


# --- judging a stop command ------------------------------------------------------------------

STOP_REFUSALS = ("foreign_process", "unresolved")


STOP_REFUSALS = ("foreign_process", "unresolved")


@dataclass(frozen=True)
class StopRefusal:
    """Why a stop command is refused. `reason` (one of `STOP_REFUSALS`): `foreign_process` (a
    process it would stop is not own), `unresolved` (a target the guard could not read, so
    ownership cannot be proven); `pid`: the first foreign process, or None; `rule_id`: the stop
    that triggered."""

    reason: str
    pid: int | None
    rule_id: str

    def __post_init__(self):
        if self.reason not in STOP_REFUSALS:
            raise ValueError(f"not a stop refusal: {self.reason}")


def judge_stop(stop, *, facts, session_id: str | None, agent_pid: int | None,
               recorded: set[tuple[int, float]]) -> StopRefusal | None:
    """None when every process the stop would end is own; a `StopRefusal` when any is foreign or
    a target cannot be read. A name or pattern target is resolved against the live process list,
    and one foreign match refuses the whole command; a tree stop judges every descendant too.
    A target that matches no live process is nothing to stop, not a refusal; a target that
    exists but whose facts cannot be read is `unresolved`, never silently passed."""

    if getattr(stop, "opaque", False):
        return StopRefusal("unresolved", None, stop.rule_id)
    targets: set[int] = set()
    for pid in stop.pids:
        info = facts.get(pid)
        if info is UNREADABLE:
            return StopRefusal("unresolved", pid, stop.rule_id)  # exists but unprovable: refuse
        if info is not None:
            targets.add(pid)
    for matcher in stop.matchers:
        matched = _matching_pids(matcher, facts)
        if matched is UNREADABLE:
            return StopRefusal("unresolved", None, stop.rule_id)
        targets |= matched
    if getattr(stop, "tree", False):
        targets = _with_descendants(targets, facts)
    for pid in sorted(targets):
        if not is_own(pid, facts=facts, session_id=session_id, agent_pid=agent_pid,
                      recorded=recorded):
            return StopRefusal("foreign_process", pid, stop.rule_id)
    return None


def _matching_pids(matcher, facts):
    """The live pids a name or pattern target matches, or `UNREADABLE` when the match could not
    be run safely (the live list unreadable, an unknown matcher kind, a pattern that is invalid
    or runs past its time bound): the caller then refuses, never passes."""

    import fnmatch

    if matcher.kind not in ("name_glob", "name_regex", "cmdline_regex"):
        return UNREADABLE
    try:
        live = facts.live_pids()
    except Exception:
        return UNREADABLE
    out: set[int] = set()
    for pid in live:
        if matcher.kind == "name_glob":
            name = facts.name(pid)
            if name is None:
                continue
            hit = _name_matches(name, matcher.text, matcher.exact)
        else:
            text = (facts.cmdline(pid) if matcher.kind == "cmdline_regex" else facts.name(pid)) or ""
            hit = _regex_hit(matcher.text, text, matcher.exact)
            if hit is UNREADABLE:
                return UNREADABLE
        if hit:
            out.add(pid)
    return out


MAX_MATCH_TEXT = 8192
# a quantified group whose own body also quantifies — `(a+)+`, `(.*)*`, `(x+){2,}` — is the
# shape of catastrophic backtracking; such a pattern is refused unrun, since CPython's `re`
# holds the GIL and cannot be interrupted once started
_REDOS = __import__("re").compile(r"\([^()]*[*+][^()]*\)\s*[*+{]")


_EXE_SUFFIXES = (".exe", ".com", ".bat", ".cmd")


def _name_matches(name: str, pattern: str, exact: bool) -> bool:
    """A process name matched against a kill target's name (case-insensitive). The target may
    omit the executable extension (`Stop-Process -Name explorer` for `explorer.exe`), so the
    name's bare stem is tried too."""

    import fnmatch

    candidates = {name.casefold()}
    lower = name.casefold()
    for suffix in _EXE_SUFFIXES:
        if lower.endswith(suffix):
            candidates.add(lower[: -len(suffix)])
    pat = pattern.casefold()
    if exact:
        return pat in candidates
    return any(fnmatch.fnmatch(c, pat) for c in candidates)


def _regex_hit(pattern: str, text: str, exact: bool):
    """Whether `pattern` matches `text`; `UNREADABLE` when the pattern is too long, is invalid,
    or carries a nested-quantifier shape that could backtrack catastrophically (then the stop
    is refused rather than run). The text is capped so a huge command line cannot slow a safe
    pattern."""

    import re

    if len(pattern) > MAX_PATTERN or _REDOS.search(pattern):
        return UNREADABLE
    try:
        compiled = re.compile((r"\A(?:" + pattern + r")\Z") if exact else pattern)
    except re.error:
        return UNREADABLE  # a pattern Python will not compile: the real tool might still match
    try:
        return bool(compiled.search(text[:MAX_MATCH_TEXT]))
    except (re.error, RuntimeError):
        return UNREADABLE


def _with_descendants(pids: set[int], facts) -> set[int]:
    """`pids` plus every live descendant of each, read from the live tree's parent links (a
    light read, no environment)."""

    try:
        live = facts.live_pids()
    except Exception:
        return set(pids)
    children: dict[int, list[int]] = {}
    for pid in live:
        ppid = facts.ppid(pid)
        if ppid is not None:
            children.setdefault(ppid, []).append(pid)
    out: set[int] = set()
    stack = list(pids)
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        stack += children.get(pid, [])
    return out


# --- the psutil reading ----------------------------------------------------------------------

class PsutilFacts:
    """The one adapter behind `Facts`: the live process tree read through psutil (#378). `get`
    returns a `Proc`, None when the process is gone, or `UNREADABLE` when it exists but its
    start time cannot be read (so ownership cannot be proven and a stop of it is refused, never
    silently passed). `ppid` is a light read with no environment, for building the tree cheaply."""

    def __init__(self, session_env_var: str = DEFAULT_SESSION_ENV_VAR):
        import psutil

        self._psutil = psutil
        self._var = session_env_var

    def get(self, pid: int):
        psutil = self._psutil
        try:
            p = psutil.Process(pid)
        except (psutil.NoSuchProcess, ValueError):
            return None
        except (psutil.Error, OSError):
            return UNREADABLE
        try:
            with p.oneshot():
                create_time = p.create_time()
                try:
                    ppid = p.ppid()
                except (psutil.Error, OSError):
                    ppid = None
        except psutil.NoSuchProcess:
            return None
        except (psutil.Error, OSError, ValueError):
            return UNREADABLE  # exists, but we cannot stamp it: unprovable, not absent
        return Proc(pid=pid, create_time=create_time, ppid=ppid,
                    env_session=self._env_session(p))

    def ppid(self, pid: int) -> int | None:
        try:
            return self._psutil.Process(pid).ppid()
        except (self._psutil.Error, OSError, ValueError):
            return None

    def _env_session(self, p) -> str | None:
        try:
            return p.environ().get(self._var)
        except (self._psutil.Error, OSError, ValueError, NotImplementedError):
            return None  # another user's environment, or a platform that will not give it

    def live_pids(self) -> list[int]:
        return self._psutil.pids()

    def name(self, pid: int) -> str | None:
        try:
            return self._psutil.Process(pid).name()
        except (self._psutil.Error, OSError, ValueError):
            return None

    def cmdline(self, pid: int) -> str | None:
        try:
            return " ".join(self._psutil.Process(pid).cmdline())
        except (self._psutil.Error, OSError, ValueError):
            return None


KILL_WAIT_SECONDS = 10.0


def stop_tree(pid: int, *, timeout: float = KILL_WAIT_SECONDS) -> int:
    """End the process `pid` and its whole tree, children first so a wrapper is never left
    holding the pipes of a child that outlives it (P521, P905). Terminate, wait, then kill any
    survivor; returns how many of the tree are confirmed gone. Refuses to act on its own
    process. A process already gone is no error."""

    import os

    import psutil

    if pid == os.getpid():
        return 0  # never stop the process doing the stopping
    try:
        root = psutil.Process(pid)
    except (psutil.Error, OSError, ValueError):
        return 0
    try:
        descendants = root.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    # deepest first: a child is asked to end before its parent
    ordered = sorted(descendants, key=_depth, reverse=True) + [root]
    for proc in ordered:
        try:
            proc.terminate()
        except (psutil.Error, OSError):
            pass
    psutil.wait_procs(ordered, timeout=timeout)
    for proc in ordered:
        try:
            proc.kill()
        except (psutil.Error, OSError):
            pass
    gone, _ = psutil.wait_procs(ordered, timeout=timeout)
    return len(gone)  # only those confirmed ended, never a blind count


def _depth(proc) -> int:
    import psutil

    depth = 0
    seen: set[int] = set()
    try:
        while proc.ppid() and depth < MAX_CHAIN and proc.pid not in seen:
            seen.add(proc.pid)
            proc = psutil.Process(proc.ppid())
            depth += 1
    except (psutil.Error, OSError):
        pass
    return depth


def this_agent_pid(facts=None, session_id: str | None = None) -> int:
    """The pid of the session's agent process: walking up from this process while the session
    id is still in a parent's environment, the topmost such ancestor, so a worker's own shells
    and children resolve to it. Falls back to this process's pid when nothing can be read."""

    import os

    pid = os.getpid()
    if facts is None or not session_id:
        return pid
    current = facts.get(pid)
    agent = pid
    seen: set[int] = set()
    for _ in range(MAX_CHAIN):
        if not isinstance(current, Proc) or current.ppid is None or current.pid in seen:
            break
        seen.add(current.pid)
        parent = facts.get(current.ppid)
        if not isinstance(parent, Proc) or parent.create_time > current.create_time:
            break
        if parent.env_session != session_id:
            break  # the parent is outside the session: `current` is the session's top process
        agent = parent.pid
        current = parent
    return agent
