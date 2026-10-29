"""The guard decision the PreToolUse hook makes, and the CLI entry points that wire it (#395
decisions 6, 7). One command becomes a verdict and the events to write: a triggered command is
read into its delete, move and stop actions (unit 2), each action's object is judged (the real
path against the repo's worktrees, unit 3; the real process tree, unit 4), and any refusal
denies the command while every triggered command writes a `guard.fired`. A guard that crashes
lets the command through and writes `error.raised` with `failed_open` true, so one bug never
halts every session (decision 7). The object's private value never reaches an event."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from orch_it.events.schema import EventError
from orch_it.events.store import StoreError
from orch_it.guards import paths, processes, triggers

DEFAULT_REPO = "Rasslabsya4el/orch-it"
# the Claude Code tool names the hook fires for; a PowerShell tool reads the Windows dialect
POWERSHELL_TOOLS = ("PowerShell",)

# the guard a refusal is attributed to, by the action it came from
DELETE_GUARD = "worktree_delete"
STOP_GUARD = "foreign_process_stop"
REFUSAL_WAY = ("the guard reads the object, not the command; to stop an own process with its "
               "tree use `python -m orch_it stop <pid>`")


@dataclass(frozen=True)
class Fired:
    """One `guard.fired` to write: `guard` (a `guard` code), `verdict` (deny/pass), `rule_id`
    (which trigger or finding), `segment` (the command, ≤120), `path` (the protected object or
    None — never a private value)."""

    guard: str
    verdict: str
    rule_id: str
    segment: str
    path: str | None


@dataclass(frozen=True)
class Decision:
    """What the hook does with a command. `deny`: refuse it; `reason`: the refusal text and the
    way (None when allowed); `fired`: the `guard.fired` events to write; `error`: a
    `(component, error_code, failed_open)` to write when a guard crashed, else None."""

    deny: bool
    reason: str | None
    fired: list[Fired] = field(default_factory=list)
    error: tuple[str, str, bool] | None = None


def decide(command: str, *, tool: str, cwd: str, env, windows: bool, session_id: str | None,
           world) -> Decision:
    """The verdict for `command`. `world` answers the object questions: `protected_roots(cwd)`,
    `own_root(cwd, roots)`, `resolve(spelled, cwd)`, `judge_delete(real, roots, own_root)`,
    `judge_stop(stop)`. Never raises: a guard that crashes fails open with an error."""

    try:
        actions = triggers.actions(command, tool=tool, cwd=cwd, env=env, windows=windows)
    except Exception as exc:  # the trigger itself promises not to raise; if it does, fail open
        return Decision(False, None, [], ("trigger", type(exc).__name__, True))

    fired: list[Fired] = []
    deny_reason: str | None = None
    error: tuple[str, str, bool] | None = None
    cwd_roots = None
    own = None
    for action in actions:
        try:
            if isinstance(action, (triggers.Delete, triggers.Move)):
                if cwd_roots is None:
                    cwd_roots = tuple(world.protected_roots(cwd))
                    own = world.own_root(cwd, cwd_roots)
                finding = _judge_delete(action, world, cwd_roots, own)
                verdict = "deny" if finding else "pass"
                fired.append(Fired(DELETE_GUARD, verdict, finding.rule if finding else action.rule_id,
                                   action.segment, finding.path if finding else None))
                if finding and deny_reason is None:
                    deny_reason = f"{DELETE_GUARD}: {finding.rule}; {REFUSAL_WAY}"
            elif isinstance(action, triggers.Stop):
                refusal = world.judge_stop(action)
                verdict = "deny" if refusal else "pass"
                fired.append(Fired(STOP_GUARD, verdict, refusal.reason if refusal else action.rule_id,
                                   action.segment,
                                   str(refusal.pid) if refusal and refusal.pid is not None else None))
                if refusal and deny_reason is None:
                    deny_reason = f"{STOP_GUARD}: {refusal.reason}; {REFUSAL_WAY}"
        except Exception as exc:  # a guard crashed on THIS action: fail open for it, record the
            # crash, but keep any deny already found and judge the rest (a crash never hides a deny)
            guard = DELETE_GUARD if isinstance(action, (triggers.Delete, triggers.Move)) else STOP_GUARD
            if error is None:
                error = (guard, getattr(exc, "code", type(exc).__name__), True)
    return Decision(deny_reason is not None, deny_reason, fired, error)


def _judge_delete(action, world, cwd_roots, own):
    """The first protected object a delete or move reaches, or None. A target the trigger could
    not read (None), or one that does not resolve to a real path, is itself a refusal (an
    unresolved target is never guessed). Each target is judged against the union of the repo
    around cwd and the repo around the target itself, so a delete aimed into another repo from
    outside it is seen (brain comments on #395)."""

    from orch_it.guards.paths import Finding

    is_delete = isinstance(action, triggers.Delete)
    sources = action.targets if is_delete else action.sources
    roots_too = action.roots_too if is_delete else True
    walk_links = action.walk_links if is_delete else False
    # a command whose own cwd we could not locate (a Windows drive `cd` under Git Bash, say) has
    # lost the ground its relative targets stand on: those are unresolved, never guessed
    if action.cwd is not None and not world.cwd_exists(action.cwd):
        if any(s is not None and world.resolve(s, None) is None for s in sources):
            return Finding("unresolved_cwd", "")
    for spelled in sources:
        if spelled is None:
            return Finding("unresolved_target", "")
        real = world.resolve(spelled, action.cwd)
        if real is None:
            return Finding("unresolved_target", "")
        if not world.exists(real):
            # a glob (`<other>/*`) or a not-yet-existing target: judge the nearest existing
            # folder it would delete within; a glob over one's own worktree contents is allowed
            ancestor = world.nearest_dir(real)
            if ancestor is None:
                return Finding("unresolved_target", "")
            if own is not None and _same_path(ancestor, own):
                continue
            real = ancestor
        roots = tuple(dict.fromkeys(tuple(cwd_roots) + tuple(world.roots_around(real))))
        finding = world.judge_delete(real, roots, own, roots_too=roots_too, walk_links=walk_links)
        if finding is not None:
            return finding
    return None


def _same_path(a: str, b: str) -> bool:
    import os

    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


# --- the real world the hook judges against --------------------------------------------------

class World:
    """The object questions answered against the real machine: the repo's worktrees on disk,
    the live process tree. Built once per hook call."""

    def __init__(self, *, windows: bool, session_id: str | None, session_env_var: str):
        self.windows = windows
        self.session_id = session_id
        self._facts = processes.PsutilFacts(session_env_var)
        self._agent = processes.this_agent_pid(self._facts, session_id)
        self._roots_cache: dict = {}
        # fact (c), a store record of (pid, start), is filled when available; the env id and the
        # parent chain prove a session's own processes without it
        self.recorded: set = set()

    def protected_roots(self, cwd):
        return paths.protected_roots(cwd, windows=self.windows)

    def roots_around(self, real_target):
        """The worktrees of the repo the target itself sits in, read at its nearest existing
        folder, so a delete aimed into another repo from a cwd outside it is still judged.
        Cached per folder: a command with many targets does not re-run git each time."""

        here = paths.nearest_existing_dir(real_target)
        if here is None:
            return ()
        if here not in self._roots_cache:
            self._roots_cache[here] = paths.protected_roots(here, windows=self.windows)
        return self._roots_cache[here]

    def exists(self, path):
        import os

        return path is not None and os.path.lexists(path)

    def nearest_dir(self, path):
        return paths.nearest_existing_dir(path)

    def cwd_exists(self, cwd):
        import os

        real = paths.resolve(cwd, None, windows=self.windows)
        return bool(real) and os.path.isdir(real)

    def own_root(self, cwd, roots):
        return paths.deepest_root(paths.resolve(cwd, None, windows=self.windows), roots,
                                  windows=self.windows)

    def resolve(self, spelled, cwd):
        return paths.resolve(spelled, cwd, windows=self.windows)

    def judge_delete(self, real, roots, own, *, roots_too=True, walk_links=True):
        return paths.judge(real, roots, windows=self.windows, own_root=own,
                           roots_too=roots_too, walk_links=walk_links)

    def judge_stop(self, stop):
        return processes.judge_stop(stop, facts=self._facts, session_id=self.session_id,
                                    agent_pid=self._agent, recorded=self.recorded)


# --- the PreToolUse entry point --------------------------------------------------------------

_ORIGIN = re.compile(r"(?:github\.com[:/])([A-Za-z0-9-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def run_pre_tool_use(raw_stdin: str, store, *, platform: str = sys.platform) -> tuple[str, int]:
    """Read one PreToolUse hook input, decide, write the events, and return the text to print
    and the exit code. A deny prints the decision JSON that refuses the tool call; an allow
    prints nothing. Never raises: anything unexpected lets the call through (fail open)."""

    try:
        return _run_pre_tool_use(raw_stdin, store, platform)
    except Exception:  # the hook must never block a tool call by raising; fail open
        return "", 0


def _run_pre_tool_use(raw_stdin: str, store, platform: str) -> tuple[str, int]:
    import os

    try:
        payload = json.loads(raw_stdin) if raw_stdin.strip() else {}
    except (ValueError, RecursionError):
        return "", 0  # input we cannot read: do not block the tool
    if not isinstance(payload, dict):
        return "", 0
    tool_name = payload.get("tool_name") if isinstance(payload.get("tool_name"), str) else ""
    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    cwd = payload.get("cwd")
    cwd = cwd if isinstance(cwd, str) and cwd else "."
    session_id = payload.get("session_id")
    session_id = session_id if isinstance(session_id, str) else None
    if not isinstance(command, str) or not command.strip():
        return "", 0  # not a shell tool call this guard reads
    tool = "PowerShell" if tool_name in POWERSHELL_TOOLS else "Bash"
    windows = tool == "PowerShell" or platform == "win32"
    repo = _repo_of(cwd)
    try:
        world = World(windows=windows, session_id=session_id,
                      session_env_var=processes.DEFAULT_SESSION_ENV_VAR)
        decision = decide(command, tool=tool, cwd=cwd, env=dict(os.environ), windows=windows,
                          session_id=session_id, world=world)
    except Exception as exc:  # building the world or judging crashed: fail open, record it
        _safe_write_error(store, "hook", getattr(exc, "code", type(exc).__name__),
                          repo=repo, session=session_id)
        return "", 0
    _write_events(store, decision, repo=repo, session=session_id)
    if decision.deny:
        return _deny_json(decision.reason), 0
    return "", 0


def _repo_of(cwd: str) -> str:
    try:
        import subprocess

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        done = subprocess.run(["git", "-C", cwd, "remote", "get-url", "origin"], capture_output=True,
                              text=True, timeout=paths.GIT_SECONDS, stdin=subprocess.DEVNULL,
                              creationflags=flags)
        found = _ORIGIN.search(done.stdout.strip()) if done.returncode == 0 else None
        if found:
            return found.group(1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return DEFAULT_REPO


def _deny_json(reason: str | None) -> str:
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": reason or "refused by the orch-it guard"}})


def _write_events(store, decision: Decision, *, repo: str, session: str | None) -> None:
    try:
        with store.writer() as writer:
            for f in decision.fired:
                writer.write(_fired_event(f, repo, session))
            if decision.error is not None:
                writer.write(_error_event(decision.error, repo, session))
    except (OSError, StoreError, EventError) as e:
        # the store must never block a tool call; a lost event beats a stuck agent, but it is
        # said once on stderr so a validation slip does not vanish silently
        print(f"orch-it guard: event not written: {e}", file=sys.stderr)


def _envelope(kind: str, repo: str, session: str | None, detail: dict) -> dict:
    """The full envelope an event needs beyond what the writer fills (id, at, machine, os,
    layer_version): every envelope field present, with the ones that do not apply set null."""

    return {"kind": kind, "repo": repo, "ticket": None, "role": "hook", "session": session,
            "parent_session": None, "turn": None, "step": None, "run": None, "note": None,
            "detail": detail}


def _fired_event(f: Fired, repo: str, session: str | None) -> dict:
    return _envelope("guard.fired", repo, session,
                     {"guard": f.guard, "verdict": f.verdict, "segment": f.segment[:120],
                      "rule_id": f.rule_id, "path": f.path})


def _error_event(error, repo: str, session: str | None) -> dict:
    component, code, failed_open = error
    return _envelope("error.raised", repo, session,
                     {"component": component, "error_code": code, "failed_open": failed_open})


def _safe_write_error(store, component, code, *, repo, session) -> None:
    try:
        with store.writer() as writer:
            writer.write(_error_event((component, code, True), repo, session))
    except (OSError, StoreError, EventError):
        pass


# --- a false fire, and the counts ------------------------------------------------------------

REVERSE_ERRORS = ("no_such_fire", "not_a_deny", "already_reversed")


class ReverseError(Exception):
    """A `guards reverse` that names no reversible fire. `code` is one of `REVERSE_ERRORS`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in REVERSE_ERRORS:
            raise ValueError(f"not a reverse error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def reverse(store, fired_event_id: str, *, by: str) -> dict:
    """Mark a `guard.fired` deny as a false fire (decision 7): write a `guard.reversed` naming
    the fired event and who reversed it, so false fires are counted from the store, never from
    memory. Raises `ReverseError` when the id names no `guard.fired` deny, or one already
    reversed. The reversal keeps the fired event's own repo, so the count stays with it."""

    found = store.query(
        "SELECT guard, verdict, repo FROM events WHERE id = :id AND kind = 'guard.fired'",
        {"id": fired_event_id})
    if not found.rows:
        raise ReverseError("no_such_fire", fired_event_id)
    guard, verdict, repo = found.rows[0]
    if verdict != "deny":
        raise ReverseError("not_a_deny", fired_event_id)
    already = store.query(
        "SELECT 1 FROM events WHERE kind = 'guard.reversed' AND fired_event = :id", {"id": fired_event_id})
    if already.rows:
        raise ReverseError("already_reversed", fired_event_id)
    with store.writer() as writer:
        return writer.write(_envelope("guard.reversed", repo, None,
                                      {"guard": guard, "fired_event": fired_event_id, "by": by}))


# `error.raised` carries no guard code (its component names the part), so it is counted under
# its component; the other kinds under their guard
_GUARD_FIRES_SQL = """
SELECT COALESCE(guard, component) AS guard,
       strftime('%Y-W%W', at) AS week,
       SUM(CASE WHEN kind = 'guard.fired' AND verdict = 'deny' THEN 1 ELSE 0 END) AS denies,
       SUM(CASE WHEN kind = 'guard.fired' AND verdict = 'pass' THEN 1 ELSE 0 END) AS passes,
       SUM(CASE WHEN kind = 'guard.reversed' THEN 1 ELSE 0 END) AS reversals,
       SUM(CASE WHEN kind = 'error.raised' AND failed_open THEN 1 ELSE 0 END) AS failed_open
FROM events
WHERE kind IN ('guard.fired', 'guard.reversed', 'error.raised')
GROUP BY COALESCE(guard, component), week
ORDER BY week, guard
"""


def guard_fires(store):
    """Per guard and per week: denies, passes, reversals and failed-open errors, read from the
    store (the named query `guard-fires`)."""

    return store.query(_GUARD_FIRES_SQL)
