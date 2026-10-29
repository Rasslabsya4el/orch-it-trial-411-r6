"""The report-line hold (#399 decision 6): a Claude Code SubagentStop hook,
`python -m orch_it hook subagent-stop`, that reads the stopping subagent's own transcript
(`agent_transcript_path`; the parent's transcript is never read) and holds its stop once when the
final report breaks a rule of its role's `report` in the roles settings:

- a report line (`tdd:`, `diagnosis:`, `code-review:`, `research:`, `writing-for-agents:`,
  `domain-modeling:`, `codebase-design:`; plain, in bold or code, or a table row `| tdd | called |`)
  that claims a call (any value but a no-claim form such as `skipped (why)`, `no`, `n/a`, `none`,
  `not a defect`) with no Skill tool use of that skill in the transcript that did not fail (with or
  his plugin's `mattpocock-skills:` prefix; a leading `/` aside);
- a unit that edits a code file in its worktree (an Edit, Write, MultiEdit or NotebookEdit that did
  not fail, of a file with a code suffix, `CODE_SUFFIXES`, under the payload's `cwd`) with no `tdd:`
  line; files outside the worktree (the spawn's scratchpad) are no edits of the unit;
- a line the role `owes` (researcher-how's `research:`) that is missing;
- a report of more non-empty lines (split at line breaks) than the role's `cap_lines`;
- a context past the role's `context_line` (the largest prompt of any of its turns: input, cache
  read and cache write tokens, as the transcript's usage records them) with no `table: <path>` line
  naming a file that exists: survey's 100k hand-back, its table written as it goes.

A role with no `report` in the settings (ORCH), a session (a role with no inputs: the brain) and any
other agent type are never read. The final report is `last_assistant_message`, else the last assistant text of the transcript; lines in a
fenced block or a quote are no report lines. A stop already held once (`stop_hook_active` true)
passes, so a subagent is held at most once and never loops. All problems of a stop go into one
reason.

Each judged stop writes one `report_line.checked` per line it judged (`claimed`, `verified`,
`accepted`; `first_edit_path` for `tdd`; for `report_cap` the line count and the cap), a
`context.crossed` when the context line was passed, and, when the stop is let through and the
prompt carried `Spawn: <run>`, the end of that spawn's `subagent` step (outcome `passed`, or
`failed` when a held problem was never fixed). A payload that is no object or a transcript that
cannot be read lets the stop through with `error.raised` (`payload_unreadable`,
`transcript_unreadable`, `failed_open`), as does any error of the hook itself; a store that cannot
be written never changes the verdict. With no GitHub repo around `cwd` there is no envelope, and no
event is written.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from orch_it.roles import SKILL_OF_LINE, RoleSet

CODE_EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
HIS_PLUGIN = "mattpocock-skills"
CODE_SUFFIXES = (".py", ".pyi", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".ps1", ".psm1", ".sh", ".bash", ".html",
                 ".css", ".json", ".sql", ".go", ".rs", ".java", ".kt", ".rb", ".c", ".h", ".cpp", ".hpp", ".cs",
                 ".ipynb")
REASON_HEADER = "The stop is held once; fix the final report, then end again:"
_LABELS = "|".join(map(re.escape, [*SKILL_OF_LINE, "table"]))
_LINE = re.compile(r"^\W*?[*_`]{0,2}(" + _LABELS + r")[*_`]{0,2}\s*:[*_`]{0,2}[ \t]*(.*)$", re.IGNORECASE)
_CELL = re.compile(r"^\s*\|\s*[*_`]{0,2}(" + _LABELS + r")[*_`]{0,2}\s*\|\s*([^|]*)\|", re.IGNORECASE)
_NO_CLAIM = re.compile(r"^(?:skipped|skip|no|n/a|n\.a\.|na|none|nothing|false|-|not a defect|no defect|not applicable"
                       r"|not run|not needed|not required|not relevant|not called|not used|not invoked|did not"
                       r"|didn't)(?:\W|$)",
                       re.IGNORECASE)
_SPAWN = re.compile(r"^Spawn:\s*(spawn-[\w-]+)\s*$", re.MULTILINE)
_FENCE = re.compile(r"^\s*(```+|~~~+)")


@dataclass
class Verdict:
    role: str
    run: str | None = None
    problems: list[tuple[str, str]] = field(default_factory=list)  # (label, its sentence in the reason)
    checks: list[dict] = field(default_factory=list)  # the detail of each `report_line.checked`
    owed: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)
    context_tokens: int = 0
    crossed: bool = False
    table_named: bool = False

    def reason(self) -> str:
        return "\n".join([REASON_HEADER] + [f"- {sentence}" for _, sentence in self.problems])


class TranscriptUnreadable(Exception):
    """The subagent's transcript cannot be read at all: the stop goes through with an event."""


def transcript_records(path: str) -> list[dict]:
    """The transcript's JSON objects; a half-written, broken or too deeply nested line is skipped."""

    records = []
    try:
        with open(path, encoding="utf-8", errors="replace") as lines:
            for line in lines:
                try:
                    record = json.loads(line)
                except (ValueError, RecursionError):
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except (OSError, ValueError) as e:
        raise TranscriptUnreadable(type(e).__name__) from None
    return records


def _blocks(record: dict | None) -> list[dict]:
    message = record.get("message") if isinstance(record, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _tool_uses(records: list[dict]) -> list[dict]:
    """Every tool use of the subagent that did not come back as an error."""

    failed = {b.get("tool_use_id") for r in records if r.get("type") == "user" for b in _blocks(r)
              if b.get("type") == "tool_result" and b.get("is_error") is True}
    return [b for r in records if r.get("type") == "assistant" for b in _blocks(r)
            if b.get("type") == "tool_use" and b.get("id") not in failed]


def _skills(uses: list[dict]) -> set[str]:
    names = set()
    for block in uses:
        given = block.get("input")
        skill = given.get("skill") if block.get("name") == "Skill" and isinstance(given, dict) else None
        if not isinstance(skill, str) or not skill.strip():
            continue
        plugin, _, name = skill.strip().lstrip("/").rpartition(":")
        if plugin in ("", HIS_PLUGIN):  # his copy by its bare name, or his plugin's; another plugin's is not it
            names.add(name.strip().lower())
    return names


def _code_edits(uses: list[dict], worktree: Path | None) -> list[str]:
    paths = []
    for block in uses:
        given = block.get("input")
        path = (given.get("file_path") or given.get("notebook_path")) if isinstance(given, dict) else None
        if block.get("name") in CODE_EDIT_TOOLS and isinstance(path, str) and _is_code(path, worktree):
            paths.append(path)
    return paths


def _is_code(path: str, worktree: Path | None) -> bool:
    """A file with a code suffix in the unit's worktree (anywhere, when the worktree is unknown)."""

    if not path.lower().endswith(CODE_SUFFIXES):
        return False
    if worktree is None:
        return True
    try:
        target = Path(path.replace("\\", "/"))
        target = (target if target.is_absolute() else worktree / target).resolve()
        target.relative_to(worktree.resolve())
    except (ValueError, OSError):
        return False
    return True


def _final(payload: dict, records: list[dict]) -> str:
    message = payload.get("last_assistant_message")
    if isinstance(message, str) and message.strip():
        return message
    last = ""
    for record in records:
        if record.get("type") == "assistant":
            last = next((b["text"] for b in reversed(_blocks(record))
                         if b.get("type") == "text" and isinstance(b.get("text"), str)), last)
    return last


def _lines(text: str) -> list[str]:
    """The lines of the report: split at line breaks only (a form feed or U+2028 is inside a line)."""

    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _report_lines(text: str) -> list[str]:
    """The lines that can be report lines: not inside a fenced block, not a quote. A line that
    opens with a fence and closes it again (inline triple backticks) is a line, not a fence."""

    kept, fence = [], None
    for line in _lines(text):
        opened = _FENCE.match(line)
        if fence is None and opened and opened.group(1)[0] not in line.strip()[len(opened.group(1)):]:
            fence = opened.group(1)
        elif fence is not None:
            fence = None if opened and opened.group(1)[0] == fence[0] else fence
        elif not line.lstrip().startswith(">"):
            kept.append(line)
    return kept


def largest_prompt(records: list[dict]) -> int:
    return max((prompt_tokens(record) for record in records), default=0)


def prompt_tokens(record: object) -> int:
    """The prompt size of one transcript record: input, cache read and cache write tokens of its
    usage (0 for a record that carries none)."""

    message = record.get("message") if isinstance(record, dict) else None
    usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        return 0
    parts = [usage.get(k) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
    return sum(p for p in parts if type(p) is int and p > 0)


def _spawn_run(records: list[dict]) -> str | None:
    first = next((r for r in records if r.get("type") == "user"), None)
    text = "\n".join(str(b.get("text", "")) for b in _blocks(first) if b.get("type") == "text")
    found = _SPAWN.search(text)
    return found.group(1) if found else None


def _claims(lines: list[str]) -> dict[str, list[str]]:
    """Every value written for each report line label, in report order."""

    found: dict[str, list[str]] = {}
    for line in lines:
        match = _CELL.match(line) or _LINE.match(line)
        if match:
            value = match.group(2).strip().strip("*_` ")
            found.setdefault(match.group(1).lower(), []).append(value)
    return found


def _table_path(value: str) -> str:
    """The path of a `table:` value, its markup and a trailing remark (`(12 rows)`) aside."""

    value = value.split(" (", 1)[0].strip()
    return value.strip("*`\"'").strip().removeprefix("<").removesuffix(">").strip()


def judge(payload: dict, roles: RoleSet, *, work_tree: str | Path | None = None) -> Verdict | None:
    """The verdict on a SubagentStop payload, or None for a role with no `report`. A transcript
    that cannot be read raises `TranscriptUnreadable`."""

    role_settings = role_of(payload, roles)
    # a role with no inputs is never spawned (the brain): its `report` holds only its context line
    if role_settings is None or role_settings.report is None or not role_settings.inputs:
        return None
    role = role_settings.name
    path = payload.get("agent_transcript_path")
    if not isinstance(path, str) or not path:
        raise TranscriptUnreadable("no transcript path")
    records = transcript_records(path)
    rule = role_settings.report
    cwd = work_tree or payload.get("cwd")
    worktree = Path(cwd) if isinstance(cwd, (str, Path)) and str(cwd) else None
    final = _final(payload, records)
    uses = _tool_uses(records)
    called = _skills(uses)
    edits = _code_edits(uses, worktree)
    verdict = Verdict(role, _spawn_run(records), context_tokens=largest_prompt(records))
    claims = _claims(_report_lines(final))
    verdict.present = [label for label in claims if label != "table"]
    for label, values in claims.items():
        if label == "table" or all(_NO_CLAIM.match(value) for value in values):
            continue
        skill = SKILL_OF_LINE[label]
        backed = skill in called
        verdict.checks.append({"label": label, "claimed": True, "verified": backed, "accepted": backed})
        if not backed:
            verdict.problems.append((label, f"`{label}:` claims the skill `{skill}`, and this subagent never called it."
                                            f" Call the skill `{skill}` now, or write the line as"
                                            f" `{label}: skipped (why)`."))
    if edits:
        verdict.owed.append("tdd")
        if "tdd" not in claims:
            verdict.checks.append({"label": "tdd", "claimed": False, "verified": "tdd" in called, "accepted": False,
                                   "first_edit_path": edits[0]})
            verdict.problems.append(("tdd", f"this unit edits code (`{edits[0]}`) and the report has no `tdd:` line."
                                            " Call the skill `tdd` and report it, or end with `tdd: skipped (why)`."))
    for label in rule.owes:
        verdict.owed.append(label)
        if label not in claims:
            verdict.checks.append({"label": label, "claimed": False, "verified": SKILL_OF_LINE[label] in called,
                                   "accepted": False})
            verdict.problems.append((label, f"your role owes the line `{label}:` and the report has none."))
    count = sum(1 for line in _lines(final) if line.strip())
    if rule.cap_lines is not None and count > rule.cap_lines:
        verdict.checks.append({"label": "report_cap", "claimed": count, "verified": rule.cap_lines, "accepted": False})
        verdict.problems.append(("report_cap", f"the report has {count} lines; your role's cap is {rule.cap_lines}."
                                               " Keep the lines that carry a fact and its source."))
    if rule.context_line is not None and verdict.context_tokens > rule.context_line:
        verdict.crossed = True
        table = _table_path(claims.get("table", [""])[-1])
        base = worktree or Path(".")
        verdict.table_named = bool(table) and (Path(table) if Path(table).is_absolute() else base / table).is_file()
        verdict.checks.append({"label": "table", "claimed": bool(table), "verified": verdict.table_named,
                               "accepted": verdict.table_named})
        if not verdict.table_named:
            verdict.problems.append(("table", f"your context passed {rule.context_line} tokens: hand back now with the"
                                              " line `table: <path>` naming the table file written so far."))
    return verdict


def _envelope(payload: dict | None, repo: str, role: str, run: str | None, step: str | None = None) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    session = payload.get("agent_id") if isinstance(payload.get("agent_id"), str) else None
    parent = payload.get("session_id") if isinstance(payload.get("session_id"), str) else None
    return {"repo": repo, "ticket": None, "role": role, "session": session, "parent_session": parent, "turn": None,
            "step": step, "run": run, "note": None}


def record(verdict: Verdict, payload: dict, *, store, repo: str, let_through: bool) -> None:
    """The events of a judged stop: its checks, a crossed context line, and the spawn's end."""

    with store.writer() as writer:
        _record_with(writer, verdict, payload, repo, let_through)


def _record_with(writer, verdict: Verdict, payload: dict, repo: str, let_through: bool) -> None:
    envelope = _envelope(payload, repo, verdict.role, verdict.run)
    for check in verdict.checks:
        writer.write(envelope | {"kind": "report_line.checked", "detail": check})
    if verdict.crossed:
        writer.write(envelope | {"kind": "context.crossed", "detail": {
            "threshold": "context_line", "tokens": verdict.context_tokens,
            "action_after": "handed_off" if verdict.table_named else "told"}})
    if let_through and verdict.run:
        writer.write(envelope | {"step": "subagent", "kind": "step.ended", "detail": {
            "outcome": "failed" if verdict.problems else "passed", "report_lines_owed": verdict.owed,
            "report_lines_present": verdict.present}})


def run_subagent_stop(raw: str, *, store=None, open_store: Callable | None = None,
                      repo: str | None = None) -> tuple[str, int]:
    """The hook: (stdout, exit code). Prints a block decision when the stop is held; always exits
    0. `store`, or `open_store()` called only when an event is written, is where events go."""

    payload = None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            _write(lambda writer, repo_: writer.write(_error_event(None, repo_, "payload_unreadable")),
                   payload, store, open_store, repo)
            return "", 0
        try:
            verdict = judge(payload, RoleSet.load())
        except TranscriptUnreadable:
            _write(lambda writer, repo_: writer.write(_error_event(payload, repo_, "transcript_unreadable")),
                   payload, store, open_store, repo)
            return "", 0
        if verdict is None:
            return "", 0
        held = bool(verdict.problems) and payload.get("stop_hook_active") is not True
        _write(None, payload, store, open_store, repo, verdict=verdict, let_through=not held)
        if held:
            return json.dumps({"decision": "block", "reason": verdict.reason()}), 0
        return "", 0
    except Exception as e:  # noqa: BLE001 - a broken hook lets the stop through, and says so
        _write(lambda writer, repo_: writer.write(_error_event(payload, repo_, type(e).__name__)),
               payload, store, open_store, repo)
        return "", 0


def role_of(payload: dict, roles: RoleSet):
    """The role of a hook's payload, from its `agent_type` (a plugin's `plugin:role` too), or None."""

    agent = payload.get("agent_type")
    return roles.roles.get(agent.rsplit(":", 1)[-1]) if isinstance(agent, str) else None


def error_event(payload: dict | None, repo: str, component: str, code: str) -> dict:
    """The `error.raised` of a hook that let its event through (`failed_open`)."""

    return _envelope(payload, repo, "hook", None) | {
        "kind": "error.raised", "detail": {"component": component, "error_code": code[:60], "failed_open": True}}


def write_events(payload: dict | None, write: Callable, *, store=None, open_store: Callable | None = None,
                 repo: str | None = None, hook: str = "subagent-stop") -> None:
    """Call `write(writer, repo)` on a writer of the store: the repo is `repo`, else the GitHub
    origin around the payload's `cwd` (none: nothing is written); the store is `store`, else
    `open_store()`, else the default store. A store or repo that cannot be had is said on stderr,
    never raised: a hook's answer stands whatever the store does."""

    try:
        if repo is None:
            from orch_it.tickets.hook import repo_of

            cwd = payload.get("cwd") if isinstance(payload, dict) and isinstance(payload.get("cwd"), str) else None
            repo = repo_of(cwd)
        if repo is None:
            return
        if store is None:
            if open_store is None:
                from orch_it.events.store import Store

                open_store = Store.open
            store = open_store()
        with store.writer() as writer:
            write(writer, repo)
    except Exception as e:  # noqa: BLE001
        print(f"{hook} hook: events not written: {type(e).__name__}: {e}", file=sys.stderr)


def _write(write_one, payload, store, open_store, repo, *, verdict: Verdict | None = None,
           let_through: bool = True) -> None:
    if verdict is not None:
        write_one = lambda writer, repo_: _record_with(writer, verdict, payload, repo_, let_through)  # noqa: E731
    write_events(payload, write_one, store=store, open_store=open_store, repo=repo)


def _error_event(payload: dict | None, repo: str, code: str) -> dict:
    return error_event(payload, repo, "subagent_stop", code)
