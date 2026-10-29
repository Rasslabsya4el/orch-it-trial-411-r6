"""The first writer of the store: Claude Code transcripts become events (#393 decisions 6, 7).

A transcript folder (`~/.claude/projects/<project>/`) holds one `<session id>.jsonl` per
session and, per session, `<session id>/subagents/agent-<id>.jsonl` (with a `.meta.json`) per
subagent. Each becomes a `session` or `subagent` step pair, one `turn.used` per assistant
message id (a message written over several lines counts once, at the usage of its last line),
one `tool.called` per tool use and one `skill.called` per Skill tool use. The files are read
whole every run and read-only; every event's id is a hash of its record, so a second run of
the same transcripts writes nothing, and a half-written last line is left for the next run.

A record still in motion waits: a message until a line of it carries its `stop_reason`, a tool
call until its result, a session's end until its transcript has been quiet for
`quiet_minutes` (a subagent whose meta says `finished` ends at once). Once the transcript is
quiet, what is there is written as it stands. A session that resumes after its end was written
keeps that end; its later turns are still read.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from orch_it.events import schema
from orch_it.events.ids import import_id
from orch_it.events.pricing import price_turn
from orch_it.brain.session import agent_setting_in_head, brain_by, title_of
from orch_it.events.store import Store

SOURCE = "claude-code"
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
# The words Claude Code answers a tool use with when the user or a permission rule refused it.
REFUSAL_MARKS = ("doesn't want to proceed with this tool use", "the tool use was rejected")
ARGS_HEAD_CHARS = 120
TEXT_CHARS = 200  # model, title, effort, skill: longer is no name
MAX_TOKENS = 10**12  # a count past this is no token count
# Secrets an argument may carry. Each match keeps its key (group 1) and loses its value, before
# the head is taken, so no private value reaches the store.
_SECRETS = re.compile(
    r"(?i)((?:authorization|x-[a-z-]*(?:auth|token|key|secret)[a-z-]*):\s*(?:bearer\s+|basic\s+)?|bearer\s+"
    r"|(?:password|passwd|pwd|secret|token|api[_-]?key)[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s\"']+)"
    r"|(?i:(-u\s+|--user[\s=]+)[^\s:]+:)\S+"
    r"|(://[^/\s:@]+:)[^@\s/]+(?=@)"
    r"|sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+")


class ImportError_(Exception):
    """`code`: `no_transcript_folder`, `bad_now` (a `now` without a zone), `bad_quiet_minutes`."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass
class ImportReport:
    written: int = 0
    already_stored: int = 0
    sessions: int = 0
    bad_lines: int = 0  # lines no record could be read from, and turns with no usable time
    pending_lines: int = 0
    refused_events: int = 0  # events the store refused (each counted, the rest still written)
    unreadable_files: int = 0


@dataclass
class _Message:
    at: datetime
    model: str
    usage: dict
    order: int
    finished: bool = False


@dataclass
class _ToolUse:
    at: datetime
    name: str
    args: object
    turn: str | None
    order: int


@dataclass
class _Transcript:
    """What one transcript file says, read once."""

    session: str
    parent: str | None
    meta: dict
    first_at: datetime | None = None
    last_at: datetime | None = None
    title: str | None = None
    full_title: str | None = None  # the title as written, uncut: the one compared with the brain title
    agent_setting: str | None = None  # the `--agent` a main session was started with
    model: str | None = None
    effort: str | None = None
    last_stop: str | None = None
    messages: dict[str, _Message] = field(default_factory=dict)
    tool_uses: dict[str, _ToolUse] = field(default_factory=dict)
    results: dict[str, tuple[datetime, bool, str]] = field(default_factory=dict)
    bad_lines: int = 0
    pending_lines: int = 0
    quiet_tail: tuple[dict, int] | None = None  # a whole last record still without its newline

    def saw(self, at: datetime) -> None:
        self.first_at = at if self.first_at is None else min(self.first_at, at)
        self.last_at = at if self.last_at is None else max(self.last_at, at)


def import_transcripts(folder: str | Path, store: Store, *, repo: str, role: str = "worker",
                       subagent_role: str | None = None, ticket: int | None = None,
                       session: str | None = None, quiet_minutes: float = 30,
                       now: datetime | None = None, brain_title: str | None = None) -> ImportReport:
    """Write the events of the transcripts in `folder` the store does not hold yet.

    `role` is the role of the top-level sessions, except the brain's: a session whose last title is
    the repo's brain title (`brain_title`, else the repo's `brain_chat_title` setting, default
    `Brain: <repo name>`; orchestration-layer#403), or that was started with `--agent brain`, is
    `brain`. A subagent takes `subagent_role`, else its meta's `agentType` when that is a role
    code, else `worker`. `ticket` goes on every event.
    `session` limits the run to one top-level session and its subagents. Bad options raise
    before any file is read."""

    folder = Path(folder)
    if not folder.is_dir():
        raise ImportError_("no_transcript_folder", str(folder))
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ImportError_("bad_now", str(now))
    if not (isinstance(quiet_minutes, (int, float)) and math.isfinite(quiet_minutes)
            and 0 <= quiet_minutes <= 60 * 24 * 365):
        raise ImportError_("bad_quiet_minutes", str(quiet_minutes))
    for probe_role in (role, subagent_role or role):
        schema.validate(_event("step.started", "probe", now.isoformat(), {
            "repo": repo, "ticket": ticket, "role": probe_role, "session": None,
            "parent_session": None, "step": "session", "run": None, "note": None,
            "machine": "probe", "os": "probe", "layer_version": "probe"}, {}))

    brain = brain_title if brain_title is not None else _brain_title(repo)
    report = ImportReport()
    events: dict[str, dict] = {}
    for path, parent, meta in _transcript_files(folder, session):
        try:
            transcript = _read(path, parent, meta)
        except OSError:
            report.unreadable_files += 1
            continue
        report.sessions += 1
        report.bad_lines += transcript.bad_lines
        report.pending_lines += transcript.pending_lines
        quiet = transcript.last_at is not None and (
            now - transcript.last_at >= timedelta(minutes=quiet_minutes))
        envelope = {
            "repo": repo, "ticket": ticket,
            "role": _top_role(transcript, role, brain) if parent is None else _subagent_role(subagent_role, meta),
            "session": transcript.session, "parent_session": parent,
            "step": None, "run": None, "note": None,
        }
        for event in _events(transcript, envelope, quiet):
            events.setdefault(event["id"], event)
    stored = store.existing_ids(list(events))
    report.already_stored = len(stored)
    with store.writer() as writer:
        for event_id, event in events.items():
            if event_id in stored:
                continue
            try:
                writer.write(event)
            except schema.EventError:
                report.refused_events += 1
                continue
            report.written += 1
    return report


def _brain_title(repo: str) -> str:
    from orch_it import settings
    from orch_it.brain import BrainSettings, default_title

    try:
        return BrainSettings.of(settings.load(), repo).chat_title
    except settings.SettingsError as e:  # a broken setting never stops an import: the default title stands
        print(f"import: the brain title setting cannot be read, the default is used: {e}", file=sys.stderr)
        return default_title(repo)


def _top_role(transcript: _Transcript, given: str, brain_title: str) -> str:
    """The brain by its title first, by its `--agent brain` second; else the role given."""

    return "brain" if brain_by(transcript.full_title, transcript.agent_setting, brain_title) else given


def _subagent_role(given: str | None, meta: dict) -> str:
    if given:
        return given
    agent_type = meta.get("agentType")
    roles = schema.codes_of("envelope", "role")
    return agent_type if isinstance(agent_type, str) and agent_type in roles else "worker"


def _transcript_files(folder: Path, session: str | None):
    """(path, parent session or None, meta) of every transcript, top-level sessions first."""

    for path in sorted(folder.glob("*.jsonl")):
        if (session is None or path.stem == session) and path.is_file():
            yield path, None, {}
    for path in sorted(folder.glob("*/subagents/*.jsonl")):
        parent = path.parent.parent.name
        if (session is None or parent == session) and path.is_file():
            yield path, parent, _meta(path)


def _meta(path: Path) -> dict:
    try:
        meta = json.loads(path.with_name(path.stem + ".meta.json").read_text("utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _moment(timestamp: object) -> datetime | None:
    """A transcript timestamp as an aware datetime, or None when it is none."""

    if not isinstance(timestamp, str):
        return None
    try:
        moment = datetime.fromisoformat(timestamp)
        if moment.tzinfo is None:
            return None
        moment.astimezone(UTC).astimezone()  # the local view of it must exist too
    except (ValueError, OverflowError, OSError):
        return None
    return moment


def _local(moment: datetime) -> str:
    """The machine's local time of `moment`, with its offset."""

    return moment.astimezone().isoformat(timespec="microseconds")


def _read(path: Path, parent: str | None, meta: dict) -> _Transcript:
    agent_id = meta.get("agentId")
    if parent is None:
        session = path.stem
    elif isinstance(agent_id, str) and agent_id.strip():
        session = _clean(agent_id)
    else:
        session = path.stem.removeprefix("agent-")
    transcript = _Transcript(session=_clean(session), parent=_clean(parent) if parent else None,
                             meta=meta)
    lines = path.read_bytes().split(b"\n")
    if lines[-1]:
        transcript.pending_lines = 1  # a last line still being written: read next run
    for order, raw in enumerate(lines[:-1]):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except (ValueError, RecursionError):
            transcript.bad_lines += 1
            continue
        if isinstance(record, dict):
            _take(transcript, record, order)
    if lines[-1] and transcript.last_at is not None:
        _take_quiet_tail(transcript, lines[-1], len(lines) - 1)
    return transcript


def _take_quiet_tail(transcript: _Transcript, raw: bytes, order: int) -> None:
    """A last line without its newline that is still a whole record is kept aside: it is read
    only once the transcript is quiet (see `_events`), when no newline is coming."""

    try:
        record = json.loads(raw)
    except (ValueError, RecursionError):
        return
    if isinstance(record, dict):
        transcript.quiet_tail = (record, order)


def _take(transcript: _Transcript, record: dict, order: int) -> None:
    kind = record.get("type")
    title = title_of(record)
    if title is not None:
        transcript.title, transcript.full_title = _name(title), title
    if transcript.agent_setting is None:
        transcript.agent_setting = agent_setting_in_head(order, record)
    at = _moment(record.get("timestamp"))
    if at is None:
        if kind == "assistant":
            transcript.bad_lines += 1  # a turn with no time cannot be placed: counted, not lost
        return
    transcript.saw(at)
    message = record.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
    if kind == "assistant":
        _take_assistant(transcript, record, message, blocks, at, order)
    elif kind == "user":
        for block in blocks:
            if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                text = _text_of(block.get("content"))
                transcript.results[_clean(block["tool_use_id"])] = (
                    at, bool(block.get("is_error")), text)


def _take_assistant(transcript: _Transcript, record: dict, message: dict, blocks: list,
                    at: datetime, order: int) -> None:
    model, message_id = message.get("model"), message.get("id")
    turn = _clean(message_id) if isinstance(message_id, str) and message_id else None
    for block in blocks:
        if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
            transcript.tool_uses.setdefault(_clean(block["id"]), _ToolUse(
                at, _name(str(block.get("name") or "")), block.get("input"), turn, order))
    if not isinstance(model, str) or model == "<synthetic>" or turn is None:
        return
    model = _name(model)
    transcript.model = transcript.model or model
    if isinstance(record.get("effort"), str):
        transcript.effort = transcript.effort or _name(record["effort"])
    stop = message.get("stop_reason")
    if stop is not None:
        transcript.last_stop = stop
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else None
    seen = transcript.messages.get(turn)
    if seen is None:
        seen = transcript.messages[turn] = _Message(at, model, usage or {}, order)
    elif usage:
        seen.usage = usage  # the last line of a message holds its whole usage
    seen.finished = seen.finished or stop is not None


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def _events(transcript: _Transcript, envelope: dict, quiet: bool):
    if quiet and transcript.quiet_tail is not None:
        _take(transcript, *transcript.quiet_tail)
    if transcript.first_at is None:
        return
    is_subagent = transcript.parent is not None
    step = "subagent" if is_subagent else "session"
    session_record = f"{step}:{transcript.session}"
    description = transcript.meta.get("description")
    agent_type = transcript.meta.get("agentType")
    started = {"model": transcript.model, "effort": transcript.effort,
               "title": (_name(description) if isinstance(description, str) else None)
               if is_subagent else transcript.title}
    if is_subagent:
        started["launched_by"] = "session"
        started["agent_type"] = _name(agent_type) if isinstance(agent_type, str) else None
    yield _event("step.started", session_record, _local(transcript.first_at),
                 envelope | {"step": step}, {k: v for k, v in started.items() if v is not None})

    for message_id, message in transcript.messages.items():
        if message.finished or quiet:
            yield _turn_used(message_id, message, envelope)

    first_edit = min((use.order for use in transcript.tool_uses.values() if use.name in EDIT_TOOLS),
                     default=None)
    for tool_id, use in transcript.tool_uses.items():
        result = transcript.results.get(tool_id)
        if result is None and not quiet:
            continue
        detail = {"tool": use.name, "args_head": _args_head(use.args)}
        if result is not None:
            result_at, is_error, text = result
            refused = is_error and any(mark in text.lower() for mark in REFUSAL_MARKS)
            detail["outcome"] = "refused" if refused else "failed" if is_error else "passed"
            detail["refused_by"] = "permission" if refused else None
            detail["ms"] = max(0, round((result_at - use.at).total_seconds() * 1000))
        of_turn = envelope | {"turn": use.turn}
        yield _event("tool.called", tool_id, _local(use.at), of_turn, detail)
        if use.name == "Skill" and isinstance(use.args, dict) and isinstance(use.args.get("skill"), str):
            yield _event("skill.called", tool_id, _local(use.at), of_turn, {
                "skill": _name(use.args["skill"]),
                "before_first_edit": first_edit is None or use.order < first_edit})

    finished = is_subagent and transcript.meta.get("finished") is True
    if quiet or finished:
        passed = transcript.last_stop == "end_turn" or finished
        detail = {"outcome": "passed" if passed else "no_result",
                  "reason": "turn_ended" if passed else "idle"}
        if not is_subagent and transcript.title:
            detail["title"] = transcript.title
        yield _event("step.ended", session_record, _local(transcript.last_at),
                     envelope | {"step": step}, detail)


def _count(usage: dict, key: str) -> int | None:
    """A token count of `usage`: 0 when absent, None when it is no count."""

    value = usage.get(key)
    if value is None:
        return 0
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if type(value) is not int or not 0 <= value <= MAX_TOKENS:
        return None
    return value


def _turn_used(message_id: str, message: _Message, envelope: dict) -> dict:
    usage = message.usage
    creation = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    write_1h, write_5m = _count(creation, "ephemeral_1h_input_tokens"), _count(creation, "ephemeral_5m_input_tokens")
    write = _count(usage, "cache_creation_input_tokens")
    if "cache_creation_input_tokens" not in usage and write_1h is not None and write_5m is not None:
        write = write_1h + write_5m
    tokens = {
        "input_tokens": _count(usage, "input_tokens"),
        "output_tokens": _count(usage, "output_tokens"),
        "cache_read_tokens": _count(usage, "cache_read_input_tokens"),
        "cache_write_tokens": write,
        "cache_write_1h_tokens": None if write is None or write_1h is None else min(write_1h, write),
    }
    if not usage or None in tokens.values():
        detail = {"model": message.model, **{k: v or 0 for k, v in tokens.items()},
                  "usd": None, "price_table": None, "reason": "usage_unreadable"}
    else:
        usd, table = price_turn(message.model, tokens)
        detail = {"model": message.model, **tokens, "usd": usd, "price_table": table}
        if usd is None:
            detail["reason"] = "price_missing"
    return _event("turn.used", message_id, _local(message.at), envelope | {"turn": message_id},
                  detail)


def _args_head(args: object) -> str:
    """The tool's arguments as compact JSON, secrets cut out, at most 120 characters."""

    try:
        text = json.dumps(_without_secrets(args), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        text = ""
    return _clean(text)[:ARGS_HEAD_CHARS]


def _without_secrets(value: object, depth: int = 0) -> object:
    """`value` with every string's secrets cut out (strings as written, before JSON escapes
    them); past 50 levels the rest is dropped."""

    if depth > 50:
        return None
    if isinstance(value, str):
        return _SECRETS.sub(_hide, _clean(value))
    if isinstance(value, dict):
        return {_SECRETS.sub(_hide, _clean(str(k))): _without_secrets(v, depth + 1)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_without_secrets(v, depth + 1) for v in value]
    return value


def _hide(match: re.Match) -> str:
    kept = next((group for group in match.groups() if group), "")
    return kept + "<secret>"


def _clean(text: str) -> str:
    """`text` with any lone surrogate (valid in JSON, not in UTF-8) replaced."""

    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def _name(text: str) -> str:
    return _clean(text)[:TEXT_CHARS]


def _event(kind: str, record_id: str, at: str, envelope: dict, detail: dict) -> dict:
    return {"id": import_id(SOURCE, record_id, kind), "kind": kind, "at": at,
            "turn": None, **envelope, "detail": detail}
