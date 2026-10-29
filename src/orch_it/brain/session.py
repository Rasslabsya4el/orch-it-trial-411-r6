"""Who a session is (orchestration-layer#403 decision 2): its title, read as the old layer read it,
and whether it is the brain.

- `session_title(payload)`: the payload's `session_title`, else the `customTitle` of the last
  `custom-title` record in the session's transcript (a rename writes a new record, so the last one
  wins): the last `TAIL_BYTES` of the file first, the whole file only when they hold none. A
  half-written or broken line is skipped. A transcript that cannot be read raises
  `TranscriptUnreadable`.
- `is_brain(payload, brain_title)`: the title equals the repo's brain title exactly (the key), else
  the session was started with `--agent brain` (the second key): the payload's `agent_type` (a
  plugin's `plugin:brain` too), or the `agentSetting` its transcript head records, since a main
  session started with `--agent` carries no `agent_type` in its payload.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

TAIL_BYTES = 256 * 1024  # the newest title record sits near the end: a rename appends one
HEAD_LINES = 50  # the launch facts a transcript records first
BRAIN = "brain"


class TranscriptUnreadable(Exception):
    """The session's transcript is not named or cannot be read."""


def session_title(payload: dict) -> str | None:
    title = payload.get("session_title")
    if isinstance(title, str) and title:
        return title
    return _last_title(_path(payload))


def is_brain(payload: dict, brain_title: str) -> bool:
    """See the module text. A session started as the brain is the brain without its transcript
    being read; otherwise a transcript that cannot be read raises `TranscriptUnreadable`."""

    agent = payload.get("agent_type")
    if brain_by(None, agent if isinstance(agent, str) else None, brain_title):
        return True
    if brain_by(session_title(payload), None, brain_title):
        return True
    path = payload.get("transcript_path")
    if (isinstance(agent, str) and agent) or not isinstance(path, str) or not path:
        return False
    return brain_by(None, _agent_setting(path), brain_title)


def brain_by(title: str | None, agent: str | None, brain_title: str) -> bool:
    """The one rule both the session hook and the store's importer apply: the title equals the brain
    title, else the agent it was started as (`agent_type`, `agentSetting`; a plugin's
    `plugin:brain` too) is the brain."""

    return (title is not None and title == brain_title) or (
        isinstance(agent, str) and agent.rsplit(":", 1)[-1] == BRAIN)


def agent_setting_in_head(number: int, record: dict) -> str | None:
    """The `agentSetting` a transcript's head records (its first `HEAD_LINES` lines), else None."""

    value = record.get("agentSetting") if number < HEAD_LINES else None
    return value if isinstance(value, str) and value else None


def title_of(record: dict) -> str | None:
    """The title a `custom-title` record sets; an empty one sets none."""

    value = record.get("customTitle") if record.get("type") == "custom-title" else None
    return value if isinstance(value, str) and value else None


def _path(payload: dict) -> str:
    path = payload.get("transcript_path")
    if not isinstance(path, str) or not path:
        raise TranscriptUnreadable("no transcript path")
    return path


def _last_title(path: str) -> str | None:
    try:
        with open(path, "rb") as source:
            size = source.seek(0, os.SEEK_END)
            source.seek(max(0, size - TAIL_BYTES))
            tail = source.read()
            if size > TAIL_BYTES:  # the cut landed inside a line: that first piece is no record
                tail = tail.split(b"\n", 1)[1] if b"\n" in tail else b""
            title = _title_in(tail.split(b"\n"))
            if title is None and size > TAIL_BYTES:
                source.seek(0)
                title = _title_in(source)
    except (OSError, ValueError) as e:
        raise TranscriptUnreadable(type(e).__name__) from None
    return title


def _title_in(lines: Iterable[bytes]) -> str | None:
    title = None
    for raw in lines:
        if b'"custom-title"' not in raw:
            continue
        title = title_of(_record(raw)) or title
    return title


def is_background(payload: dict) -> bool:
    """A background session (`claude --bg`): its transcript's head records `sessionKind` `bg`. A
    transcript that cannot be read raises `TranscriptUnreadable`."""

    return any(record.get("sessionKind") == "bg" for record in _head(_path(payload), b"sessionKind"))


def _agent_setting(path: str) -> str | None:
    for record in _head(path, b"agentSetting"):
        value = record.get("agentSetting")
        if isinstance(value, str) and value:
            return value
    return None


def _head(path: str, key: bytes) -> list[dict]:
    """The records of the transcript's first `HEAD_LINES` lines that name `key`."""

    try:
        with open(path, "rb") as source:
            return [_record(raw) for number, raw in zip(range(HEAD_LINES), source) if key in raw]
    except (OSError, ValueError) as e:
        raise TranscriptUnreadable(type(e).__name__) from None


def _record(raw: bytes) -> dict:
    try:
        record = json.loads(raw)
    except (ValueError, RecursionError):
        return {}
    return record if isinstance(record, dict) else {}
