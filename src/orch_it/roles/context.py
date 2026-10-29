"""The context line of a session as a check (#399 decision 2; orchestration-layer#403 finds the
session by its title): the brain's session hook (`orch_it.brain.hook`) asks `reminder` on each of
its prompts. The session's size is its latest turn's prompt (input, cache read and cache write
tokens), read from the end of its own transcript, so a session that compacted is measured as it is
now. Past the line the turn gets a reminder; the turn that first passes it (the turn before was
under it) is a crossing, so the `context.crossed` event counts crossings, not messages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from orch_it.roles.hold import prompt_tokens

TAIL_BYTES = 4 * 1024 * 1024  # the end of the transcript read for its last turns
REMINDER = ("Your context is past your context line ({tokens} of {line} tokens). Answer the owner in full and finish"
            " the step in hand; then write the hand-off comment and leave the next ticket work to the next chat.")


class Unreadable(Exception):
    """The transcript cannot be read."""


@dataclass(frozen=True)
class Past:
    tokens: int
    crossed: bool  # this turn is the one that passed the line
    said: str


def reminder(path: str, line: int) -> Past | None:
    """None under the line; past it, the reminder and whether this turn crossed it. A transcript
    that cannot be read raises `Unreadable`."""

    sizes = prompt_sizes(path)
    if not sizes or sizes[-1] <= line:
        return None
    return Past(sizes[-1], len(sizes) == 1 or sizes[-2] <= line, REMINDER.format(tokens=sizes[-1], line=line))


def prompt_sizes(path: str) -> list[int]:
    """The prompt size of each of the last turns that carry usage, oldest first, read from the end
    of the transcript; a half-written or broken line is skipped."""

    try:
        with open(path, "rb") as source:
            size = os.fstat(source.fileno()).st_size
            source.seek(max(0, size - TAIL_BYTES))
            tail = source.read()
    except (OSError, ValueError) as e:
        raise Unreadable(type(e).__name__) from None
    lines = tail.split(b"\n")
    if size > TAIL_BYTES:
        lines = lines[1:]  # the first line of the tail is cut
    sizes = []
    for line in lines:
        if b"usage" not in line:
            continue
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            continue
        message = record.get("message") if isinstance(record, dict) else None
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            sizes.append(prompt_tokens(record))
    return sizes
