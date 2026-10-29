"""The marks the merge flow leaves in its comments (#400), so its state lives on GitHub and any
machine reads the same thing: hidden HTML lines after the layer's own `LAYER_MARK`.

- the acceptance comment on a ticket, written by `pr open`: which pull request and head the rows
  were checked for, the hash of the ticket body they were checked against, and the results table;
- the ready mark on a pull request: it entered the merge line at that moment (its ready time);
- the left mark on a pull request: it left the line (red, conflict, a person's comment, ...), until
  the next ready mark puts it back;
- the follow mark on a ticket: what the follow did with it (`red`, `conflict`, ...), so the red
  returns in a row are counted from GitHub alone.

A mark counts only in a comment that carries `LAYER_MARK`, written by the layer: a bot (the
workflows' token) or the owner's own login (the layer's commands run under the owner's `gh`), which
`of_layer` keeps; a person who pastes a mark is no layer. Closing keywords are found here too: a pull request says `Refs #N` and no
text the layer pushes may close an issue on merge (MERGE-10).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from orch_it.tickets.hub import LAYER_MARK, Comment
from orch_it.tickets.model import without_park_lines

RESULTS = ("pass", "fail", "na", "after merge")
_MARK = re.compile(r"<!-- orch-it (acceptance|ready|left|follow) ([^>]*?) ?-->")
_FIELD = re.compile(r"([a-z_]+)=(\S*)")
# GitHub's closing keywords before an issue, `#N`, `owner/repo#N` or an issue link
_CLOSING = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b:?\s*"
    r"(?:[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+#\d+|#\d+|GH-\d+|https://github\.com/[^\s/]+/[^\s/]+/issues/\d+)",
    re.IGNORECASE)
# a part's pull request (#434): its own line `Part: k of n`
_PART = re.compile(r"(?m)^Part: ([0-9]{1,9}) of ([0-9]{1,9})[ \t]*\r?$")


@dataclass(frozen=True)
class ResultRow:
    id: str
    result: str  # one of RESULTS
    evidence: str


@dataclass(frozen=True)
class Acceptance:
    pr: int
    head: str
    body: str  # the hash of the ticket body the rows were checked against
    rows: tuple[ResultRow, ...]
    at: datetime


@dataclass(frozen=True)
class Ready:
    ticket: int | None
    head: str
    at: datetime


@dataclass(frozen=True)
class Followed:
    outcome: str
    at: datetime


def of_layer(comments: Iterable[Comment], owner: str) -> list[Comment]:
    """The comments the layer may have written: a bot's, or the owner's (`owner`, any case)."""

    return [c for c in comments if c.bot or c.author.lower() == owner.lower()]


def is_person(comment: Comment, owner: str) -> bool:
    """A comment by a person: no bot, and not the layer's own (the owner's login with `LAYER_MARK`);
    a stranger's comment is a person's whatever it holds."""

    if comment.bot:
        return False
    return comment.author.lower() != owner.lower() or LAYER_MARK not in comment.body


def closing_keywords(text: str) -> list[str]:
    """Every closing keyword with its issue in `text`, as written."""

    return [m.group(0) for m in _CLOSING.finditer(text or "")]


def part_of(body: str) -> tuple[int, int] | None:
    """The part a pull request's body names with its line `Part: k of n` (#434), or None when it names
    none, or names one that is not 1 <= k <= n."""

    found = _PART.search(body or "")
    if found is None:
        return None
    k, n = int(found.group(1)), int(found.group(2))
    return (k, n) if 1 <= k <= n else None


def body_hash(body: str) -> str:
    """The ticket body as the brain wrote it: park lines out, line ends and edge blanks aside."""

    plain = without_park_lines((body or "").replace("\r\n", "\n").replace("\r", "\n")).strip()
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()[:16]


def read_results(text: str) -> list[ResultRow] | None:
    """The rows of the first table whose header names `row` and `result`; None without one. A
    result is read in lower case; one outside `RESULTS` is kept as written for the caller to refuse."""

    lines = [line.strip() for line in (text or "").replace("\r\n", "\n").split("\n")]
    for i, line in enumerate(lines):
        if not line.startswith("|") or i + 1 >= len(lines):
            continue
        header = [cell.lower() for cell in _cells(line)]
        if "row" not in header or "result" not in header or not re.fullmatch(r"\|[\s:|-]+\|?", lines[i + 1]):
            continue
        at_row, at_result = header.index("row"), header.index("result")
        at_evidence = header.index("evidence") if "evidence" in header else None
        rows = []
        for body in lines[i + 2:]:
            if not body.startswith("|"):
                break
            cells = _cells(body) + [""] * len(header)
            rows.append(ResultRow(cells[at_row], " ".join(cells[at_result].lower().split()),
                                  cells[at_evidence] if at_evidence is not None else ""))
        return rows
    return None


def acceptance_comment(pr: int, head: str, body: str, rows: Iterable[ResultRow]) -> str:
    rows = list(rows)
    later = [row.id for row in rows if row.result == "after merge"]
    table = "".join(f"| {row.id} | {row.result} | {cell(row.evidence)} |\n" for row in rows)
    tail = (f"\nAfter merge: row{'s' if len(later) > 1 else ''} {', '.join(later)} "
            f"run{'' if len(later) > 1 else 's'} on main once #{pr} merges.\n" if later else "")
    return (f"{LAYER_MARK}\n<!-- orch-it acceptance pr={pr} head={head} body={body} -->\n"
            f"Acceptance of this ticket before pull request #{pr}, at head `{head[:12]}`:\n\n"
            f"| row | result | evidence |\n|---|---|---|\n{table}{tail}")


def acceptance(comments: Iterable[Comment]) -> Acceptance | None:
    """The latest acceptance comment of the layer."""

    found = None
    for comment, fields in _marked(comments, "acceptance"):
        rows = read_results(comment.body)
        if rows is not None and fields.get("pr", "").isdigit():
            found = Acceptance(int(fields["pr"]), fields.get("head", ""), fields.get("body", ""), tuple(rows),
                               comment.at)
    return found


def ready_comment(ticket: int | None, head: str, at: datetime) -> str:
    named = f"ticket=#{ticket} " if ticket else ""
    return (f"{LAYER_MARK}\n<!-- orch-it ready {named}head={head} -->\n"
            f"In the merge line from {at.isoformat(timespec='seconds')}"
            f"{f' for ticket #{ticket}' if ticket else ''}, at head `{head[:12]}`. A comment by a person here "
            "takes it out of the line and sends its ticket to the brain.")


def left_comment(reason: str, detail: str) -> str:
    return f"{LAYER_MARK}\n<!-- orch-it left reason={reason} -->\nOut of the merge line: {detail}"


def ready_since(comments: Iterable[Comment]) -> Ready | None:
    """The ready mark that holds the pull request in the line: the latest one, when no left mark
    came after it; None when it is not in the line."""

    held = None
    for comment, fields in _marked(comments, "ready", "left"):
        if fields["mark"] == "left":
            held = None
        else:
            ticket = fields.get("ticket", "").removeprefix("#")
            held = Ready(int(ticket) if ticket.isdigit() else None, fields.get("head", ""), comment.at)
    return held


def last_ready(comments: Iterable[Comment]) -> Ready | None:
    """The latest ready mark, a left mark after it or not: the start of the pull request's step."""

    held = None
    for comment, fields in _marked(comments, "ready"):
        ticket = fields.get("ticket", "").removeprefix("#")
        held = Ready(int(ticket) if ticket.isdigit() else None, fields.get("head", ""), comment.at)
    return held


def left_reason(comments: Iterable[Comment]) -> str | None:
    """The reason of the latest left mark, when it is later than every ready mark."""

    reason = None
    for _, fields in _marked(comments, "ready", "left"):
        reason = fields.get("reason") if fields["mark"] == "left" else None
    return reason


def follow_mark(outcome: str, pr: int | None = None) -> str:
    """`pr`: the pull request the outcome is about, for an outcome written once per pull request."""

    named = f" pr={pr}" if pr is not None else ""
    return f"{LAYER_MARK}\n<!-- orch-it follow outcome={outcome}{named} -->\n"


def part_landed(comments: Iterable[Comment], pr: int) -> bool:
    """Whether the ticket already says that a part, pull request `pr`, landed (#434)."""

    return any(fields.get("outcome") == "part_landed" and fields.get("pr") == str(pr)
               for _, fields in _marked(comments, "follow"))


def follows(comments: Iterable[Comment]) -> list[Followed]:
    """The follow marks on a ticket, oldest first."""

    return [Followed(fields.get("outcome", ""), comment.at) for comment, fields in _marked(comments, "follow")]


def _marked(comments: Iterable[Comment], *kinds: str):
    for comment in sorted(comments, key=lambda c: c.at):
        if comment.by_a_person or LAYER_MARK not in comment.body:
            continue
        for found in _MARK.finditer(comment.body):
            if found.group(1) in kinds:
                fields = dict(_FIELD.findall(found.group(2)))
                fields["mark"] = found.group(1)
                yield comment, fields
                break


def _cells(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") and not inner.endswith("\\|") else inner
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", inner)]


def cell(text: str) -> str:
    """Text made fit for one cell of a Markdown table: one line, pipes escaped, at most 500 characters."""

    return " ".join(text.split()).replace("|", "\\|")[:500]
