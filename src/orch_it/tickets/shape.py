"""The shape rules (#398 decision 2): `violations` returns every rule a ticket breaks, not the
first, each with its code. The session hook and the GitHub workflow call the same function
(decision 3), and the `write-ticket` skill states the same rules code for code (decision 7).

A missing block is one violation: with no `## For the human` heading the five fields are not
judged, with no `## For the agent` heading nothing of the agent block is, and with no acceptance
table the `Outside case:` line is not matched to a row. A title left out (`None`) is not judged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from orch_it.tickets.model import HUMAN_FIELDS, Ticket, none_reason

TITLE_MAX = 80
WEIGHT_RANGE = range(1, 11)

# code -> the rule as the skill states it
RULES: dict[str, str] = {
    "title_too_long": f"the title is at most {TITLE_MAX} characters",
    "title_parked": "the title never holds the word \"parked\"",
    "title_issue_number": "the title holds no issue number",
    "human_block_missing": "the body has the heading `## For the human`",
    "human_block_not_first": "above `## For the human` stand only `Parked:` lines",
    "human_field_missing": "each of What, Goal, Plan, Result, Undo is there and not empty",
    "human_field_order": "the five fields keep the order What, Goal, Plan, Result, Undo, each once",
    "human_field_not_russian": "the text of each human field is Russian",
    "human_field_path": "a human field holds no file path and no file:line",
    "agent_block_missing": "the body has the heading `## For the agent` after the human block",
    "acceptance_missing": "the agent block has an acceptance table with `check` and `pass` filled in",
    "outside_case_missing": "the agent block has `Outside case: <case> (row <id>)` or `Class: none (<why>)`",
    "outside_case_no_row": "the `Outside case:` line names its acceptance row as `row <id>`",
    "same_object_missing": "the agent block has `Same object: <what else acts on it>` or `none (<why>)`",
    "events_row_missing": "a ticket whose `Adds:` line adds a role or a step has an `events` row",
    "weight_missing": "the agent block has a line `Weight: <1-10>`",
    "weight_bad": "the weight is a whole number from 1 to 10",
}


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str = ""

    @property
    def rule(self) -> str:
        """The rule broken, with this violation's detail."""

        return f"{RULES[self.code]} ({self.detail})" if self.detail else RULES[self.code]

    @property
    def message(self) -> str:
        said = f"{self.code}: {RULES[self.code]}"
        return f"{said} ({self.detail})" if self.detail else said


_ISSUE_NUMBER = re.compile(r"(?:[\w.-]+/[\w.-]+)?#\d+")
_PARKED = re.compile(r"parked", re.IGNORECASE)
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+")
# the fixed English words of the Undo field (`two-way door`, `touches`) are the form, not text
_UNDO_WORDS = re.compile(r"\b(?:two-way door|one-way door|one revert|touches)\b", re.IGNORECASE)
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")
_EXTENSIONS = ("py|pyi|md|js|mjs|cjs|ts|tsx|jsx|json|jsonl|yml|yaml|toml|sql|txt|sh|ps1|bat|cmd|cfg|ini"
               "|html|css|xml|csv|go|rs|java|kt|rb|php|cs|c|h|cpp|hpp|swift|lock|log|env")
_SEGMENT = r"[A-Za-z0-9_.-]"  # ASCII only: a Russian `и/или` is no path
_PATH = re.compile(
    # a file with a known extension, maybe in folders, maybe with `:line`: `docs/events.md:62`
    rf"(?<![\w.])(?:[A-Za-z]:[\\/]|\.{{0,2}}[\\/])?(?:{_SEGMENT}+[\\/])*[A-Za-z0-9_-]{_SEGMENT}*"
    rf"\.(?i:{_EXTENSIONS})(?![\w-])(?::\d+(?:-\d+)?)?"
    # any name with `:line`, like `Makefile:12` or `src/x:40`
    rf"|(?<![\w.:])(?:{_SEGMENT}+[\\/])*[A-Za-z_]{_SEGMENT}*:\d+(?:-\d+)?(?![\w:])"
    # folders of the code without an extension: `src/orch_it`, `.claude/skills`
    rf"|(?<![\w.])(?:\.?[a-z0-9_-]+[\\/])+(?:[a-z0-9_-]*_[a-z0-9_-]*|[a-z0-9_-]+[\\/][a-z0-9_.-]+)"
    rf"|(?<![\w.])(?:\.claude|\.github|\.git|src|tests|docs|tools|scripts|config)[\\/]{_SEGMENT}*"
    r"|#L\d+")
# a product named like a file, `Node.js`, is a name, not a path
_PRODUCT = re.compile(r"[A-Z][a-z]*\.js")
_ROW = re.compile(r"\brows?\s+#?([\w.-]+?)(?=[\s,;:)\]]|\.(?:[\s)\]]|$)|$)", re.IGNORECASE)
_EVENTS_CHECK = re.compile(r"\bevents\b", re.IGNORECASE)
_ADDS_ROLE_OR_STEP = re.compile(r"\b(?:role|step)s?\b", re.IGNORECASE)
_ADDS_NOTHING = re.compile(r"(?:none|nothing|no)\b", re.IGNORECASE)
TOKEN_MAX = 300  # a word longer than any path is read by its head: no regex runs long


def violations(ticket: Ticket) -> list[Violation]:
    """Every shape rule the ticket breaks, in the order of `RULES`' groups: title, human
    block, agent block; empty for a good ticket."""

    found = _title(ticket.title)
    found += _human(ticket)
    if ticket.agent is None:
        found.append(Violation("agent_block_missing"))
    else:
        found += _agent(ticket)
    return found


def title_violations(title: str | None) -> list[Violation]:
    """The title rules alone, for an edit that rewrites only the title."""

    return _title(title)


def _title(title: str | None) -> list[Violation]:
    if title is None:
        return []
    found = []
    title = title.strip()  # GitHub trims a title
    if len(title) > TITLE_MAX:
        found.append(Violation("title_too_long", f"{len(title)} characters"))
    if _PARKED.search(title):
        found.append(Violation("title_parked"))
    numbers = _ISSUE_NUMBER.findall(title)
    if numbers:
        found.append(Violation("title_issue_number", ", ".join(dict.fromkeys(numbers))))
    return found


def _human(ticket: Ticket) -> list[Violation]:
    if not ticket.has_human_block:
        return [Violation("human_block_missing")]
    found = [] if ticket.human_block_on_top else [Violation("human_block_not_first")]
    texts = ticket.human
    for name in HUMAN_FIELDS:
        if not texts.get(name, "").strip():
            found.append(Violation("human_field_missing", name))
    furthest = -1
    for field in ticket.human_fields:
        place = HUMAN_FIELDS.index(field.name)
        if place <= furthest:
            found.append(Violation("human_field_order", field.name))
        furthest = max(furthest, place)
    for name, text in texts.items():
        if text.strip() and not _is_russian(text):
            found.append(Violation("human_field_not_russian", name))
        paths = _paths(text)
        if paths:
            found.append(Violation("human_field_path", f"{name}: {', '.join(paths)}"))
    return found


def _paths(text: str) -> list[str]:
    words = (word[:TOKEN_MAX] for word in _URL.sub(" ", text).split())
    found = (m.group(0).rstrip(".") for word in words for m in _PATH.finditer(word))  # a sentence's dot
    return list(dict.fromkeys(path for path in found if not _PRODUCT.fullmatch(path)))


def _is_russian(text: str) -> bool:
    """Russian when at least half its letters are Cyrillic, leaving out code spans, links and the
    Undo field's fixed words: a domain term in backticks or a product name stays as written."""

    prose = _UNDO_WORDS.sub(" ", _URL.sub(" ", _CODE_SPAN.sub(" ", text)))
    cyrillic, latin = len(_CYRILLIC.findall(prose)), len(_LATIN.findall(prose))
    return cyrillic > 0 and cyrillic >= latin


def _agent(ticket: Ticket) -> list[Violation]:
    found = []
    rows = ticket.acceptance or ()
    if not rows:
        found.append(Violation("acceptance_missing"))
    half = [row.id or "?" for row in rows if not row.check or not row.passes]
    if half:
        found.append(Violation("acceptance_missing", f"no check or pass in row {', '.join(half)}"))
    if ticket.outside_case:
        named = {m.group(1).lower() for m in _ROW.finditer(ticket.outside_case)}
        if rows and not named & {row.id.lower() for row in rows}:
            found.append(Violation("outside_case_no_row", ", ".join(sorted(named)) or "no row named"))
    elif not ticket.class_none:
        found.append(Violation("outside_case_missing"))
    same = (ticket.same_object or "").strip()
    if not same or none_reason(same) == "":
        found.append(Violation("same_object_missing"))
    adds = [item for item in ticket.adds
            if _ADDS_ROLE_OR_STEP.search(item) and not _ADDS_NOTHING.match(item)]
    if adds and not any(_EVENTS_CHECK.search(row.check) for row in rows):
        found.append(Violation("events_row_missing", ", ".join(adds)))
    if ticket.weight_text is None:
        found.append(Violation("weight_missing"))
    elif ticket.weight not in WEIGHT_RANGE:
        found.append(Violation("weight_bad", ticket.weight_text[:40]))
    return found
