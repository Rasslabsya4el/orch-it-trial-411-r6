"""The shape check on GitHub (#398 decision 3, the GitHub side): a workflow on `issues` events
calls `check_event` with the event's payload when a ticket gets `ready-for-agent` or a ready
ticket is edited, so a label set by hand on github.com, past every session, is judged too.

The ticket is read again through the `Hub` and judged as it stands now by `shape.violations`,
the same function the session hook calls. With violations, a comment lists every one (first,
so a label is never taken off unexplained) and the `ready-for-agent` label comes off; the body
is never edited. Any other event, a ticket no longer ready, or a pull request is skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from orch_it.tickets.hub import LAYER_MARK, Hub
from orch_it.tickets.model import READY_LABEL, parse
from orch_it.tickets.shape import violations

CHECK_ERROR_CODES = ("event_unreadable",)


class CheckError(ValueError):
    """`code`: `event_unreadable` (the event file is no JSON object)."""

    def __init__(self, code: str, detail: str = ""):
        if code not in CHECK_ERROR_CODES:
            raise ValueError(f"not a check error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class CheckResult:
    number: int | None
    outcome: str  # kept | removed | skipped
    codes: list[str] = field(default_factory=list)

    @property
    def line(self) -> str:
        said = f": {', '.join(self.codes)}" if self.codes else ""
        return f"#{self.number} {self.outcome}{said}" if self.number else self.outcome


def read_event(path: str | Path) -> dict:
    try:
        found = json.loads(Path(path).read_text("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise CheckError("event_unreadable", f"{path}: {type(e).__name__}") from None
    if not isinstance(found, dict):
        raise CheckError("event_unreadable", f"{path}: not an object")
    return found


def check_event(event: object, hub: Hub) -> CheckResult:
    """Judge the ticket of an `issues` event; take `ready-for-agent` off a ticket with violations."""

    issue_part = event.get("issue") if isinstance(event, dict) else None
    repository = event.get("repository") if isinstance(event, dict) else None
    if not isinstance(issue_part, dict) or not isinstance(repository, dict):
        return CheckResult(None, "skipped")
    number, repo = issue_part.get("number"), repository.get("full_name")
    if type(number) is not int or not isinstance(repo, str):
        return CheckResult(None, "skipped")
    labels = issue_part.get("labels") if isinstance(issue_part.get("labels"), list) else []
    action = event.get("action")
    if action == "labeled":
        if _name(event.get("label")) != READY_LABEL:
            return CheckResult(number, "skipped")
    elif action != "edited" or READY_LABEL not in {_name(label) for label in labels}:
        return CheckResult(number, "skipped")
    issue = hub.issue(repo, number)
    ready = [label for label in issue.labels if label.lower() == READY_LABEL] if issue else []
    if issue is None or issue.is_pr or not ready:
        return CheckResult(number, "skipped")  # its label already came off, or it is gone
    found = violations(parse(issue.body, title=issue.title, labels=issue.labels))
    if not found:
        return CheckResult(number, "kept")
    # the label first: a comment never says it came off when it did not; a failed comment fails the run
    hub.remove_label(repo, number, ready[0])
    listed = "\n".join(f"- `{v.code}`: {_quiet(v.rule)}" for v in found)
    hub.comment(repo, number, f"{LAYER_MARK}\nThe shape check took `{READY_LABEL}` off this ticket. Write it again "
                              f"with the `write-ticket` skill, then set the label again:\n\n{listed}\n")
    return CheckResult(number, "removed", list(dict.fromkeys(v.code for v in found)))


def _quiet(text: str) -> str:
    """A detail copied from the body, with no `@name` that would notify someone."""

    return text.replace("@", "@" + chr(0x200B))


def _name(label: object) -> str:
    name = label.get("name") if isinstance(label, dict) else label
    return name.strip().lower() if isinstance(name, str) else ""
