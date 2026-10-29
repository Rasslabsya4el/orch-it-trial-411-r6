"""The two queues a ticket waits in (orchestration-layer#403 decision 3):

- `ready`: what the dispatcher offers (part 6 reads it): open `ready-for-agent` issues that are not
  parked (no `parked` label and no `Parked:` line: a line the label lost is still a park);
- `sweep`: what the brain's sweep works: open `needs-triage` issues.

Both put `priority` first, then the oldest number. Neither ever returns the brain's hand-off issue:
an issue carrying `brain-handoff`, or the one the `brain_handoff_issue` setting names, is left out
whatever other labels it carries. Pull requests are no tickets.
"""

from __future__ import annotations

from orch_it import settings
from orch_it.brain import handoff_issue_of
from orch_it.tickets.hub import Issue, ListingHub
from orch_it.tickets.model import HANDOFF_LABEL, PARKED_LABEL, PRIORITY_LABEL, READY_LABEL, TRIAGE_LABEL, parse

QUEUES = {"ready": READY_LABEL, "sweep": TRIAGE_LABEL}


def queue(hub: ListingHub, repo: str, name: str) -> list[Issue]:
    """The issues of the queue `name` (`ready` or `sweep`), in the order they are taken."""

    if name not in QUEUES:
        raise ValueError(f"no such queue: {name!r}; one of {', '.join(QUEUES)}")
    home = handoff_issue_of(settings.load(), repo)
    found = [issue for issue in hub.listed(repo, QUEUES[name])
             if issue.open and not issue.is_pr and issue.number != home
             and HANDOFF_LABEL not in _lower(issue.labels)
             and not (name == "ready" and _parked(issue))]
    return sorted(found, key=lambda issue: (PRIORITY_LABEL not in _lower(issue.labels), issue.number))


def _parked(issue: Issue) -> bool:
    return PARKED_LABEL in _lower(issue.labels) or bool(parse(issue.body).park_lines)


def _lower(labels) -> set[str]:
    return {label.lower() for label in labels}
