"""The events of the merge flow (#400), one writer per call: the envelope filled the same way for
`pr open`, the follow, the line, the main check and the sweep."""

from __future__ import annotations

from datetime import datetime

from orch_it.events.store import Store


def write(store: Store, repo: str, ticket: int | None, role: str, run: str | None, step: str | None,
          events: list[tuple[str, dict]]) -> None:
    """Write `(kind, detail)` pairs; `step` goes on the `step.*` ones."""

    with store.writer() as writer:
        for kind, detail in events:
            writer.write({"kind": kind, "repo": repo, "ticket": ticket, "role": role, "session": None,
                          "parent_session": None, "turn": None, "step": step if kind.startswith("step.") else None,
                          "run": run, "note": None, "detail": detail})


def pr_run(repo: str, pr: int, ready_at: datetime | None) -> str:
    """The run id of a pull request's `pr` step: the pull request and the moment of its ready mark,
    read from GitHub, so a follow on any machine (or in a workflow) ends the step `pr open` began."""

    return f"pr:{repo}#{pr}@{ready_at.isoformat() if ready_at else 'unknown'}"


def pr_updated(pr: int, old_head: str, new_head: str | None) -> tuple[str, dict]:
    """The `pr.updated` of the merge line bringing a head up to date with its base: never forced;
    the run it cancels is GitHub's to cancel, not counted here."""

    return ("pr.updated", {"pr": pr, "old_head": old_head, "new_head": new_head, "forced": False, "base_moved": True,
                           "runs_cancelled": None})


def ticket_closed(rows_total: int | None, rows_passed: int | None) -> tuple[str, dict]:
    """The `ticket.closed` of a ticket the layer closes as done (the follow's close, the main check's)."""

    return ("ticket.closed", {"via": "close_command", "state_reason": "completed", "rows_total": rows_total,
                              "rows_passed": rows_passed, "comments_unanswered": 0})
