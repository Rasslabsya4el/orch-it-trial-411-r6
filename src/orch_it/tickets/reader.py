"""The one reader of parks (#398 decision 5): `read_parks` reads every open ticket of a repo
that holds a park, the `parked` label or a `Parked:` line, from GitHub alone, and reports each:

- `due`: its event happened the right way (the PR merged, the ticket closed completed, the date
  came, the run ended green, the owner answered on the ticket named, the repo runs the version):
  line and label come off and it goes back to its queue with its other labels (PARK-11);
- `other_way`: the event happened the other way (a PR closed unmerged, a ticket closed not
  planned, a run ended red: MERGE-14, PARK-10);
- `comment`: a person commented on it after the park (PARK-9); the layer's own comments carry
  `LAYER_MARK` and a bot's never count;
- `late`: its `until` day (UTC) has passed and its event has not come (PARK-5, PARK-12);
- `unreadable`: the label without one line in the form, a line without the label (PARK-4), or
  a ref that is not there;
- `waiting`: none of these; nothing is changed;
- `failed`: a `gh` call about it failed, or it changed while it was read; nothing is written
  and the next run reads it again (PARK-6);
- `followed`: given `follow` (#400 decision 3), a `pr_ended` park whose pull request of the same
  repo has ended is the follow's: the ticket follows its pull request (closed with its rows on a
  merge, never sent back to ready by the reader) and the report holds the follow's outcome.

`other_way`, `comment`, `late` and `unreadable` go to the brain: the park comes off,
`needs-triage` goes on, `ready-for-agent` and `ready-for-human` come off (so no agent takes it
before the brain), and a comment says why. An event that came is `due` even past its `until`;
its lateness is in `late_minutes`. Each ticket is acted on in this order: the comment, then
the write of body and labels if the ticket is unchanged since it was read, then the events, so
a failure at any step leaves it parked for the next run. Each run re-reads every park (PARK-6)
and needs no process of the session that parked (PARK-7): the schedule that runs it belongs to
the launcher (part 6).

Events: `due` ends the park's `wait` step passed; `other_way` failed, `late` no_result,
`comment` stopped, `unreadable` no_result, each with its reason, plus `ticket.unparked` (by
`event`, `returned_as` the outcome) and, for `unreadable`, `park.unreadable` with its reason; a
waiting park writes a `step.progressed` of its `wait`; a failed one an `error.raised`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from orch_it.events.store import Store
from orch_it.testruns.github import GithubError
from orch_it.tickets.hub import LAYER_MARK, Hub, Issue
from orch_it.tickets.model import PARKED_LABEL, READY_LABEL, parse
from orch_it.tickets.park import (ROLE, Park, ParkError, held_park, late_minutes, park_run, run_ended, take_off,
                                  version_reached, write_events)

TRIAGE_LABEL = "needs-triage"
HUMAN_LABEL = "ready-for-human"
_TO_THE_BRAIN = {"other_way": "failed", "late": "no_result", "comment": "stopped", "unreadable": "no_result"}
_SAID = {
    "other_way": "its event happened the other way",
    "late": "its `until` day passed and its event has not come",
    "comment": "a person commented on it while it was parked",
    "unreadable": "its park cannot be read",
}


@dataclass(frozen=True)
class ParkReport:
    number: int
    outcome: str
    line: str  # the park line as read, or what stood in its place
    detail: str = ""


def read_parks(hub: Hub, store: Store, repo: str, *, now: datetime, owner: str | None = None,
               role: str = ROLE, follow: Callable[[int], str] | None = None) -> list[ParkReport]:
    """Read, act on and record every parked ticket of `repo`, in number order. `owner`: the
    login whose comment answers an `owner_answer` park (default: the repo's owner). `follow`: given a
    pull request number, moves its ticket and returns the outcome."""

    answerer = (owner or repo.split("/")[0]).lower()
    reports = []
    for number in hub.parked(repo):
        try:
            report = _read_one(hub, store, repo, number, now, answerer, role, follow)
        except (GithubError, ParkError) as e:
            write_events(store, repo, number, role, None, [
                ("error.raised", {"component": "tickets_parks", "error_code": e.code, "failed_open": False})])
            report = ParkReport(number, "failed", "", str(e)[:200])
        if report is not None:
            reports.append(report)
    return reports


def _read_one(hub: Hub, store: Store, repo: str, number: int, now: datetime, answerer: str,
              role: str, follow: Callable[[int], str] | None = None) -> ParkReport | None:
    issue = hub.issue(repo, number)
    if issue is None or not issue.open or issue.is_pr:
        return None  # gone since the list was read
    held = parse(issue.body, labels=issue.labels)
    if not held.parked and not held.park_lines:
        return None
    parked_at = hub.parked_at(repo, number)
    run = park_run(repo, number, parked_at)
    found = held_park(held.park_lines)
    shown = " / ".join(held.park_lines)[:200]
    if found is None or not held.parked:
        reason = ("label_missing" if not held.parked else "line_missing" if not held.park_lines
                  else "lines_many" if len(held.park_lines) > 1 else "line_bad")
        return _to_the_brain(hub, store, issue, run, role, "unreadable", shown, found, now, reason)
    if follow is not None and found.kind == "pr_ended":
        target = found.ticket.in_repo(repo)
        pull = hub.issue(*target) if target[0].lower() == repo.lower() else None
        if pull is not None and pull.is_pr and not pull.open:
            return ParkReport(number, "followed", found.line, follow(pull.number))
    outcome, detail = _event(hub, store, repo, found, parked_at, now, answerer)
    if outcome == "waiting" and _a_person_commented(hub, repo, number, parked_at):
        outcome, detail = "comment", "a person commented"
    if outcome == "waiting" and now.date() > found.until:
        outcome, detail = "late", f"until {found.until.isoformat()}"
    if outcome == "waiting":
        write_events(store, repo, number, role, run, [("step.progressed", {"on": f"{found.kind} {found.ref}"})])
        return ParkReport(number, "waiting", found.line, detail)
    if outcome == "due":
        hub.comment(repo, number, f"{LAYER_MARK}\nReturned from the park `{found.line}`: {detail}.")
        take_off(hub, issue)
        write_events(store, repo, number, role, run, [
            ("step.ended", {"outcome": "passed"}),
            ("ticket.unparked", {"by": "event", "returned_as": "due", "on_ref": found.ref,
                                 "late_minutes": late_minutes(found, now)})])
        return ParkReport(number, "due", found.line, detail)
    reason = "ref_missing" if outcome == "unreadable" else None
    return _to_the_brain(hub, store, issue, run, role, outcome, found.line, found, now, reason, detail)


def _to_the_brain(hub: Hub, store: Store, issue: Issue, run: str, role: str, outcome: str, line: str,
                  found: Park | None, now: datetime, unreadable: str | None, detail: str = "") -> ParkReport:
    why = f"{_SAID[outcome]}{f' ({detail or unreadable})' if detail or unreadable else ''}"
    hub.comment(issue.repo, issue.number,
                f"{LAYER_MARK}\nTo the brain: the park `{line or 'with no Parked: line'}` ended, {why}.")
    take_off(hub, issue, add=(TRIAGE_LABEL,), remove=(READY_LABEL, HUMAN_LABEL))
    events = [("step.ended", {"outcome": _TO_THE_BRAIN[outcome], "reason": outcome}),
              ("ticket.unparked", {"by": "event", "returned_as": outcome, "on_ref": found.ref if found else None,
                                   "late_minutes": late_minutes(found, now) if found else None})]
    if unreadable:
        events.append(("park.unreadable", {"reason": unreadable}))
    write_events(store, issue.repo, issue.number, role, run, events)
    return ParkReport(issue.number, outcome, line, detail or (unreadable or ""))


def _event(hub: Hub, store: Store, repo: str, found: Park, parked_at: datetime | None, now: datetime,
           answerer: str) -> tuple[str, str]:
    """(`due` | `other_way` | `waiting` | `unreadable`, what was seen) for the park's event."""

    ticket = found.ticket
    if found.kind == "date":
        return ("due", f"{found.ref} came") if now.date() >= found.date else ("waiting", "")
    if found.kind == "owner_answer":
        answered = [c for c in hub.comments(*ticket.in_repo(repo))
                    if c.by_a_person and c.author.lower() == answerer and parked_at is not None and c.at > parked_at]
        return ("due", f"the owner answered on {found.ref}") if answered else ("waiting", "")
    if ticket is not None:
        other = hub.issue(*ticket.in_repo(repo))
        if other is None:
            return "unreadable", f"{found.ref} is not there"
        if other.open:
            return "waiting", ""
        if found.kind == "pr_ended":
            return ("due", f"{found.ref} merged") if other.merged else ("other_way", f"{found.ref} closed unmerged")
        if other.state_reason == "completed":
            return "due", f"{found.ref} closed completed"
        return "other_way", f"{found.ref} closed {other.state_reason or 'not planned'}"
    if found.kind == "run_ended":
        state = run_ended(hub, store, repo, found.ref)
        if state is None:
            return "unreadable", f"run {found.ref} is not there"
        if not state.ended:
            return "waiting", ""
        return ("due", f"run {found.ref} ended green") if state.passed else ("other_way", f"run {found.ref} ended red")
    held = hub.layer_version(repo)
    if held is not None and version_reached(held, found.ref):
        return "due", f"{repo} runs {held}"
    return "waiting", ""


def _a_person_commented(hub: Hub, repo: str, number: int, parked_at: datetime | None) -> bool:
    """A comment on the parked ticket after its park, by a person, not the layer (PARK-9); with
    no time of the park known, no comment can be shown to come after it."""

    return parked_at is not None and any(c.by_a_person and c.at > parked_at for c in hub.comments(repo, number))
