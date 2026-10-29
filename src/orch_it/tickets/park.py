"""The park form (#398 decision 4): one line at the top of a ticket's body,

    Parked: <kind> <ref> until <YYYY-MM-DD>

`kind` is one of the six events of `park_on` (#384 PARK-3: one event per park, MERGE-14: it
names its PR or ticket); `ref` is what the kind needs: `#N` or `owner/repo#N` for `pr_ended`,
`ticket_closed` and `owner_answer` (where the owner is asked), a date for `date`, a GitHub
Actions run id (digits) or a store run id (a UUID) for `run_ended`, a version `X.Y.Z` for
`layer_version`; `until` is the last day the park may wait (PARK-5, PARK-12: a park whose event
never comes ends late). Anything else is refused with a code (PARK-1, PARK-2).

`park` is the only writer of the line: it writes the line and the `parked` label in one request
(PARK-4), keeps every other label (PARK-11: priority stays), and refuses at write a park that
could never return: a second event (PARK-3), an `until` in the past, a ref that is not there or
not of its kind (MERGE-14), the ticket itself or its own parent (PARK-10), an event that has
already happened. `unpark` takes line and label off together. Both write their events: a park
is a `wait` step (reason `parked`) whose run id names the ticket and the moment of its label,
so any machine reading GitHub alone finds the same step; a refused park is a `wait` pair
ended `refused` with the refusal's code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

from orch_it.events.store import Store
from orch_it.tickets.hub import Hub, Issue, RunState
from orch_it.tickets.model import PARK_PREFIX, PARKED_LABEL, parse, without_park_lines

KINDS = ("pr_ended", "ticket_closed", "date", "run_ended", "owner_answer", "layer_version")
TICKET_KINDS = ("pr_ended", "ticket_closed", "owner_answer")
ROLE = "script"
BODY_MAX = 65_536  # GitHub's own bound on an issue body

# every code but `not_parked` is also a `wait_end_reason`: a refused park writes it to the store
PARK_ERROR_CODES = (
    "kind_unknown",       # no kind of the six (free text, PARK-1)
    "ref_bad",            # a ref not of its kind's form, or two events (PARK-3)
    "until_bad",          # no `until <YYYY-MM-DD>`, or no real date (PARK-5)
    "date_after_until",   # a date park that would be late before its date comes
    "until_past",         # an `until` before today (PARK-5)
    "ticket_missing",     # no such issue, or a pull request
    "ticket_not_open",
    "already_parked",     # one event per park (PARK-3)
    "ref_missing",        # the ticket, pull request or run named is not there
    "ref_not_pr",         # `pr_ended` on an issue (MERGE-14)
    "ref_not_ticket",     # `ticket_closed` on a pull request
    "ref_self",           # waiting on itself
    "ref_parent",         # waiting on its own parent, which waits on it (PARK-10)
    "event_passed",       # the event has already happened
    "park_mismatch",      # the read-back does not hold line and label together (PARK-4)
    "ticket_changed",     # the ticket changed between the read and the write: nothing written
    "body_too_long",      # the body with its line would pass GitHub's bound
    "not_parked",         # unpark of a ticket that holds no park
)

_LINE = re.compile(rf"{re.escape(PARK_PREFIX)}\s*(?P<kind>\S*)\s*(?P<ref>.*?)\s*(?:\buntil\s+(?P<until>\S+))?\s*")
_TICKET_REF = re.compile(
    r"(?:(?P<repo>[A-Za-z0-9][A-Za-z0-9-]{0,38}/(?!\.\.?#)[A-Za-z0-9_.-]{1,100}))?#(?P<number>[1-9][0-9]{0,9})")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_GITHUB_RUN = re.compile(r"[1-9][0-9]{0,19}")
_STORE_RUN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_VERSION = re.compile(r"(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})")


class ParkError(ValueError):
    """A park that cannot be written or read; `code` is one of `PARK_ERROR_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in PARK_ERROR_CODES:
            raise ValueError(f"not a park error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class TicketRef:
    repo: str | None  # None: the parked ticket's own repo
    number: int

    def in_repo(self, home: str) -> tuple[str, int]:
        return (self.repo or home, self.number)

    def names(self, repo: str, number: int, home: str) -> bool:
        """Whether this ref is ticket `number` of `repo`; repo names compare as GitHub does,
        in any letter case."""

        mine, its = self.in_repo(home)
        return its == number and mine.lower() == repo.lower()


@dataclass(frozen=True)
class Park:
    kind: str
    ref: str
    until: date

    @property
    def line(self) -> str:
        return f"{PARK_PREFIX} {self.kind} {self.ref} until {self.until.isoformat()}"

    @property
    def ticket(self) -> TicketRef | None:
        found = _TICKET_REF.fullmatch(self.ref) if self.kind in TICKET_KINDS else None
        return TicketRef(found.group("repo"), int(found.group("number"))) if found else None

    @property
    def date(self) -> date | None:
        return date.fromisoformat(self.ref) if self.kind == "date" else None


def make(kind: str, ref: str, until: str | date) -> Park:
    """The park of `kind` on `ref` until `until`, or `ParkError` with the first reason it is
    no park: the kind, then the ref, then the date."""

    if kind not in KINDS:
        raise ParkError("kind_unknown", kind[:40])
    if not _ref_fits(kind, ref):
        raise ParkError("ref_bad", f"{kind} takes {_REF_FORMS[kind]}, not {ref[:40]!r}")
    last = until if isinstance(until, date) else _date(until)
    if last is None:
        raise ParkError("until_bad", str(until)[:40])
    park = Park(kind, ref, last)
    if park.date is not None and park.date > last:
        raise ParkError("date_after_until", f"{ref} is after {last.isoformat()}")
    return park


def read_line(line: str) -> Park:
    """The park a body's `Parked:` line holds, or `ParkError`."""

    found = _LINE.fullmatch(line.strip()) if line.strip().startswith(PARK_PREFIX) else None
    if not found or found.group("kind") not in KINDS:
        raise ParkError("kind_unknown", line.strip()[:60])
    if found.group("until") is None:
        raise ParkError("until_bad", "no `until <YYYY-MM-DD>`")
    return make(found.group("kind"), found.group("ref"), found.group("until"))


_REF_FORMS = {
    "pr_ended": "a pull request `#N` or `owner/repo#N`",
    "ticket_closed": "a ticket `#N` or `owner/repo#N`",
    "owner_answer": "the ticket the owner is asked on, `#N` or `owner/repo#N`",
    "date": "a date `YYYY-MM-DD`",
    "run_ended": "a GitHub Actions run id or a store run id",
    "layer_version": "a version `X.Y.Z`",
}


def _ref_fits(kind: str, ref: str) -> bool:
    if kind in TICKET_KINDS:
        return bool(_TICKET_REF.fullmatch(ref))
    if kind == "date":
        return _date(ref) is not None
    if kind == "run_ended":
        return bool(_GITHUB_RUN.fullmatch(ref) or _STORE_RUN.fullmatch(ref))
    return bool(_VERSION.fullmatch(ref))


def _date(text: str) -> date | None:
    if not _DATE.fullmatch(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def park(hub: Hub, store: Store, repo: str, number: int, kind: str, ref: str, until: str | date, *,
         today: date, role: str = ROLE) -> Park:
    """Park ticket `number` of `repo` on one event until a date, or `ParkError` with the reason
    it could never return; a refusal writes nothing to GitHub. The ticket is read again just
    before the write and left alone if it changed meanwhile (`ticket_changed`: a person's edit
    is never overwritten); after the write it is read back, and a second park that raced this
    one leaves its loser with `park_mismatch`."""

    try:
        made = make(kind, ref, until)
        if made.until < today:
            raise ParkError("until_past", f"{made.until.isoformat()} is before {today.isoformat()}")
        issue = _open_ticket(hub, repo, number)
        held = parse(issue.body, labels=issue.labels)
        if held.parked or held.park_lines:
            raise ParkError("already_parked", (held.park_lines or (f"the label `{PARKED_LABEL}`",))[0][:80])
        _check_it_can_come(hub, store, repo, number, made, today)
        body = f"{made.line}\n\n{issue.body.removeprefix(chr(0xFEFF)).lstrip()}"
        if len(body) > BODY_MAX:
            raise ParkError("body_too_long", f"{len(body)} characters")
        write_unchanged(hub, issue, body, [*issue.labels, PARKED_LABEL])
    except ParkError as refused:
        run = f"park-refused:{repo}#{number}@{datetime.now(UTC).isoformat()}"
        write_events(store, repo, number, role, run, [
            ("step.started", {"reason": PARKED_LABEL, "on": f"{kind} {ref}"[:120]}),
            ("step.ended", {"outcome": "refused", "reason": refused.code})])
        raise
    back = hub.issue(repo, number)
    written = parse(back.body, labels=back.labels) if back else None
    if written is None or written.park_lines != (made.line,) or not written.parked:
        raise ParkError("park_mismatch", "the ticket read back does not hold the line and the label")
    run = park_run(repo, number, hub.parked_at(repo, number))
    write_events(store, repo, number, role, run, [
        ("step.started", {"reason": PARKED_LABEL, "on": f"{made.kind} {made.ref}"}),
        ("ticket.parked", {"on": made.kind, "on_ref": made.ref, "until": made.until.isoformat()})])
    return made


def unpark(hub: Hub, store: Store, repo: str, number: int, *, now: datetime, role: str = ROLE) -> Park | None:
    """Take a ticket's park off by hand: its `Parked:` lines and the `parked` label in one
    request, every other label kept. Returns the park it held, None when its line was no park."""

    issue = _open_ticket(hub, repo, number, need_open=False)
    held = parse(issue.body, labels=issue.labels)
    if not held.parked and not held.park_lines:
        raise ParkError("not_parked", f"{repo}#{number}")
    found = held_park(held.park_lines)
    run = park_run(repo, number, hub.parked_at(repo, number))
    take_off(hub, issue)
    write_events(store, repo, number, role, run, [
        ("step.ended", {"outcome": "stopped", "reason": "hand"}),
        ("ticket.unparked", {"by": "hand", "returned_as": "hand", "on_ref": found.ref if found else None,
                             "late_minutes": late_minutes(found, now) if found else None})])
    return found


def held_park(lines: tuple[str, ...]) -> Park | None:
    """The park of a ticket's one `Parked:` line; None for no line, two lines or a bad one."""

    if len(lines) != 1:
        return None
    try:
        return read_line(lines[0])
    except ParkError:
        return None


def take_off(hub: Hub, issue: Issue, *, add: tuple[str, ...] = (), remove: tuple[str, ...] = ()) -> None:
    """Write the issue back without its `Parked:` lines and the `parked` label, in one request,
    adding and removing the labels named (case as GitHub compares); `ticket_changed` when it
    changed since `issue` was read."""

    gone = {PARKED_LABEL, *(label.lower() for label in remove)}
    labels = [label for label in issue.labels if label.lower() not in gone]
    labels += [label for label in add if label.lower() not in {held.lower() for held in labels}]
    write_unchanged(hub, issue, without_park_lines(issue.body), labels)


def write_unchanged(hub: Hub, issue: Issue, body: str, labels: list[str]) -> None:
    """Write body and labels in one request, only if the issue still holds what `issue` read:
    the edit of a person (or of another reader) between the read and the write is never lost.
    What is left is the moment between this read and GitHub's write; the next run sees it."""

    fresh = hub.issue(issue.repo, issue.number)
    if fresh is None or fresh.body != issue.body or sorted(fresh.labels) != sorted(issue.labels):
        raise ParkError("ticket_changed", f"{issue.repo}#{issue.number} changed while it was read")
    hub.write(issue.repo, issue.number, body, labels)


def park_run(repo: str, number: int, parked_at: datetime | None) -> str:
    """The run id of a park's `wait` step: the ticket and the moment its label was put on, read
    from GitHub, so the reader on any machine names the same step as the command that parked."""

    moment = parked_at.isoformat() if parked_at else "unknown"
    return f"park:{repo}#{number}@{moment}"


def late_minutes(found: Park, now: datetime) -> int:
    """Whole minutes past the end of the park's `until` day (UTC); 0 within it."""

    end = datetime.combine(found.until, datetime.min.time(), tzinfo=now.tzinfo or None)
    past = (now - end).total_seconds() / 60 - 24 * 60
    return max(0, int(past))


def _open_ticket(hub: Hub, repo: str, number: int, *, need_open: bool = True) -> Issue:
    issue = hub.issue(repo, number)
    if issue is None or issue.is_pr:
        raise ParkError("ticket_missing", f"{repo}#{number} is no ticket")
    if need_open and not issue.open:
        raise ParkError("ticket_not_open", f"{repo}#{number}")
    return issue


def _check_it_can_come(hub: Hub, store: Store, repo: str, number: int, made: Park, today: date) -> None:
    """Refuse a park whose event is not there to wait for, or has already happened."""

    ticket = made.ticket
    if ticket is not None:
        target = ticket.in_repo(repo)
        if ticket.names(repo, number, repo) and made.kind != "owner_answer":
            raise ParkError("ref_self", made.ref)
        other = hub.issue(*target)
        if other is None:
            raise ParkError("ref_missing", made.ref)
        if made.kind == "pr_ended" and not other.is_pr:
            raise ParkError("ref_not_pr", made.ref)
        if made.kind == "ticket_closed":
            if other.is_pr:
                raise ParkError("ref_not_ticket", made.ref)
            parent = hub.parent(repo, number)
            if parent is not None and ticket.names(*parent, repo):
                raise ParkError("ref_parent", f"{made.ref} is the parent of #{number}")
        if made.kind != "owner_answer" and not other.open:
            raise ParkError("event_passed", f"{made.ref} has ended")
    elif made.kind == "date":
        if made.date <= today:
            raise ParkError("event_passed", made.ref)
    elif made.kind == "run_ended":
        state = run_ended(hub, store, repo, made.ref)
        if state is None:
            raise ParkError("ref_missing", made.ref)
        if state.ended:
            raise ParkError("event_passed", f"run {made.ref} has ended")
    elif made.kind == "layer_version":
        held = hub.layer_version(repo)
        if held is not None and version_reached(held, made.ref):
            raise ParkError("event_passed", f"{repo} runs {held}")


def run_ended(hub: Hub, store: Store, repo: str, run: str) -> RunState | None:
    """The state of a GitHub Actions run of the repo (digits) or of a store run (a UUID, read
    from its `step` pair); None when no such run is known."""

    if _GITHUB_RUN.fullmatch(run):
        return hub.run(repo, run)
    rows = store.query("SELECT kind, outcome FROM events WHERE run = :run AND kind IN ('step.started', 'step.ended')",
                       {"run": run}).rows
    if not rows:
        return None
    ended = [row for row in rows if row[0] == "step.ended"]
    return RunState(ended=bool(ended), passed=bool(ended) and ended[0][1] == "passed")


def version_reached(held: str, wanted: str) -> bool:
    """Whether version `held` (`X.Y.Z`, anything after a `+` or `-` ignored) is at least `wanted`."""

    def parts(text: str) -> tuple[int, ...]:
        core = re.split(r"[+-]", text.strip(), maxsplit=1)[0]
        return tuple(int(part) if part.isdigit() else 0 for part in core.split("."))

    return parts(held) >= parts(wanted)


def write_events(store: Store, repo: str, number: int, role: str, run: str, events: list[tuple[str, dict]]) -> None:
    with store.writer() as writer:
        for kind, detail in events:
            writer.write({"kind": kind, "repo": repo, "ticket": number, "role": role, "session": None,
                          "parent_session": None, "turn": None, "step": "wait" if kind.startswith("step.") else None,
                          "run": run, "note": None, "detail": detail})
