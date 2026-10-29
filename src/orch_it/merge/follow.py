"""The ticket follows its pull request (#400 decision 3): `follow` reads one pull request and its
ticket from GitHub alone and moves the ticket once, by the `pr` and `close` rules of the pipeline
model. It is called by the workflow `follow` on pull request, review, comment and ticket events, by
`reconcile`, by the merge line on its head and by the park reader on a `pr_ended` park whose pull
request ended. Calling it again changes nothing: a pull request it took out of the line (auto-merge
off, a left mark) is `idle` until `pr open` puts it back.

The ticket of a pull request is the one its line `Refs #N` names; of several such lines, the one
whose park waits on this pull request; with none, the open ticket parked on it. A skills update's pull
request (branch `skills-upstream/<commit>-<base>`) with none of these is followed through the issue `skills
report` opened for its commit, on a red or no result only: that issue goes to the brain (#412).

Merged (`close` rules): the ticket is closed with its acceptance rows (`closed`), or, when the ticket
has rows marked `after merge`, its park comes off and it goes back to ready for those rows only
(`after_merge`); with no acceptance comment for this pull request (`acceptance_missing`), a failed
row (`row_failed`), a body changed since the rows were checked (`ticket_changed`) or a person's
comment on the ticket after them (`unanswered`) it goes to the brain and stays open; a ticket
already closed is only recorded (`already_closed`); a ticket that waits on another pull request
is not touched (`merged_other`). A skills update's pull request with no ticket of its own (`merged_alone`
otherwise) closes the open issue `skills report` opened for its commit, with `done_text` and no acceptance
table (`closed_skills`).

Closed unmerged: a ticket parked on it goes to the brain (`closed_unmerged`).

Open and in the line (a ready mark, or auto-merge on), in this order: the ticket closed
(`withdrawn`: out of the line) or edited since its rows were checked (`edited`: out of the line,
back to ready to be planned again); a person's comment on the pull request from its ready mark on
(`commented`) or on the ticket after its acceptance comment (`unanswered`): out of the line, the
ticket to the brain; a conflict (`conflict`): out of the line, the ticket back to ready with the
files, not counted; then the required checks of the head sha only (MERGE-12):

- red: the failed tests' ids from the runs' reports; every one also red on main (an open
  red-on-main ticket lists it) parks the ticket on that ticket, not counted (`red_on_main`);
  else the ticket goes back to ready with the ids, counted (`red`), or to the brain on the
  `red_in_a_row`th red in a row (`red_in_a_row`); a red that names no test is no result;
- no result (cut, cancelled, no report): run once more (`rerun`), never an attempt; no result again
  on the next attempt, or with no run to run again, goes to the brain (`no_result`);
- a check pending longer than `check_stuck_minutes` is cancelled (`stuck`), so it ends with no result;
- open past `pr_max_hours` from its ready mark: out of the line, to the brain (`timeout`);
- else `waiting` (a check pending or missing: no check is never green, MERGE-1), `behind` or `green`:
  nothing changes, the line brings it up to date and merges it.

"To ready" is the park off and `ready-for-agent` on; "to the brain" the park off, `needs-triage` on,
the ready labels off; "out of the line" auto-merge off and a left mark on the pull request. The
ticket is written first, through #398's write that refuses a ticket changed since it was read: when
another follow or a person changed it meanwhile nothing is written at all (`raced`) and the next
follow acts on it as it is then. Then the ticket's comment with the follow mark of its outcome, then
the pull request leaves the line, then the events: the end of the `pr` step (run
`pr:<repo>#<n>@<ready mark>`), the `close` pair, the park's `wait` end with `ticket.unparked`,
`ticket.closed`, `comment.written`, `attempt.counted`, `pr.commented`, `pr.conflicted`. A pull
request with no ticket only leaves the line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from orch_it.events.store import Store
from orch_it.merge import marks, record, red_main
from orch_it.merge.hub import FAILED, NO_RESULT, PENDING, Check, Pull, PullHub
from orch_it.merge.settings import MergeSettings
from orch_it.skills.update import BRANCH_PREFIX, ISSUE_TITLE, done_text
from orch_it.testruns.github import DEFAULT_REQUIRED_JOBS
from orch_it.tickets import park
from orch_it.tickets.hub import Issue
from orch_it.tickets.model import READY_LABEL, parse
from orch_it.tickets.reader import HUMAN_LABEL, TRIAGE_LABEL

ROLE = "github"
SHORT_SHA = 12
IDS_SHOWN = 100  # failing ids a comment lists; the rest are counted
ID_CHARS = 200  # a longer id is cut in a comment: 100 of them stay far under GitHub's 65536
_REFS = re.compile(r"(?m)^Refs #([1-9][0-9]{0,9})\b")
# how a ticket's park ends by what the follow did with it: `returned_as` -> (wait outcome, wait reason)
_PARK_END = {"due": ("passed", None), "other_way": ("failed", "other_way"), "comment": ("stopped", "comment"),
             "late": ("no_result", "late")}


class FollowError(LookupError):
    """`code`: `pr_missing` (no such pull request), `bad_options` (the command given both a pull
    request and an event, or neither)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


class _Raced(Exception):
    """The ticket changed between its read and its write: nothing was written."""


@dataclass(frozen=True)
class FollowReport:
    pr: int
    ticket: int | None
    outcome: str
    detail: str = ""


def follow(hub: PullHub, store: Store, repo: str, number: int, *, base: str = "main", settings: MergeSettings,
           now: datetime, role: str = ROLE, required: tuple[str, ...] = tuple(DEFAULT_REQUIRED_JOBS),
           owner: str | None = None) -> FollowReport:
    """Move the ticket of pull request `number` as its pull request stands now; see the module text.
    `owner`: the login whose comment is the owner's and the layer's (default: the repo's owner)."""

    one = _Follow.of(hub, store, repo, number, settings, now, role, owner, required)
    try:
        if one.pull.state == "merged":
            return one.merged()
        if one.pull.state == "closed":
            return one.closed()
        if one.pull.base != base:
            return one.report("idle", f"into {one.pull.base}, not {base}")
        return one.open()
    except _Raced as e:
        return one.report("raced", str(e))


def follow_conflict(hub: PullHub, store: Store, repo: str, number: int, files: list[str], *,
                    settings: MergeSettings, now: datetime, role: str = ROLE, owner: str | None = None,
                    other_pr: int | None = None) -> FollowReport:
    """The conflict rule for a pull request the merge line could not bring up to date: out of the
    line, its ticket back to ready with the files, not counted."""

    one = _Follow.of(hub, store, repo, number, settings, now, role, owner, tuple(DEFAULT_REQUIRED_JOBS))
    try:
        return one.conflict(files, other_pr)
    except _Raced as e:
        return one.report("raced", str(e))


def refs_of(body: str) -> list[int]:
    """Every ticket a pull request's body names with a line `Refs #N`, in body order."""

    return list(dict.fromkeys(int(n) for n in _REFS.findall(body or "")))


def ticket_of(hub: PullHub, repo: str, pull: Pull) -> Issue | None:
    """The ticket of a pull request: of its `Refs` tickets the one parked on it, else the first;
    with no `Refs` line, the open ticket parked on it."""

    named = [found for n in refs_of(pull.body) if (found := hub.issue(repo, n)) is not None and not found.is_pr]
    waiting = [issue for issue in named if _parked_on(issue, repo, pull.number)]
    if waiting or named:
        return (waiting or named)[0]
    for number in hub.parked(repo):
        issue = hub.issue(repo, number)
        if issue is not None and issue.open and not issue.is_pr and _parked_on(issue, repo, pull.number):
            return issue
    return None


_SKILLS_HEAD = re.compile(re.escape(BRANCH_PREFIX) + r"([0-9a-f]{12})-[0-9a-f]+")


def skills_issue_of(hub: PullHub, repo: str, pull: Pull) -> Issue | None:
    """The open issue `skills report` opened for the upstream commit of a skills update's pull request
    (its branch `skills-upstream/<commit>-<base>`), found by its title; None for any other pull request."""

    found = _SKILLS_HEAD.fullmatch(pull.branch)
    if found is None:
        return None
    title = ISSUE_TITLE + found.group(1)
    for label in (TRIAGE_LABEL, READY_LABEL):
        for number in hub.issues_with_label(repo, label):
            issue = hub.issue(repo, number)
            if issue is not None and issue.open and not issue.is_pr and issue.title == title:
                return issue
    return None


def pr_of_event(event: dict, hub: PullHub, repo: str) -> tuple[int | None, str]:
    """The pull request a GitHub event of the `follow` workflow is about, and why when none: a
    pull request, review or review comment event names it; a comment on a pull request is about
    it; a comment on, edit of or close of a ticket is about the pull request its `pr_ended` park
    names."""

    pull = event.get("pull_request")
    if isinstance(pull, dict) and isinstance(pull.get("number"), int):
        return pull["number"], ""
    issue = event.get("issue")
    if not isinstance(issue, dict) or not isinstance(issue.get("number"), int):
        return None, "no pull request or issue in the event"
    if isinstance(issue.get("pull_request"), dict):
        return issue["number"], ""
    found = hub.issue(repo, issue["number"])
    held = park.held_park(parse(found.body).park_lines) if found is not None else None
    ref = held.ticket if held is not None and held.kind == "pr_ended" else None
    if ref is None or (ref.repo and ref.repo.lower() != repo.lower()):
        return None, f"#{issue['number']} waits on no pull request of {repo}"
    return ref.number, ""


def _parked_on(issue: Issue, repo: str, pr: int) -> bool:
    found = park.held_park(parse(issue.body).park_lines)
    ref = found.ticket if found is not None and found.kind == "pr_ended" else None
    return ref is not None and ref.names(repo, pr, repo)


def _ended(outcome: str, reason: str | None, **detail) -> dict:
    return {"outcome": outcome, **({"reason": reason} if reason else {}), **detail}


def _listed(ids: list[str]) -> str:
    shown = "\n".join(f"- `{test_id[:ID_CHARS]}`" for test_id in ids[:IDS_SHOWN])
    more = len(ids) - IDS_SHOWN
    return shown + (f"\n- and {more} more (the run's reports name them)" if more > 0 else "")


class _Follow:
    def __init__(self, hub: PullHub, store: Store, repo: str, pull: Pull, issue: Issue | None,
                 settings: MergeSettings, now: datetime, role: str, owner: str, required: tuple[str, ...]):
        self.hub, self.store, self.repo, self.pull, self.issue = hub, store, repo, pull, issue
        self.settings, self.now, self.role, self.owner, self.required = settings, now, role, owner, required
        self.said = hub.pull_comments(repo, pull.number)
        started = marks.last_ready(marks.of_layer(self.said, owner))
        self.run = record.pr_run(repo, pull.number, started.at if started else None)
        self.skills = False  # the ticket is the issue of a skills update (`skills_issue_of`)

    @classmethod
    def of(cls, hub, store, repo, number, settings, now, role, owner, required) -> _Follow:
        pull = hub.pull(repo, number)
        if pull is None:
            raise FollowError("pr_missing", f"{repo}#{number} is no pull request")
        return cls(hub, store, repo, pull, ticket_of(hub, repo, pull), settings, now, role,
                   (owner or repo.split("/")[0]).lower(), required)

    @property
    def ticket(self) -> int | None:
        return self.issue.number if self.issue else None

    def report(self, outcome: str, detail: str = "") -> FollowReport:
        return FollowReport(self.pull.number, self.ticket, outcome, detail)

    # merged: the close rules

    def merged(self) -> FollowReport:
        merge = {"merge_sha": self.pull.merge_sha}
        if self.issue is None and self._adopt_skills_issue():
            return self._closed_skills(merge)
        if self.issue is None:
            self._end_pr("passed", "merged", merge)
            return self.report("merged_alone")
        if not self.issue.open:
            self._close_step("refused", "already_closed")
            self._end_pr("passed", "merged", merge)
            return self.report("already_closed")
        held = parse(self.issue.body, labels=self.issue.labels)
        if (held.parked or held.park_lines) and not _parked_on(self.issue, self.repo, self.pull.number):
            self._end_pr("passed", "merged", merge)
            return self.report("merged_other", f"#{self.issue.number} waits on another pull request")
        accepted, why = self._acceptance_judged()
        if why:
            self._to_brain(why[0], f"#{self.pull.number} merged, but the ticket is not closed: {why[1]}.", "other_way")
            self._close_step("failed", why[0])
            self._end_pr("passed", "merged", merge)
            return self.report(*why)
        later = [row.id for row in held.acceptance or () if row.after_merge]
        sha = (self.pull.merge_sha or "")[:SHORT_SHA]
        if later:
            rows, many = ", ".join(later), len(later) > 1
            self._to_ready("after_merge", f"#{self.pull.number} merged at `{sha}`. Only row{'s' if many else ''} {rows} "
                           f"{'are' if many else 'is'} left: run on main, then close the ticket.", "due")
            self._close_step("passed", None)
            self._end_pr("passed", "merged", merge)
            return self.report("after_merge", f"rows {rows}")
        issue = self._fresh()
        self._end_park(issue, "due", write=lambda: park.take_off(self.hub, issue))
        table = "".join(f"| {row.id} | {row.result} | {marks.cell(row.evidence)} |\n" for row in accepted.rows)
        self.hub.comment(self.repo, issue.number,
                         f"{marks.follow_mark('closed')}Closed: #{self.pull.number} merged at `{sha}` with every "
                         f"acceptance row checked.\n\n| row | result | evidence |\n|---|---|---|\n{table}")
        self.hub.close_issue(self.repo, issue.number, "completed")
        passed = sum(row.result in ("pass", "na") for row in accepted.rows)
        self._write(None, None, [("comment.written", {"comment_kind": "closing"}),
                                 record.ticket_closed(len(accepted.rows), passed)])
        self._close_step("passed", None)
        self._end_pr("passed", "merged", merge)
        return self.report("closed", f"rows {passed}/{len(accepted.rows)}")

    def _closed_skills(self, merge: dict) -> FollowReport:
        """A skills update's pull request merged: its skills issue (no acceptance table to judge) gets the closing
        comment `done_text` of the commit in its branch, and closes as completed. Like the ticket's close, the issue
        is read again and refused when it changed since it was read (`_Raced`, nothing written); a comment an
        earlier follow posted before its close was refused is not posted a second time."""

        commit = _SKILLS_HEAD.fullmatch(self.pull.branch).group(1)
        text = marks.follow_mark("closed") + done_text(commit)
        issue = self._fresh()
        if issue is None or not issue.open:
            self._close_step("refused", "already_closed")
            self._end_pr("passed", "merged", merge)
            return self.report("already_closed")
        if (issue.body, issue.labels) != (self.issue.body, self.issue.labels):
            raise _Raced(f"#{issue.number} changed since it was read")
        if not any(said.body.startswith(text) for said in self.hub.comments(self.repo, issue.number)):
            self.hub.comment(self.repo, issue.number, text)
            self._write(None, None, [("comment.written", {"comment_kind": "closing"})])
        self.hub.close_issue(self.repo, issue.number, "completed")
        self._write(None, None, [record.ticket_closed(None, None)])
        self._close_step("passed", None)
        self._end_pr("passed", "merged", merge)
        return self.report("closed_skills", commit)

    def _acceptance_judged(self) -> tuple[marks.Acceptance | None, tuple[str, str] | None]:
        said = self.hub.comments(self.repo, self.issue.number)
        accepted = marks.acceptance(marks.of_layer(said, self.owner))
        if accepted is None or accepted.pr != self.pull.number:
            return accepted, ("acceptance_missing", f"no acceptance comment for #{self.pull.number}")
        failed = [row.id for row in accepted.rows if row.result == "fail"]
        if failed:
            return accepted, ("row_failed", f"row {', '.join(failed)} failed")
        if accepted.body != marks.body_hash(self.issue.body):
            return accepted, ("ticket_changed", "the ticket was edited after its rows were checked")
        if self._a_person_said(said, accepted.at):
            return accepted, ("unanswered", "a person's comment on the ticket after its acceptance is not answered")
        return accepted, None

    # closed unmerged

    def closed(self) -> FollowReport:
        if self.issue is None or not self.issue.open or not _parked_on(self.issue, self.repo, self.pull.number):
            self._end_pr("failed", "closed_unmerged")
            return self.report("closed_unmerged", "no ticket waits on it")
        self._to_brain("closed_unmerged", f"#{self.pull.number} was closed without a merge.", "other_way")
        self._end_pr("failed", "closed_unmerged")
        return self.report("closed_unmerged")

    # open

    def open(self) -> FollowReport:
        ready = marks.ready_since(marks.of_layer(self.said, self.owner))
        if ready is None and self.pull.auto_merge is None:
            return self.report("idle", "not in the merge line")
        since = ready.at if ready else self.pull.created_at
        if self.issue is not None and not self.issue.open:
            self._leave("withdrawn", f"its ticket #{self.issue.number} was closed")
            self._end_pr("stopped", "withdrawn")
            return self.report("withdrawn")
        if self.issue is not None:
            said = self.hub.comments(self.repo, self.issue.number)
            accepted = marks.acceptance(marks.of_layer(said, self.owner))
            if accepted is not None and accepted.pr == self.pull.number:
                if accepted.body != marks.body_hash(self.issue.body):
                    self._to_ready("edited", f"The ticket was edited while #{self.pull.number} waited: plan it "
                                   "again, then `pr open` puts the pull request back in the line.", "other_way")
                    self._leave("edited", f"its ticket #{self.issue.number} was edited after its rows were checked")
                    self._end_pr("stopped", "withdrawn")
                    return self.report("edited")
                if self._a_person_said(said, accepted.at):
                    self._to_brain("unanswered", f"A person commented here while #{self.pull.number} waited to "
                                   "merge: it is out of the merge line until the brain answers.", "comment")
                    self._leave("unanswered", f"a person commented on its ticket #{self.issue.number}")
                    self._end_pr("stopped", "withdrawn")
                    return self.report("unanswered")
        people = [c for c in self.said if marks.is_person(c, self.owner) and c.at >= since]
        if people:
            return self._commented(people[0])
        if self.pull.mergeable == "conflicting":
            return self.conflict(self.hub.conflict_files(self.repo, self.pull.base, self.pull.number))
        checks = {check.name: check for check in self.hub.checks(self.repo, self.pull.head_sha)}
        found = [checks.get(name) for name in self.required]
        if any(c is not None and c.state == FAILED for c in found):
            return self._red([c for c in found if c is not None and c.state == FAILED])
        if any(c is not None and c.state == NO_RESULT for c in found):
            return self._no_result([c for c in found if c is not None and c.state == NO_RESULT])
        bound = timedelta(minutes=self.settings.check_stuck_minutes)
        stuck = [c for c in found if c is not None and c.state == PENDING and c.started_at
                 and self.now - c.started_at > bound]
        if stuck:
            for run_id in dict.fromkeys(c.run_id for c in stuck if c.run_id):
                self.hub.cancel(self.repo, run_id)
            return self.report("stuck", ", ".join(c.name for c in stuck))
        if self.now - since > timedelta(hours=self.settings.pr_max_hours):
            hours = f"{self.settings.pr_max_hours:g}"
            self._to_brain("timeout", f"#{self.pull.number} has waited more than {hours} hours in the merge line "
                           "without merging.", "late")
            self._leave("timeout", f"open more than {hours} hours in the line")
            self._end_pr("no_result", "timeout")
            return self.report("timeout")
        if any(c is None or c.state == PENDING for c in found):
            return self.report("waiting", ", ".join(n for n, c in zip(self.required, found) if c is None or
                                                    c.state == PENDING))
        if self.pull.behind:
            return self.report("behind", f"{self.pull.behind} commits")
        return self.report("green")

    def conflict(self, files: list[str], other_pr: int | None = None) -> FollowReport:
        shown = "\n".join(f"- `{name[:ID_CHARS]}`" for name in files[:IDS_SHOWN]) or "- (GitHub names no file)"
        if self._is_open_ticket():
            self._to_ready("conflict", f"#{self.pull.number} conflicts with `{self.pull.base}` in these files: merge "
                           f"`{self.pull.base}` into the branch, resolve them, and `pr open` again.\n\n{shown}",
                           "other_way")
        self._leave("conflict", f"it conflicts with {self.pull.base} in {len(files)} files")
        self._write(None, None, [("pr.conflicted", {"pr": self.pull.number, "files": files[:IDS_SHOWN],
                                                    "other_pr": other_pr})])
        self._attempt("conflict", False, 0)
        self._end_pr("refused", "conflict")
        return self.report("conflict", ", ".join(files[:IDS_SHOWN]))

    def _commented(self, first) -> FollowReport:
        if self._is_open_ticket():
            self._to_brain("commented", f"{first.author} commented on #{self.pull.number}: it is out of the merge "
                           "line until the brain answers.", "comment")
        self._leave("commented", f"{first.author} commented on it")
        role = "owner" if first.author.lower() == self.owner else ROLE
        self._write(None, None, [("pr.commented", {"pr": self.pull.number, "author_role": role, "answered": False})])
        self._end_pr("stopped", "withdrawn")
        return self.report("commented", first.author)

    def _adopt_skills_issue(self) -> bool:
        """A skills update's pull request with no ticket of its own is followed through the issue `skills
        report` opened for its commit (`skills_issue_of`); True when this pull request is one."""

        if self.issue is None:
            self.issue = skills_issue_of(self.hub, self.repo, self.pull)
            self.skills = self.issue is not None
        return self.skills

    def _red(self, failed: list[Check]) -> FollowReport:
        got = {run_id: self.hub.failing_ids(self.repo, run_id) for run_id in dict.fromkeys(c.run_id for c in failed)
               if run_id}
        ids = sorted({test_id for named in got.values() if named for test_id in named})
        if not ids:  # a red that names no test is no result, never an attempt (MERGE-5)
            return self._no_result(failed)
        names = ", ".join(c.name for c in failed)
        skills = self._adopt_skills_issue()
        if not self._is_open_ticket():
            self._leave("red", f"{names} red")
            self._end_pr("failed", "red")
            return self.report("red", f"{len(ids)} tests")
        failing = {red_main.normal(test_id) for test_id in ids}
        covering = sorted(red.number for red in red_main.open_reds(self.hub, self.repo) if red.covers(failing))
        if covering and self._park_on_red_main(covering[0], _listed(ids)):
            self._leave("red_on_main", f"red only on tests red on main (#{covering[0]})")
            self._attempt("red_on_main", False, 0)
            self._end_pr("failed", "red")
            return self.report("red_on_main", f"#{covering[0]}")
        reds = 0
        for mark in reversed(marks.follows(marks.of_layer(self.hub.comments(self.repo, self.issue.number), self.owner))):
            if mark.outcome != "red":
                break
            reds += 1
        in_a_row = reds + 1
        head = self.pull.head_sha[:SHORT_SHA]
        if skills:  # a skills update's red goes to the brain at once, whatever its count
            self._to_brain("red", f"#{self.pull.number} (a skills update) is red at `{head}` on these tests:\n\n"
                           f"{_listed(ids)}", "other_way")
            outcome = "red"
        elif in_a_row >= self.settings.red_in_a_row:
            self._to_brain("red_in_a_row", f"#{self.pull.number} is red for the {in_a_row}th time in a row:\n\n"
                           f"{_listed(ids)}", "other_way")
            outcome = "red_in_a_row"
        else:
            self._to_ready("red", f"#{self.pull.number} is red at `{head}` on these tests:\n\n{_listed(ids)}",
                           "other_way")
            outcome = "red"
        self._leave("red", f"{names} red on `{head}`")
        self._attempt("red", True, in_a_row)
        self._end_pr("failed", "red")
        return self.report(outcome, f"{len(ids)} tests")

    def _no_result(self, checks: list[Check]) -> FollowReport:
        self._adopt_skills_issue()
        runs = list(dict.fromkeys(c.run_id for c in checks if c.run_id))
        if max(c.attempt for c in checks) >= 2 or not runs:
            if self._is_open_ticket():
                again = "twice" if runs else "with no run to run again"
                self._to_brain("no_result", f"The checks of #{self.pull.number} gave no result {again} "
                               f"({', '.join(c.name for c in checks)}).", "late")
            self._leave("no_result", f"no result on `{self.pull.head_sha[:SHORT_SHA]}`")
            self._attempt("no_result", False, 0)
            self._end_pr("no_result", None)
            return self.report("no_result")
        for run_id in runs:
            self.hub.rerun(self.repo, run_id)
        self._attempt("no_result", False, 0)
        return self.report("rerun", ", ".join(c.name for c in checks))

    # moving the ticket

    def _park_on_red_main(self, red: int, listed: str) -> bool:
        """Park the ticket on the red-on-main ticket; False (nothing changed) when that park is
        refused, such as when the red ticket closed a moment ago."""

        issue = self._fresh()
        until = self.now.date() + timedelta(days=self.settings.red_main_park_days)
        try:
            park.make("ticket_closed", f"#{red}", until)
            other = self.hub.issue(self.repo, red)
            if other is None or not other.open:
                return False
            self._end_park(issue, "other_way", write=lambda: park.take_off(self.hub, issue))
            park.park(self.hub, self.store, self.repo, issue.number, "ticket_closed", f"#{red}", until,
                      today=self.now.date(), role=self.role)
        except park.ParkError as e:
            if e.code == "ticket_changed":
                raise _Raced(str(e)) from None
            # the red ticket ended between the read and the park: the ticket, already unparked, is red on its own
            self.issue = self._fresh()
            return False
        self.hub.comment(self.repo, issue.number,
                         f"{marks.follow_mark('red_on_main')}#{self.pull.number} is red only on tests red on main, "
                         f"listed in #{red}: it waits for that ticket, not counted.\n\n{listed}")
        return True

    def _to_ready(self, outcome: str, text: str, returned_as: str) -> None:
        self._move(outcome, text, returned_as, add=(READY_LABEL,), remove=(TRIAGE_LABEL,), kind="send_back")

    def _to_brain(self, outcome: str, text: str, returned_as: str) -> None:
        self._move(outcome, f"To the brain: {text}", returned_as, add=(TRIAGE_LABEL,),
                   remove=(READY_LABEL, HUMAN_LABEL), kind="machine")

    def _move(self, outcome: str, text: str, returned_as: str, *, add, remove, kind: str) -> None:
        """The ticket's park off and its labels set in one write that refuses a ticket changed
        since it was read (`_Raced`, nothing written); then its comment and events."""

        issue = self._fresh()
        if issue is None or not issue.open:
            return
        self._end_park(issue, returned_as, write=lambda: park.take_off(self.hub, issue, add=add, remove=remove))
        self.hub.comment(self.repo, issue.number, f"{marks.follow_mark(outcome)}{text}")
        self._write(None, None, [("comment.written", {"comment_kind": kind})])

    def _end_park(self, issue: Issue, returned_as: str, *, write) -> None:
        """Run `write` (the one write of the ticket), then end the `wait` of its park, if it held
        one, with `ticket.unparked`."""

        held = parse(issue.body, labels=issue.labels)
        parked = held.parked or bool(held.park_lines)
        run = park.park_run(self.repo, issue.number, self.hub.parked_at(self.repo, issue.number)) if parked else None
        try:
            write()
        except park.ParkError as e:
            if e.code == "ticket_changed":
                raise _Raced(str(e)) from None
            raise
        if not parked:
            return
        found = park.held_park(held.park_lines)
        outcome, reason = _PARK_END[returned_as]
        self._write(run, "wait", [("step.ended", _ended(outcome, reason))])
        self._write(run, None, [("ticket.unparked", {
            "by": "event", "returned_as": returned_as, "on_ref": found.ref if found else None,
            "late_minutes": park.late_minutes(found, self.now) if found else None})])

    def _fresh(self) -> Issue | None:
        return self.hub.issue(self.repo, self.issue.number) if self.issue else None

    def _is_open_ticket(self) -> bool:
        return self.issue is not None and self.issue.open

    def _leave(self, reason: str, text: str) -> None:
        """Take the pull request out of the merge line: auto-merge off, then the left mark."""

        self.hub.set_auto_merge(self.repo, self.pull.number, None)
        self.hub.comment(self.repo, self.pull.number, marks.left_comment(reason, text))

    def _a_person_said(self, said, after: datetime) -> bool:
        return any(marks.is_person(c, self.owner) and c.at > after for c in said)

    # the store

    def _write(self, run: str | None, step: str | None, events: list[tuple[str, dict]]) -> None:
        record.write(self.store, self.repo, self.ticket, self.role, run, step, events)

    def _attempt(self, reason: str, counted: bool, in_a_row: int) -> None:
        self._write(None, None, [("attempt.counted", {"reason": reason, "counted": counted, "in_a_row": in_a_row})])

    def _end_pr(self, outcome: str, reason: str | None, detail: dict | None = None) -> None:
        self._write(self.run, "pr", [("step.ended", _ended(outcome, reason, **(detail or {})))])

    def _close_step(self, outcome: str, reason: str | None) -> None:
        run = f"close:{self.repo}#{self.ticket}@{self.pull.number}"
        self._write(run, "close", [("step.started", {"ticket": self.ticket, "pr": self.pull.number}),
                                   ("step.ended", _ended(outcome, reason))])
