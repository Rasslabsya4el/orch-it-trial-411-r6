"""The merge line (#400 decision 2): `run_line` lands a repo's green pull requests on its base one by
one, the oldest ready first, under the repo's line lock, run on the machine under the owner's `gh`
(so every update and merge starts the required checks).

A member of the line is an open, non-draft pull request into the base, its head in this repo (no
fork), with a ready mark that no left mark followed, or with auto-merge on; its ready time is its
ready mark's (else its creation). One run, until the line is empty or `line_minutes` pass:

1. every conflicting member leaves the line at once and its ticket goes back to ready with the
   files (the follow's conflict rule), so a conflict never holds the members behind it;
2. the head is the oldest ready member. Every other member that is up to date with the base has
   its auto-merge turned off for as long as it is (held): GitHub's own auto-merge would merge it
   as soon as it is green, out of the line's order, and make the head behind again. A held member
   gets it back once it is behind (the head merged) or is the head, and at the end of the run;
3. the head is served: behind → brought up to date (`update_branch`, at most `max_updates` times a
   run, else it waits for the next run: `starved`); a conflict on the update → the conflict rule;
   then its required checks of the head sha are waited for, polling every `poll_seconds`, at most
   `check_minutes` (`bound`: it stays in the line for the next run, the next member goes), with the
   follow run at each poll (a person's comment, an edit of its ticket, a red, no result: the follow
   moves the ticket and the head `left`); green and up to date → its title and body are read fresh,
   and a closing keyword in either refuses it (`refused`: auto-merge off, a left mark with the reason,
   nothing merged, the line goes on); else merged with its head sha, and the follow closes its ticket at
   once (GitHub refusing the merge more than `max_updates` times: `refused`, left for the next run). A held member gets its auto-merge back only after its follow,
   so what was said or done to it while held takes it out of the line instead.

Events: a `line` step, run `line:<repo>@<start>` (started `base`, `prs`; ended `merged`, `left`,
reason `empty`, `next_run` or `bound`; `failed` with `error` when the run breaks; `refused` with
`lock_busy`/`lock_stale` when the lock is not had) and one `line.served` per turn under that run
(`pr`, `result`, `followed`, `waited_minutes`, `updates`), plus `pr.updated` per update.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from orch_it.events.store import Store
from orch_it.merge import lock, marks, record
from orch_it.merge.follow import FollowReport, follow, follow_conflict
from orch_it.merge.hub import Pull, PullHub
from orch_it.merge.pr_open import refers_to
from orch_it.merge.settings import MergeSettings
from orch_it.testruns.github import DEFAULT_REQUIRED_JOBS
from orch_it.tickets.park import ROLE

# what the follow says of a member that is still in the line: the line goes on serving it (a
# `raced` follow wrote nothing: the member waits a poll and is followed again)
_STAYS = ("waiting", "green", "behind", "rerun", "stuck", "raced")
LINE_ERROR_CODES = ("closing_keyword",)  # a merge the line will not make


class LineError(ValueError):
    """A merge the line will not make; `code` is one of `LINE_ERROR_CODES`, `detail` says why. The line
    takes the error in and turns it into that pull request's `refused` turn; it does not stop the run."""

    def __init__(self, code: str, detail: str = ""):
        if code not in LINE_ERROR_CODES:
            raise ValueError(f"not a merge line error code: {code}")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def _hit_lines(text: str, hits: list[str]) -> list[int]:
    """The 1-based line each closing keyword hit (as `marks.closing_keywords` returned it, in order) starts
    on in `text`."""

    lines, at = [], 0
    for hit in hits:
        at = text.index(hit, at)
        lines.append(text.count("\n", 0, at) + 1)
        at += len(hit)
    return lines


@dataclass(frozen=True)
class Served:
    pr: int
    ticket: int | None
    # a `line_result`: merged, left, starved (main moved too often), bound, refused (GitHub would not merge,
    # or a closing keyword in its text)
    result: str
    waited_minutes: float  # from its ready time to the end of its turn
    updates: int
    detail: str = ""  # the follow's outcome when it left


@dataclass(frozen=True)
class LineReport:
    repo: str
    base: str
    served: tuple[Served, ...]
    reason: str  # `empty`: no member left; `next_run`: the ones left wait for the next run; `bound`: the run's bound

    @property
    def merged(self) -> int:
        return sum(s.result == "merged" for s in self.served)

    @property
    def left(self) -> int:
        return sum(s.result == "left" for s in self.served)


@dataclass
class _Member:
    pull: Pull
    ready_at: datetime
    ticket: int | None


@dataclass
class _Line:
    hub: PullHub
    store: Store
    repo: str
    base: str
    settings: MergeSettings
    clock: Callable[[], datetime]
    sleep: Callable[[float], None]
    role: str
    required: tuple[str, ...]
    owner: str
    deadline: datetime
    run_id: str
    held: dict[int, str] = field(default_factory=dict)  # pr -> the auto-merge method it had
    for_next_run: set[int] = field(default_factory=set)  # members this run leaves for the next one
    turns: list[Served] = field(default_factory=list)
    ready_at: dict[int, datetime] = field(default_factory=dict)  # pr -> its ready time, as members() read it

    def run(self) -> tuple[list[Served], str]:
        turns = self.turns
        while self.clock() < self.deadline:
            members = self.members()
            for member in [m for m in members if m.pull.mergeable == "conflicting"]:
                self.end_turn(turns, self.conflict(member, self.hub.conflict_files(self.repo, self.base,
                                                                                    member.pull.number)))
            members = [m for m in self.members() if m.pull.number not in self.for_next_run]
            if not members:
                return turns, "next_run" if self.for_next_run else "empty"
            self.hold_others(members[0].pull.number, members)
            self.end_turn(turns, self.serve(members[0]))
        return turns, "bound"

    def end_turn(self, turns: list[Served], turn: Served) -> None:
        turns.append(turn)
        record.write(self.store, self.repo, turn.ticket, self.role, self.run_id, None, [("line.served", {
            "pr": turn.pr, "result": turn.result, "followed": turn.detail or None,
            "waited_minutes": turn.waited_minutes, "updates": turn.updates})])

    # members and holds

    def members(self) -> list[_Member]:
        found = []
        for pull in self.hub.open_pulls(self.repo, self.base):
            if pull.draft or (pull.head_repo or "").lower() != self.repo.lower():
                continue
            ready = marks.ready_since(marks.of_layer(self.hub.pull_comments(self.repo, pull.number), self.owner))
            if ready is None and pull.auto_merge is None and pull.number not in self.held:
                continue
            found.append(_Member(pull, ready.at if ready else pull.created_at, refers_to(pull.body)))
            self.ready_at[pull.number] = found[-1].ready_at
        return sorted(found, key=lambda m: (m.ready_at, m.pull.number))

    def hold_others(self, head: int, members: list[_Member] | None = None) -> None:
        """Auto-merge off on every member but the head while it is up to date; back on once it is
        behind (the head is given it back by `serve`, after its follow)."""

        for member in members if members is not None else self.members():
            number, pull = member.pull.number, member.pull
            if number == head:
                continue
            if pull.behind:
                self.give_back(number)
            elif pull.auto_merge is not None and number not in self.held:
                self.hub.set_auto_merge(self.repo, number, None)
                self.held[number] = pull.auto_merge

    def give_back(self, number: int) -> None:
        """A held member's auto-merge back, after its follow: what was said or done to it while it
        was held (a person's comment, its ticket closed) takes it out of the line instead."""

        method = self.held.pop(number, None)
        if method is None:
            return
        pull = self.hub.pull(self.repo, number)
        if pull is None or pull.state != "open":
            return
        done = self.follow(number)
        if done.outcome in _STAYS:
            self.hub.set_auto_merge(self.repo, number, method)
        else:  # it left the line while it was held: its turn ends here
            member = _Member(pull, self.ready_at.get(number, pull.created_at), refers_to(pull.body))
            self.end_turn(self.turns, self.turn(member, "left", 0, done.outcome))

    def release_all(self) -> None:
        for number in list(self.held):
            try:
                self.give_back(number)
            except Exception:  # one that fails stays without auto-merge; its ready mark keeps it in the line
                self.held.pop(number, None)

    # the head

    def serve(self, member: _Member) -> Served:
        number, updates, refused = member.pull.number, 0, 0
        turn_ends = min(self.deadline, self.clock() + timedelta(minutes=self.settings.check_minutes))
        while True:
            pull = self.hub.pull(self.repo, number)
            if pull is None or pull.state != "open":  # merged by GitHub's auto-merge, or closed by a person
                if pull is not None:
                    self.follow(number)
                result = "merged" if pull is not None and pull.state == "merged" else "left"
                return self.turn(member, result, updates, "" if result == "merged" else "closed")
            done = self.follow(number)
            if done.outcome not in _STAYS:
                self.held.pop(number, None)
                return self.turn(member, "left", updates, done.outcome)
            self.give_back(number) if number in self.held else None
            if pull.behind:  # a run on a head behind the base would be thrown away: update now
                if updates >= self.settings.max_updates:
                    self.for_next_run.add(number)
                    return self.turn(member, "starved", updates)
                if not self.hub.update_branch(self.repo, number, pull.head_sha):
                    return self.conflict(member, self.hub.conflict_files(self.repo, self.base, number), updates)
                updates += 1
                fresh = self.hub.pull(self.repo, number)
                record.write(self.store, self.repo, member.ticket, self.role, self.run_id, None,
                             [record.pr_updated(number, pull.head_sha, fresh.head_sha if fresh else None)])
                turn_ends = min(self.deadline, self.clock() + timedelta(minutes=self.settings.check_minutes))
                continue
            if done.outcome == "green" and pull.behind == 0:
                try:
                    self.closing_check(number)
                except LineError as refusal:
                    return self.refuse(member, refusal, updates)
                sha = self.hub.merge(self.repo, number, pull.head_sha, pull.auto_merge or self.settings.merge_method)
                if sha:
                    self.follow(number)  # its ticket closes now, whether the workflow runs or not
                    return self.turn(member, "merged", updates)
                refused += 1  # the base moved or a check changed between the read and the merge: read again
                if refused > self.settings.max_updates:
                    self.for_next_run.add(number)
                    return self.turn(member, "refused", updates)
            if self.clock() >= turn_ends:
                self.for_next_run.add(number)
                return self.turn(member, "bound", updates, done.outcome)
            self.sleep(self.settings.poll_seconds)
            self.hold_others(number)

    def follow(self, number: int) -> FollowReport:
        return follow(self.hub, self.store, self.repo, number, base=self.base, settings=self.settings,
                      now=self.clock(), role=self.role, required=self.required, owner=self.owner)

    def closing_check(self, number: int) -> None:
        """Before the merge: the title and body as they are now (a fresh read, so an edit after `pr open`
        counts). A closing keyword in either raises `LineError`, naming each keyword, its reference, the text
        and the line it starts on."""

        now = self.hub.pull(self.repo, number)
        if now is None:
            return
        found = []
        for name, text in (("title", now.title), ("body", now.body)):
            hits = marks.closing_keywords(text)
            for hit, at_line in zip(hits, _hit_lines(text, hits)):
                found.append(f"the {name} at line {at_line} says {hit}")
        if found:
            raise LineError("closing_keyword", f"#{number}: {'; '.join(found)}: write `Refs <URL>` in its place")

    def refuse(self, member: _Member, refusal: LineError, updates: int) -> Served:
        """A pull request whose text closes a ticket by a word leaves the line the way `follow._leave` takes
        one out: auto-merge off, then a left mark carrying the reason, so no later run serves it until a new
        ready mark. The line goes on with the pull requests behind it; the ticket stays open."""

        number = member.pull.number
        self.held.pop(number, None)
        self.hub.set_auto_merge(self.repo, number, None)
        self.hub.comment(self.repo, number, marks.left_comment(refusal.code, refusal.detail))
        return self.turn(member, "refused", updates, refusal.detail)

    def conflict(self, member: _Member, files: list[str], updates: int = 0) -> Served:
        done = follow_conflict(self.hub, self.store, self.repo, member.pull.number, files, settings=self.settings,
                               now=self.clock(), role=self.role, owner=self.owner)
        self.held.pop(member.pull.number, None)
        if done.outcome == "raced":
            self.for_next_run.add(member.pull.number)
        return self.turn(member, "left", updates, done.outcome)

    def turn(self, member: _Member, result: str, updates: int, detail: str = "") -> Served:
        waited = (self.clock() - member.ready_at).total_seconds() / 60
        return Served(member.pull.number, member.ticket, result, round(waited, 1), updates, detail)


def run_line(hub: PullHub, store: Store, repo: str, base: str, *, settings: MergeSettings,
             clock: Callable[[], datetime] = lambda: datetime.now(UTC), sleep: Callable[[float], None] = time.sleep,
             root: Path | None = None, holder: str | None = None, on_wait: Callable[[lock.Holder], None] | None = None,
             role: str = ROLE, required: tuple[str, ...] = tuple(DEFAULT_REQUIRED_JOBS),
             owner: str | None = None) -> LineReport:
    """Run the repo's merge line once; see the module text. Another line holding the lock past
    `lock_wait_minutes`, or a stale lock, is `LockError` (and a `line` step ended `refused`); a run
    that breaks ends its step `failed` (`error`) and gives every held member its auto-merge back."""

    run = f"line:{repo}@{clock().isoformat()}"
    try:
        with lock.hold(repo, holder or holder_of(repo, base), settings=settings, root=root, clock=clock, sleep=sleep,
                       on_wait=on_wait):
            line = _Line(hub, store, repo, base, settings, clock, sleep, role, required,
                         (owner or repo.split("/")[0]).lower(), clock() + timedelta(minutes=settings.line_minutes), run)
            record.write(store, repo, None, role, run, "line",
                         [("step.started", {"base": base, "prs": len(line.members())})])
            try:
                turns, reason = line.run()
            except Exception:
                record.write(store, repo, None, role, run, "line",
                             [("step.ended", {"outcome": "failed", "reason": "error"})])
                raise
            finally:
                line.release_all()
    except lock.LockError as e:
        if e.code in ("lock_busy", "lock_stale"):
            record.write(store, repo, None, role, run, "line",
                         [("step.started", {"base": base, "prs": None}),
                          ("step.ended", {"outcome": "refused", "reason": e.code})])
        raise
    report = LineReport(repo, base, tuple(turns), reason)
    record.write(store, repo, None, role, run, "line",
                 [("step.ended", {"outcome": "passed", "reason": reason, "merged": report.merged,
                                  "left": report.left})])
    return report


def holder_of(repo: str, base: str) -> str:
    """How a line names itself in its lock."""

    return f"merge line of {repo} into {base}"
