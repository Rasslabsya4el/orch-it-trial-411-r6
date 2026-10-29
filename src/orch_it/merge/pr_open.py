"""`pr open <ticket>` (#400 decision 1): ORCH's acceptance step ends in one OPEN pull request per
ticket at a time; a merged one never refuses the next (#434), and a part's pull request carries
`Part: k of n` beside `Refs #N`.

In this order, and nothing written before every check passed:

1. the ticket is open, has an acceptance table, and the results ORCH measured (a table `row`,
   `result`, `evidence`) name each of its rows once: a row marked `after merge` is `after merge`,
   every other row `pass`, or `na` with its evidence; any `fail` refuses (`row_failed`);
2. one OPEN pull request per ticket at a time: an open pull request of another branch that refers to
   the ticket refuses (`ticket_has_pr`), the branch's own open pull request is reused when it refers
   to this ticket and refused when it refers to another (`pr_other_ticket`); a ticket parked on
   anything but that pull request refuses (`parked_elsewhere`);
3. no commit of the branch past the base, and not the title, holds a closing keyword
   (`closing_keyword`, MERGE-10): the pull request says `Refs #N` and the ticket is closed by the
   follow, with its acceptance rows, never by GitHub;
4. the branch is pushed (never forced), the pull request opened or reused, the acceptance comment
   written on the ticket (the rows, the head they were checked at, the hash of the body), the ready
   mark on the pull request (it enters the merge line: a later left mark is undone), auto-merge
   turned on with the setting `merge_method`, and the ticket parked `pr_ended` on it through #398's
   park command until the day after `pr_max_hours` from now;
5. the `pr` step started (run `pr:<repo>#<n>@<ready mark time>`), one `acceptance_row.checked`
   per row that was run, and `comment.written` (`accepted`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from orch_it.events.store import Store
from orch_it.merge import marks, record
from orch_it.merge.git import GitError, branch_ok
from orch_it.merge.hub import Pull, PullHub
from orch_it.merge.settings import MergeSettings
from orch_it.tickets import park
from orch_it.tickets.model import AcceptanceRow, parse

ROLE = "orch"
PR_ERROR_CODES = (
    "ticket_missing", "ticket_not_open", "acceptance_missing", "acceptance_unreadable", "row_missing",
    "row_failed", "row_unknown", "row_twice", "row_bad", "branch_is_base", "branch_bad", "ticket_has_pr",
    "pr_other_ticket", "parked_elsewhere", "closing_keyword", "git_failed", "push_failed", "head_moved",
    "bad_part",
)
_REFS = re.compile(r"(?m)^Refs #([1-9][0-9]{0,9})\b")
_PART_ARG = re.compile(r"([0-9]{1,9})/([0-9]{1,9})")

EVIDENCE_CHARS = 200  # an event's evidence; the comment keeps the whole of it
Git = Callable[[list[str]], str]  # `git <args>` in the work tree; raises `orch_it.merge.git.GitError`


class PrError(ValueError):
    """A pull request `pr open` will not open; `code` is one of `PR_ERROR_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in PR_ERROR_CODES:
            raise ValueError(f"not a pr open error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class Opened:
    pr: int
    head: str
    reused: bool
    park: str  # the park line the ticket holds


def refers_to(body: str) -> int | None:
    """The ticket a pull request's body refers to: its line `Refs #N`."""

    found = _REFS.search(body or "")
    return int(found.group(1)) if found else None


def parse_part(text: str) -> tuple[int, int]:
    """The `--part K/N` argument (#434): 1 <= K <= N, else `PrError` `bad_part`."""

    found = _PART_ARG.fullmatch(text or "")
    if found is None or not 1 <= int(found.group(1)) <= int(found.group(2)):
        raise PrError("bad_part", f"{(text or '')[:20]!r} is not K/N with 1 <= K <= N")
    return int(found.group(1)), int(found.group(2))


def open_pr(hub: PullHub, store: Store, repo: str, number: int, *, acceptance: str, branch: str, base: str,
            git: Git, settings: MergeSettings, now: datetime, role: str = ROLE, notes: str = "",
            part: tuple[int, int] | None = None) -> Opened:
    """`notes`: ORCH's report lines, put in the body of a pull request it opens, under its rows.
    `part`: the (k, n) of a part's pull request (#434), written as `Part: k of n` in a new body; a
    reused pull request keeps its body."""
    if part is not None and not 1 <= part[0] <= part[1]:
        raise PrError("bad_part", f"{part[0]}/{part[1]} is not K/N with 1 <= K <= N")
    for name in (branch, base):
        if not branch_ok(name):
            raise PrError("branch_bad", f"{name[:60]!r} is no branch name git would read as one")
    if branch == base:
        raise PrError("branch_is_base", f"{branch} is the base itself")
    issue = hub.issue(repo, number)
    if issue is None or issue.is_pr:
        raise PrError("ticket_missing", f"{repo}#{number} is no ticket")
    if not issue.open:
        raise PrError("ticket_not_open", f"{repo}#{number}")
    ticket = parse(issue.body, title=issue.title, labels=issue.labels)
    if not ticket.acceptance:
        raise PrError("acceptance_missing", f"{repo}#{number} has no acceptance table")
    rows = checked_rows(ticket.acceptance, acceptance)

    # a fork's branch of the same name is never the ticket's: only this repo's heads count
    pulls = [p for p in hub.open_pulls(repo, base) if (p.head_repo or "").lower() == repo.lower()]
    mine = next((p for p in pulls if p.branch == branch), None)
    if mine is not None and refers_to(mine.body) != number:
        said = f"refers to #{refers_to(mine.body)}" if refers_to(mine.body) else "has no line `Refs #N`"
        raise PrError("pr_other_ticket", f"#{mine.number} of {branch} {said}")
    other = next((p for p in pulls if p.branch != branch and refers_to(p.body) == number), None)
    if other is not None:
        raise PrError("ticket_has_pr", f"#{other.number} ({other.branch}) is the pull request of #{number}")
    parked_on_it = _parked_on(ticket.park_lines, repo, mine)
    if (ticket.parked or ticket.park_lines) and not parked_on_it:
        raise PrError("parked_elsewhere", " / ".join(ticket.park_lines)[:120] or "the label `parked`")

    try:
        git(["fetch", "origin", base])
        messages = git(["log", "--format=%B%x00", f"origin/{base}..{branch}", "--"])
        measured = git(["rev-parse", "--verify", "--end-of-options", f"{branch}^{{commit}}"]).strip()
    except GitError as e:
        raise PrError("git_failed", str(e)) from None
    texts = [messages, issue.title, notes, *((mine.title, mine.body) if mine else ())]
    closing = [found for text in texts for found in marks.closing_keywords(text)]
    if closing:
        raise PrError("closing_keyword", f"{', '.join(dict.fromkeys(closing))}: write `Refs #{number}`")
    try:
        git(["push", "origin", f"{branch}:refs/heads/{branch}"])
    except GitError as e:
        raise PrError("push_failed", str(e)) from None

    reused = mine is not None
    pr = mine.number if reused else hub.create_pull(repo, base, branch, issue.title,
                                                    _pr_body(number, issue.title, rows, notes, part))
    pull = hub.pull(repo, pr)
    if pull is None or pull.head_sha != measured:
        # the rows were measured on the work tree's head; another one would land unmeasured
        raise PrError("head_moved", f"#{pr} is at {pull.head_sha[:12] if pull else 'nothing'}, the rows were "
                      f"measured at {measured[:12]}")
    hub.comment(repo, number, marks.acceptance_comment(pr, measured, marks.body_hash(issue.body), rows))
    after_open(hub, repo, pr, measured, number, now, settings.merge_method)
    if parked_on_it:
        line = ticket.park_lines[0]
    else:
        until = (now + timedelta(hours=settings.pr_max_hours)).date() + timedelta(days=1)
        try:
            line = park.park(hub, store, repo, number, "pr_ended", f"#{pr}", until, today=now.date(), role=role).line
        except park.ParkError as e:
            if reused or e.code != "already_parked":
                raise
            # another `pr open` of this ticket parked it on its own pull request a moment before:
            # this one, just opened, never enters the line
            hub.set_auto_merge(repo, pr, None)
            hub.comment(repo, pr, marks.left_comment("ticket_has_pr", f"#{number} took another pull request first"))
            hub.close_pull(repo, pr)
            raise PrError("ticket_has_pr", f"#{number} was parked by another pull request first: {e}") from None
    ready = marks.ready_since(marks.of_layer(hub.pull_comments(repo, pr), repo.split("/")[0]))
    events = [("step.started", {"pr": pr, "branch": branch, "base": base, "head": measured}),
              ("comment.written", {"comment_kind": "accepted"})]
    events += [("acceptance_row.checked", {"row": row.id, "result": row.result, "place": "branch",
                                           "evidence": row.evidence[:EVIDENCE_CHARS]})
               for row in rows if row.result != "after merge"]
    record.write(store, repo, number, role, record.pr_run(repo, pr, ready.at if ready else None), "pr", events)
    return Opened(pr, measured, reused, line)


def after_open(hub: PullHub, repo: str, pr: int, head: str, ticket: int | None, now: datetime,
               method: str) -> None:
    """Whoever opens a pull request (`pr open`, `skills update`) puts it in the merge line: the ready mark at
    `head`, then auto-merge with `method`. `ticket` is None for a pull request that names no ticket."""

    hub.comment(repo, pr, marks.ready_comment(ticket, head, now))
    hub.set_auto_merge(repo, pr, method)


def checked_rows(rows: tuple[AcceptanceRow, ...], text: str) -> list[marks.ResultRow]:
    """The measured results, one per ticket row in the ticket's order, or `PrError`."""

    results = marks.read_results(text)
    if not results:
        raise PrError("acceptance_unreadable", "no table with the columns `row` and `result`")
    wanted = {row.id: row for row in rows}
    by_id: dict[str, marks.ResultRow] = {}
    for result in results:
        if result.id not in wanted:
            raise PrError("row_unknown", f"row {result.id[:20]} is not in the ticket")
        if result.id in by_id:
            raise PrError("row_twice", f"row {result.id}")
        by_id[result.id] = result
    # a row checked only after the merge is not run before it: left out, it is `after merge`
    for row in rows:
        if row.after_merge and row.id not in by_id:
            by_id[row.id] = marks.ResultRow(row.id, "after merge", "")
    missing = [row.id for row in rows if row.id not in by_id]
    if missing:
        raise PrError("row_missing", f"row {', '.join(missing)}")
    for row in rows:
        result = by_id[row.id]
        if result.result == "fail":
            raise PrError("row_failed", f"row {row.id}: {result.evidence[:80]}")
        if row.after_merge != (result.result == "after merge"):
            said = "is checked after the merge" if row.after_merge else "is checked before the pull request"
            raise PrError("row_bad", f"row {row.id} {said}, not {result.result!r}")
        if result.result not in marks.RESULTS:
            raise PrError("row_bad", f"row {row.id}: {result.result[:20]!r} is none of {', '.join(marks.RESULTS)}")
        if result.result == "na" and not result.evidence.strip():
            raise PrError("row_bad", f"row {row.id}: `na` needs its reason in `evidence`")
    return [by_id[row.id] for row in rows]


def _parked_on(lines: tuple[str, ...], repo: str, mine: Pull | None) -> bool:
    held = park.held_park(lines)
    ref = held.ticket if held is not None and held.kind == "pr_ended" else None
    return mine is not None and ref is not None and ref.names(repo, mine.number, repo)


def _pr_body(number: int, title: str, rows: list[marks.ResultRow], notes: str,
             part: tuple[int, int] | None = None) -> str:
    table = "".join(f"| {row.id} | {row.result} |\n" for row in rows)
    tail = f"\n{notes.strip()}\n" if notes.strip() else ""
    header = f"Refs #{number}\n" + (f"Part: {part[0]} of {part[1]}\n" if part else "")
    return (f"{header}\n{title}\n\nThe acceptance rows, checked before this pull request "
            f"(the ticket's acceptance comment holds their evidence):\n\n| row | result |\n|---|---|\n{table}{tail}")
