"""Main is checked after every merge (#400 decision 4): `check_main` reads the runs of the `tests`
workflow on a branch (a push, so after every merge, and the nightly run) and keeps the red-on-main
tickets true to the newest run that ended (`branch_runs` gives the required jobs only):

- green on every required job: every open red-on-main ticket closes, one whose ids block a person
  broke too (`green`);
- red: the failing ids from the run's reports, as `red_main.normal` reads them. When every job
  ended passed or failed, an open red-on-main ticket none of whose tests failed closes (they pass
  now); the ids no open ticket covers get one new ticket, ready for an agent, `priority`, in #398's
  shape, naming the pull request that brought the run's commit; none is opened for ids already
  covered (`red`). A red run that names no test is no result;
- no result (a job cut, cancelled, skipped, or red with no report): the run runs once more
  (`rerun`; a rerun GitHub refuses is one already asked for); no result again on its second
  attempt opens one ticket to the brain for that run (`no_result`, in `brain`, not `opened`).

When the newest run has not ended, an older ended run may open a red ticket but closes none: the
newer run may be red. No skip list exists: red on main is its own ticket (MERGE-7). The workflow
`main-check` calls it when a `tests` run on the default branch ends, `reconcile` on its schedule.
Events: `main.red` with `ticket.opened` per new red ticket, `ticket.closed` per closed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orch_it.events.store import Store
from orch_it.merge import marks, record, red_main
from orch_it.merge.follow import ROLE
from orch_it.merge.hub import FAILED, NO_RESULT, PASSED, PENDING, MainRun, PullHub
from orch_it.testruns.github import GithubError
from orch_it.tickets import parse
from orch_it.tickets.reader import TRIAGE_LABEL


@dataclass(frozen=True)
class MainReport:
    outcome: str  # green, red, rerun, no_result, waiting, no_run
    run_id: str | None = None
    sha: str | None = None
    opened: list[int] = field(default_factory=list)  # red-on-main tickets opened
    closed: list[int] = field(default_factory=list)  # red-on-main tickets closed
    brain: list[int] = field(default_factory=list)  # the ticket to the brain of a run with no result twice


def check_main(hub: PullHub, store: Store, repo: str, branch: str, *, role: str = ROLE) -> MainReport:
    """Bring the red-on-main tickets in line with the newest ended `tests` run on `branch`."""

    runs = hub.branch_runs(repo, branch)
    ended = [run for run in runs if run.jobs and PENDING not in run.jobs.values()]
    if not ended:
        return MainReport("waiting" if runs else "no_run")
    run, newest = ended[0], ended[0] is runs[0]
    states = set(run.jobs.values())
    reds = red_main.open_reds(hub, repo)
    if states == {PASSED}:
        if not newest:
            return MainReport("waiting", run.run_id, run.sha)
        return MainReport("green", run.run_id, run.sha,
                          closed=[_close(hub, store, repo, red.number, run, role) for red in reds])
    ids = hub.failing_ids(repo, run.run_id) if FAILED in states else None
    failing = {red_main.normal(test_id) for test_id in ids or ()} - {""}
    if failing:
        return _red(hub, store, repo, run, failing, reds, newest and NO_RESULT not in states, role)
    if run.attempt >= 2:
        return MainReport("no_result", run.run_id, run.sha, brain=_to_the_brain(hub, store, repo, run, role))
    if newest:
        try:
            hub.rerun(repo, run.run_id)
        except GithubError:
            pass  # another check asked for it a moment ago: the run runs again either way
    return MainReport("rerun", run.run_id, run.sha)


def _red(hub: PullHub, store: Store, repo: str, run: MainRun, failing: set[str], reds: list[red_main.RedTicket],
         may_close: bool, role: str) -> MainReport:
    closed = [_close(hub, store, repo, red.number, run, role) for red in reds
              if may_close and red.ids is not None and not red.truncated and not red.ids & failing]
    standing = [red for red in reds if red.number not in closed and red.ids is not None]
    if any(red.truncated for red in standing):
        return MainReport("red", run.run_id, run.sha, closed=closed)  # one ticket of a huge failure covers all
    new = sorted(failing - set().union(*(red.ids for red in standing)))
    if not new:
        return MainReport("red", run.run_id, run.sha, closed=closed)
    pulls = hub.pulls_of_commit(repo, run.sha)
    culprit = pulls[0] if pulls else None
    title, body = red_main.build(new, sha=run.sha, run_id=run.run_id, pr=culprit)
    number = hub.create_issue(repo, title, body, list(red_main.LABELS))
    # a second check racing this one may have opened the same ticket a moment before: the older one stays
    mine = red_main.ids_of(body)
    twins = [red.number for red in red_main.open_reds(hub, repo) if red.number != number and red.ids == mine]
    if twins and min(twins) < number:
        hub.comment(repo, number, f"{marks.follow_mark('duplicate')}The same failing tests are #{min(twins)}'s.")
        hub.close_issue(repo, number, "not_planned")
        return MainReport("red", run.run_id, run.sha, closed=closed)
    record.write(store, repo, number, role, None, None, [
        ("main.red", {"sha": run.sha, "test_ids": new[:red_main.IDS_IN_EVENT], "introduced_by_pr": culprit}),
        _opened(title, body, red_main.LABELS, priority=True)])
    return MainReport("red", run.run_id, run.sha, opened=[number], closed=closed)


def _opened(title: str, body: str, labels, *, priority: bool) -> tuple[str, dict]:
    ticket = parse(body, title=title, labels=labels)
    return ("ticket.opened", {"title": title, "labels": list(labels), "parent": None, "weight": ticket.weight,
                              "priority": priority, "write_ticket_called": False,
                              "has_outside_case": ticket.outside_case is not None,
                              "acceptance_rows": len(ticket.acceptance or ())})


def _close(hub: PullHub, store: Store, repo: str, number: int, run: MainRun, role: str) -> int:
    hub.comment(repo, number, f"{marks.follow_mark('closed')}Main is green on these tests in the `tests` run "
                              f"{run.run_id} at `{run.sha[:12]}`: closed.")
    hub.close_issue(repo, number, "completed")
    record.write(store, repo, number, role, None, None, [("comment.written", {"comment_kind": "closing"}),
                                                         record.ticket_closed(None, None)])
    return number


def _to_the_brain(hub: PullHub, store: Store, repo: str, run: MainRun, role: str) -> list[int]:
    mark = red_main.NO_RESULT_MARK.format(run_id=run.run_id)
    for number in hub.issues_with_label(repo, TRIAGE_LABEL):
        issue = hub.issue(repo, number)
        if issue is not None and mark in issue.body:
            return []  # one ticket per run
    title, body = red_main.build_no_result(sha=run.sha, run_id=run.run_id, jobs=run.jobs)
    number = hub.create_issue(repo, title, body, list(red_main.NO_RESULT_LABELS))
    record.write(store, repo, number, role, None, None, [_opened(title, body, red_main.NO_RESULT_LABELS,
                                                                 priority=False)])
    return [number]
