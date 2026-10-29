"""The workflow watch's decision (https://github.com/Rasslabsya4el/orchestration-layer/issues/411 rule 3, unit 2):
what a non-green workflow run opens, comments on or closes. `decide` reads the latest ended run of each workflow and
the open watch issues, and returns actions; it does no I/O. A run is non-green when it ended `failure`, `cancelled`
or `timed_out`, or when it started no job at all (a run that failed at start), unless GitHub skipped it by its `if:`.
A run GitHub skipped is neither green nor non-green: it is passed over, and the watch reads the latest run that is
not skipped. A run is green only when it ended `success` with a job started; any other run gets no action. The
caller passes the workflows it leaves to other checks in `skip`.

A workflow is identified by its file path (`.github/workflows/x.yml`), never by its display name: two files can
share a name. A watch ticket is written in #398's shape and checked by `tickets.violations` before it is returned, as
`red_main.build` does: a broken body is a bug of this module, so it raises. The title and the human block carry
`display_name` (a short, shape-clean name); the file path rides in the mark line, which `workflow_of` reads back.
A watch ticket's second line is the run mark of the run it names, which `last_run_id` reads back.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import quote, unquote

from orch_it.tickets import parse, violations
from orch_it.tickets.reader import TRIAGE_LABEL

LABELS = (TRIAGE_LABEL,)
NON_GREEN = frozenset({"failure", "cancelled", "timed_out"})
HOW = {"failure": "failure", "cancelled": "cancelled", "timed_out": "timed out"}
NO_JOB = "no job started"
NAME_MAX = 49  # `Workflow ` + name + ` did not end green` stays within the title limit of 80
_MARK = re.compile(r"<!-- orch-it workflow-watch workflow=(\S+) -->")
RUN_MARK = "<!-- orch-it workflow-watch run={run_id} -->"
_RUN = re.compile(r"<!-- orch-it workflow-watch run=(\d+) -->")
_PARKED = re.compile(r"parked", re.IGNORECASE)
_NOT_NAME = re.compile(r"[^\w\- ]+")


class WatchShapeError(AssertionError):
    """A watch ticket this module wrote breaks a shape rule: a bug of this module, never of the input."""


@dataclass(frozen=True)
class WorkflowRun:
    workflow: str  # the workflow's file path, as GitHub gives it
    run_id: int
    url: str
    ended_at: str
    outcome: str  # success, failure, cancelled, timed_out, skipped or other
    jobs_started: int
    jobs_skipped: int = 0  # jobs GitHub skipped by their `if:`; they neither start nor fail


@dataclass(frozen=True)
class WatchIssue:
    number: int
    workflow: str
    last_run_id: int | None = None  # the run the issue last named; a repeat of it is no action


@dataclass(frozen=True)
class Open:
    workflow: str
    run: WorkflowRun
    title: str
    body: str


@dataclass(frozen=True)
class Comment:
    issue_number: int
    run: WorkflowRun
    body: str


@dataclass(frozen=True)
class Close:
    issue_number: int
    run: WorkflowRun
    body: str


Action = Open | Comment | Close


def skipped(run: WorkflowRun) -> bool:
    """A run GitHub skipped by its `if:`: its conclusion is `skipped`, or no job of it started and every job of
    it was skipped with no non-green conclusion. Neither green nor non-green; the watch looks past it."""

    return run.outcome == "skipped" or (run.outcome not in NON_GREEN and run.jobs_started == 0 and run.jobs_skipped > 0)


def non_green(run: WorkflowRun) -> bool:
    return not skipped(run) and (run.jobs_started == 0 or run.outcome in NON_GREEN)


def green(run: WorkflowRun) -> bool:
    return run.outcome == "success" and run.jobs_started > 0


def how_ended(run: WorkflowRun) -> str:
    """How a non-green run ended, in the words the watch ticket names."""

    return NO_JOB if run.jobs_started == 0 else HOW[run.outcome]


def display_name(workflow: str) -> str:
    """The workflow's name for a title and a human block: a path gives its file name without a `.yml` or
    `.yaml` extension, the word "parked" becomes "held", every mark but letters, digits, dashes and spaces
    becomes a space, and the result is cut to `NAME_MAX` characters."""

    name = workflow.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith((".yml", ".yaml")):
        name = name.rsplit(".", 1)[0]
    name = _PARKED.sub("held", name)
    name = " ".join(_NOT_NAME.sub(" ", name).split())[:NAME_MAX].strip()
    return name or "unnamed"


def mark(workflow: str) -> str:
    """The mark line of a watch ticket: the workflow's file path, percent-encoded so it can hold any
    character and the comment never ends early."""

    return f"<!-- orch-it workflow-watch workflow={quote(workflow, safe='')} -->"


def workflow_of(body: str | None) -> str | None:
    """The workflow file path a watch ticket's mark holds; None for a body that is no watch ticket."""

    found = _MARK.search(body or "")
    return unquote(found.group(1)) if found else None


def run_mark(run_id: int) -> str:
    """The run mark of a run: the second line of a watch ticket it opened, the first line of a comment on it."""

    return RUN_MARK.format(run_id=run_id)


def last_run_id(texts: Iterable[str | None]) -> int | None:
    """The highest run a watch ticket's body or comments name; None when none names one."""

    found = [int(number) for text in texts for number in _RUN.findall(text or "")]
    return max(found) if found else None


def _checked(title: str, body: str) -> tuple[str, str]:
    broken = violations(parse(body, title=title, labels=LABELS))
    if broken:
        raise WatchShapeError(", ".join(v.message for v in broken))
    return title, body


def build(run: WorkflowRun) -> tuple[str, str]:
    """(title, body) of the watch ticket for the non-green latest `run` of its workflow."""

    name, shown, how = run.workflow, display_name(run.workflow), how_ended(run)
    title = f"Workflow {shown} did not end green"
    body = f"""{mark(name)}
## For the human
- **What:** workflow `{shown}` не завершился зелёным: последний прогон {run.run_id} закончился без зелёного результата.
- **Goal:** последний прогон `{shown}` снова зелёный, и тикет закрывается сам на первом таком прогоне.
- **Plan:** разобрать, почему прогон не зелёный, и починить причину; что именно чинить, решает этот тикет.
- **Result:** последний прогон `{shown}` зелёный.
- **Undo:** two-way door (один откат); touches только запуск `{shown}`.

## For the agent
Facts: the workflow `{name}` ended its latest run {run.run_id} at {run.url} as {how}; it ended {run.ended_at}.
Rule: make the latest run of `{shown}` green again by a fix or a revert; the watch closes on its first green run.

| row | check | pass |
|---|---|---|
| 1 | the next run of `{shown}` | ends with success and at least one job started |

Outside case: a run that GitHub labels success but that started no job (row 1).
Same object: the watch comments on this issue at each further non-green run of `{shown}` and closes it on a green one.
Weight: 2
"""
    return _checked(title, body)


def _comment(run: WorkflowRun) -> str:
    return (f"Workflow `{run.workflow}` did not end green again: run {run.run_id} ended as {how_ended(run)} "
            f"at {run.url}, ended {run.ended_at}.")


def _close(run: WorkflowRun) -> str:
    return (f"Workflow `{run.workflow}` ends green: run {run.run_id} at {run.url}, ended {run.ended_at}. "
            "Closing this watch.")


def latest_per_workflow(runs: list[WorkflowRun]) -> list[WorkflowRun]:
    """One run per workflow: the one that ended last (greater `ended_at`, then greater `run_id`)."""

    chosen: dict[str, WorkflowRun] = {}
    for run in runs:
        held = chosen.get(run.workflow)
        if held is None or (run.ended_at, run.run_id) > (held.ended_at, held.run_id):
            chosen[run.workflow] = run
    return list(chosen.values())


def decide(latest: list[WorkflowRun], open_issues: list[WatchIssue], *, skip: frozenset[str]) -> list[Action]:
    """The actions for the latest ended run of each workflow that is not skipped: a non-green one opens its watch or
    comments on the lowest-numbered open one (no comment for a run it already names); a green one closes every
    open watch of its workflow; any other run, and a workflow in `skip`, gets nothing."""

    actions: list[Action] = []
    for run in latest_per_workflow([run for run in latest if run.workflow not in skip and not skipped(run)]):
        watches = sorted((issue for issue in open_issues if issue.workflow == run.workflow),
                         key=lambda issue: issue.number)
        if green(run):
            actions += [Close(issue.number, run, _close(run)) for issue in watches]
        elif not non_green(run):
            continue
        elif not watches:
            title, body = build(run)
            actions.append(Open(run.workflow, run, title, body))
        elif watches[0].last_run_id != run.run_id:
            actions.append(Comment(watches[0].number, run, _comment(run)))
    return actions
