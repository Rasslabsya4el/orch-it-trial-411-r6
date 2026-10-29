"""The workflow watch's sweep (https://github.com/Rasslabsya4el/orchestration-layer/issues/411 rule 3, unit 3):
`watch_workflows` reads from GitHub the latest ended run of every workflow GitHub lists and the open watch tickets,
decides with `workflow_watch.decide`, and carries out the actions: a watch ticket opened, a comment on one, or one
closed on a green run. `reconcile` calls it beside `check_main`. The workflows whose file is `WORKFLOW_FILE` are left
out: `check_main` owns them.

A watch ticket's second line is the run mark (`workflow_watch.run_mark`) of the run it opened for, and each comment
starts with the run mark of the run it reports: the next sweep reads the run a ticket last named
(`workflow_watch.last_run_id`, over the hub's `watch_issues`), so one run gives one action at most.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from orch_it.merge import workflow_watch as watch
from orch_it.merge.hub import PullHub
from orch_it.testruns.github import WORKFLOW_FILE
from orch_it.tickets import parse, violations


def checked_names(hub: PullHub, repo: str) -> frozenset[str]:
    """The file paths of the workflows whose file is `WORKFLOW_FILE`: the `tests` path owns them, not the watch."""

    return frozenset(path for path in hub.workflow_files(repo) if PurePosixPath(path).name == WORKFLOW_FILE)


def watch_workflows(hub: PullHub, repo: str, skip: frozenset[str]) -> list[watch.Action]:
    """Carry out the watch's actions for `repo`; returns the actions done, in order."""

    done: list[watch.Action] = []
    for action in watch.decide(hub.latest_workflow_runs(repo), hub.watch_issues(repo), skip=skip):
        if isinstance(action, watch.Open):
            body = _with_run_mark(action.body, action.run.run_id)
            _checked(action.title, body)
            hub.create_issue(repo, action.title, body, list(watch.LABELS))
        else:
            hub.comment(repo, action.issue_number, f"{watch.run_mark(action.run.run_id)}\n{action.body}")
            if isinstance(action, watch.Close):
                hub.close_issue(repo, action.issue_number, "completed")
        done.append(action)
    return done


def _with_run_mark(body: str, run_id: int) -> str:
    """The watch ticket's body with the run mark as its second line, after the workflow mark."""

    first, rest = body.split("\n", 1)
    return f"{first}\n{watch.run_mark(run_id)}\n{rest}"


def _checked(title: str, body: str) -> None:
    broken = violations(parse(body, title=title, labels=watch.LABELS))
    if broken:
        raise watch.WatchShapeError(", ".join(v.message for v in broken))
