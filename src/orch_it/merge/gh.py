"""`GhPullHub` (#400): the merge flow's `PullHub` over `gh`, for a real repo.

It is the tickets' `GhHub` (issues, comments, runs, all through `gh api`) plus these parts:

- pull requests: REST `repos/{r}/pulls/{n}` and `repos/{r}/pulls?state=open&base=...` (the list
  carries no `mergeable`, so each open pull request is read again on its own); `behind` from
  `repos/{r}/compare/{base}...{head_sha}` (`behind_by`, None when the compare is not there);
  create with `gh pr create`; auto-merge with `gh pr merge --auto` or `--disable-auto`;
  `PUT .../update-branch` and `PUT .../merge`, both pinned to the head sha the caller saw;
- check runs of one sha: `repos/{r}/commits/{sha}/check-runs`, the latest run of each name, its
  Actions run and attempt from `details_url` and `repos/{r}/actions/runs/{id}`;
- pull request comments: issue comments, submitted reviews and review comments, merged in time order;
- Actions: rerun the failed jobs, cancel, the runs of `tests.yml` on a branch with the state of
  the required jobs (setting `github_required_jobs`), the failed test ids from the `junit-*`
  artifacts of a run (downloaded with `gh run download`, read by `testruns.junit`);
- the conflicting files of a pull request: exact, by `git merge-tree --write-tree` in a bare
  cache repo under the data folder (`merge-cache/<owner>__<name>.git`) that fetches the base and
  `refs/pull/<n>/head` over https with `gh` as the credential helper; git runs through an
  injectable callable (`orch_it.merge.git.run` by default), as `gh` does;
- issues of red main: by label, create, close; and whether Actions is on for the repo;
- the events artifacts of the workflows (`orch-it-events-*`): the newest `EVENT_ARTIFACT_PAGES`
  pages of the repo's artifacts (a sweep before this one brought the older ones), each unpacked
  with `gh run download`.

The error rule is the `Hub`'s: a call that fails or answers badly raises
`orch_it.testruns.github.GithubError` (`gh_failed`, `gh_timeout`, `gh_bad_output`), a thing that
is not there is None (or empty). The HTTP statuses that mean "GitHub refused" and not "the call
broke" (a merge that cannot be done, a conflicting update, a run already completed) are told apart
by the `HTTP <status>` the failed `gh` call names.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

from orch_it import settings
from orch_it.events import schema
from orch_it.merge import git as merge_git
from orch_it.merge.git import GitDone, GitError
from orch_it.merge.hub import FAILED, NO_RESULT, PASSED, PENDING, Check, MainRun, Pull
from orch_it.merge.workflow_watch import WatchIssue, WorkflowRun, last_run_id, skipped, workflow_of
from orch_it.testruns import junit
from orch_it.testruns.github import (
    REPORT_FILE,
    WORKFLOW_FILE,
    Gh,
    GithubError,
    _required_jobs,
    json_values,
)
from orch_it.tickets.hub import Comment, GhHub, _moment, _text

CACHE_FOLDER = "merge-cache"
RUN_OUTCOMES = ("success", "failure", "cancelled", "timed_out", "skipped")  # a run's conclusion the watch keeps
WORKFLOW_TOKEN_REFUSAL = "Resource not accessible by integration"  # GitHub's 403 text for a workflow token
RUNS_SCANNED = 10  # the newest completed runs of a workflow the watch reads for its latest run not skipped
EVENTS_ARTIFACT_PREFIX = "orch-it-events-"
EVENT_ARTIFACT_PAGES = 3  # of 100 artifacts each, newest first
BASE_REF, PULL_REF = "refs/orch/base", "refs/orch/pr"
BRANCH_RUN_EVENTS = ("push", "schedule", "workflow_dispatch")
METHODS = ("merge", "squash", "rebase")
_RUN_OF_JOB = re.compile(r"/actions/runs/(\d+)/job/")
_PULL_URL = re.compile(r"/pull/(\d+)/?$")


class _Workflow(NamedTuple):
    """A workflow GitHub lists: its id, and its file path (its identity)."""

    id: int
    path: str
_JUNIT_NAME = re.compile(r"junit-(.+)-(\d+)")
_CREDENTIAL = ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]


Git = Callable[[list[str]], GitDone]


class GhPullHub(GhHub):
    def __init__(self, gh: Gh | None = None, git: Git | None = None):
        super().__init__(gh)
        self.git = git or merge_git.run

    # --- pull requests ----------------------------------------------------------------------

    def pull(self, repo: str, number: int) -> Pull | None:
        found = self._object(["api", f"repos/{repo}/pulls/{number}"], missing_ok=True)
        if found is None:
            return None
        return self._pull(repo, found)

    def open_pulls(self, repo: str, base: str) -> list[Pull]:
        listed = self._list(["api", "--paginate", f"repos/{repo}/pulls?state=open&base={quote(base, safe='')}"
                             "&per_page=100"])
        pulls = []
        for item in listed:
            if type(item.get("number")) is not int:
                raise GithubError("gh_bad_output", f"{repo}: a pull request with no number")
            found = self.pull(repo, item["number"])  # the list holds no `mergeable`
            if found is not None and found.state == "open":  # it may have merged since the list
                pulls.append(found)
        return sorted(pulls, key=lambda p: (p.created_at, p.number))

    def _pull(self, repo: str, found: dict) -> Pull:
        number = found.get("number")
        head, base = found.get("head"), found.get("base")
        created = _moment(found.get("created_at"))
        if (type(number) is not int or not isinstance(head, dict) or not isinstance(base, dict)
                or not _text(head.get("sha")) or not _text(base.get("ref")) or created is None):
            raise GithubError("gh_bad_output", f"{repo}: not a pull request")
        merged_at = _moment(found.get("merged_at"))
        state = "merged" if merged_at else "closed" if found.get("state") == "closed" else "open"
        behind = None
        if state == "open":
            behind = self._behind(repo, _text(base.get("ref")), head["sha"])
        return Pull(
            repo=repo, number=number, title=_text(found.get("title")), body=_text(found.get("body")),
            branch=_text(head.get("ref")), head_sha=head["sha"], base=base["ref"], state=state,
            draft=found.get("draft") is True, auto_merge=_auto_merge(repo, found.get("auto_merge")),
            mergeable={True: "mergeable", False: "conflicting"}.get(found.get("mergeable"), "unknown"),
            behind=behind, created_at=created, merged_at=merged_at,
            merge_sha=_text(found.get("merge_commit_sha")) or None if merged_at else None,
            closed_at=_moment(found.get("closed_at")),
            head_repo=_text((head.get("repo") or {}).get("full_name")) or None)

    def _behind(self, repo: str, base: str, head_sha: str) -> int | None:
        """The base's commits the head lacks: the compare reads `base...head`. The base is
        named by its branch, so its tip now counts, not the sha the pull request opened on."""

        found = self._object(["api", f"repos/{repo}/compare/{quote(base, safe='')}...{head_sha}?per_page=1"],
                             missing_ok=True)
        if found is None:
            return None
        behind = found.get("behind_by")
        if type(behind) is not int or behind < 0:
            raise GithubError("gh_bad_output", f"{repo}: compare without behind_by")
        return behind

    def create_pull(self, repo: str, base: str, branch: str, title: str, body: str) -> int:
        folder = tempfile.mkdtemp(prefix="orch-it-gh-")
        path = Path(folder) / "body.md"
        try:
            path.write_text(body, "utf-8")
            printed = self.gh(["pr", "create", "--repo", repo, "--base", base, "--head", branch,
                               "--title", title, "--body-file", str(path)])
        finally:
            path.unlink(missing_ok=True)
            os.rmdir(folder)
        lines = [line.strip() for line in printed.splitlines() if line.strip()]
        match = _PULL_URL.search(lines[-1]) if lines else None
        if match is None:
            raise GithubError("gh_bad_output", f"pr create: no pull request url in {printed[:80]!r}")
        return int(match.group(1))

    def close_pull(self, repo: str, number: int) -> None:
        self._send(["api", "-X", "PATCH", f"repos/{repo}/pulls/{number}"], {"state": "closed"})

    def set_auto_merge(self, repo: str, number: int, method: str | None) -> None:
        if method is None:
            args, already = ["--disable-auto"], ("not enabled", "not on", "already disabled", "already off")
        elif method in METHODS:
            args, already = ["--auto", f"--{method}"], ("already enabled", "already on")
        else:
            raise ValueError(f"not a merge method: {method}")
        try:
            self.gh(["pr", "merge", str(number), "--repo", repo, *args])
        except GithubError as e:
            if not any(phrase in str(e).lower() for phrase in already):
                raise

    def update_branch(self, repo: str, number: int, head_sha: str) -> bool:
        try:
            self._send(["api", "-X", "PUT", f"repos/{repo}/pulls/{number}/update-branch"],
                       {"expected_head_sha": head_sha})
        except GithubError as e:
            if "HTTP 422" in str(e) and "merge conflict" in str(e).lower():
                return False
            raise
        return True

    def merge(self, repo: str, number: int, head_sha: str, method: str) -> str | None:
        try:
            answer = self._answer(["api", "-X", "PUT", f"repos/{repo}/pulls/{number}/merge"],
                                  {"sha": head_sha, "merge_method": method})
        except GithubError as e:
            if any(f"HTTP {status}" in str(e) for status in (405, 409, 422)):
                return None
            raise
        if not _text(answer.get("sha")):
            raise GithubError("gh_bad_output", f"{repo}#{number}: a merge answered with no sha")
        return answer["sha"]

    # --- check runs and pull comments ---------------------------------------------------------------

    def checks(self, repo: str, sha: str) -> list[Check]:
        latest: dict[str, dict] = {}
        for run in self._pages(["api", "--paginate", f"repos/{repo}/commits/{sha}/check-runs?per_page=100"],
                               "check_runs"):
            if type(run.get("id")) is not int or not _text(run.get("name")):
                raise GithubError("gh_bad_output", f"{repo}@{sha}: a check run with no id or name")
            if run["name"] not in latest or run["id"] > latest[run["name"]]["id"]:
                latest[run["name"]] = run
        attempts: dict[str, int] = {}
        found = []
        for name, run in latest.items():
            match = _RUN_OF_JOB.search(_text(run.get("details_url")))
            run_id = match.group(1) if match else None
            if run_id is not None and run_id not in attempts:
                attempts[run_id] = self._attempt(repo, run_id)
            found.append(Check(name=name, state=_state(run), run_id=run_id,
                               attempt=attempts[run_id] if run_id else 1,
                               started_at=_moment(run.get("started_at"))))
        return found

    def _attempt(self, repo: str, run_id: str) -> int:
        run = self._object(["api", f"repos/{repo}/actions/runs/{run_id}"], missing_ok=True) or {}
        attempt = run.get("run_attempt")
        return attempt if type(attempt) is int and attempt > 0 else 1

    def pull_comments(self, repo: str, number: int) -> list[Comment]:
        said = self.comments(repo, number)
        for review in self._list(["api", "--paginate", f"repos/{repo}/pulls/{number}/reviews?per_page=100"]):
            if _text(review.get("body")) or review.get("state") == "CHANGES_REQUESTED":
                said += _note(review, "submitted_at")
        for item in self._list(["api", "--paginate", f"repos/{repo}/pulls/{number}/comments?per_page=100"]):
            said += _note(item, "created_at")
        return sorted(said, key=lambda c: c.at)

    # --- Actions ----------------------------------------------------------------------------

    def rerun(self, repo: str, run_id: str) -> None:
        self.gh(["api", "-X", "POST", f"repos/{repo}/actions/runs/{run_id}/rerun-failed-jobs"])

    def cancel(self, repo: str, run_id: str) -> None:
        try:
            self.gh(["api", "-X", "POST", f"repos/{repo}/actions/runs/{run_id}/cancel"])
        except GithubError as e:
            if "HTTP 409" not in str(e):  # 409: the run is already completed
                raise

    def failing_ids(self, repo: str, run_id: str) -> list[str] | None:
        try:
            artifacts = self._pages(
                ["api", "--paginate", f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"], "artifacts")
        except GithubError as e:
            if "HTTP 404" in str(e):
                return None
            raise
        names = _latest_reports([a["name"] for a in artifacts
                                 if _text(a.get("name")).startswith("junit-") and not a.get("expired")])
        failing, readable = set(), 0
        for name in names:
            with tempfile.TemporaryDirectory(prefix="orch-it-gh-") as folder:
                self.gh(["run", "download", str(run_id), "--repo", repo, "-n", name, "-D", folder])
                for report in sorted(Path(folder).rglob(REPORT_FILE)):
                    try:
                        cases = junit.read(report)
                    except junit.ReportUnreadable:
                        continue
                    readable += 1
                    failing |= {c.test_id for c in cases if c.outcome in ("failed", "error")}
        return sorted(failing) if readable else None

    def branch_runs(self, repo: str, branch: str) -> list[MainRun]:
        required = _required_jobs(settings.load())
        listed = self._pages(["api", f"repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
                              f"?branch={quote(branch, safe='')}&per_page=30"], "workflow_runs")
        runs = []
        for run in listed:
            if run.get("event") not in BRANCH_RUN_EVENTS:
                continue
            created = _moment(run.get("created_at"))
            if type(run.get("id")) is not int or not _text(run.get("head_sha")) or created is None:
                raise GithubError("gh_bad_output", f"{repo}: a workflow run with no id, sha or time")
            attempt = run.get("run_attempt")
            completed = run.get("status") == "completed"
            by_name: dict[str, dict] = {}
            for job in self._pages(
                    ["api", "--paginate", f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100"], "jobs"):
                name = _text(job.get("name")).strip()
                if name in required and (name not in by_name or _job_id(job) > _job_id(by_name[name])):
                    by_name[name] = job
            jobs = {name: _state(by_name[name]) if name in by_name else NO_RESULT if completed else PENDING
                    for name in required}
            runs.append(MainRun(run_id=str(run["id"]), sha=run["head_sha"], event=run["event"],
                                attempt=attempt if type(attempt) is int and attempt > 0 else 1,
                                created_at=created, jobs=jobs))
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    # --- conflicts --------------------------------------------------------------------------

    def conflict_files(self, repo: str, base: str, number: int) -> list[str]:
        if not schema.is_repo(repo):
            raise GithubError("bad_repo", str(repo))
        owner, name = repo.lower().split("/", 1)
        cache = settings.data_folder() / CACHE_FOLDER / f"{owner}__{name}.git"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            self._git(["init", "--bare", "--quiet", str(cache)])
        self._git([*_CREDENTIAL, "-C", str(cache), "fetch", "--quiet", "--no-tags", self.remote(repo),
                   f"+refs/heads/{base}:{BASE_REF}", f"+refs/pull/{number}/head:{PULL_REF}"])
        done = self._run_git(["-C", str(cache), "merge-tree", "--write-tree", "--name-only", "--no-messages", "-z",
                              BASE_REF, PULL_REF])
        if done.code == 0:
            return []
        if done.code != 1:
            raise GithubError("gh_failed", f"git merge-tree: exit {done.code}: {done.err}")
        # `-z`: the tree id, then each conflicted file, NUL-ended: a name with any character comes whole
        return sorted({name for name in done.out.split("\0")[1:] if name})

    def remote(self, repo: str) -> str:
        return f"https://github.com/{repo}.git"

    def _run_git(self, args: list[str]) -> GitDone:
        try:
            return self.git(args)
        except GitError as e:  # git that cannot start or gives no answer is a failed call of this hub
            raise GithubError("gh_failed", str(e)) from None

    def _git(self, args: list[str]) -> None:
        done = self._run_git(args)
        if done.code != 0:
            raise GithubError("gh_failed", f"git {' '.join(a for a in args if not a.startswith('-'))[:60]}: "
                              f"exit {done.code}: {done.err}")

    # --- issues -----------------------------------------------------------------------------

    def pulls_of_commit(self, repo: str, sha: str) -> list[int]:
        try:
            listed = self._list(["api", "--paginate", f"repos/{repo}/commits/{sha}/pulls?per_page=100"])
        except GithubError as e:
            if "HTTP 404" in str(e):
                return []
            raise
        return sorted({p["number"] for p in listed if p.get("merged_at") and type(p.get("number")) is int})

    def issues_with_label(self, repo: str, label: str) -> list[int]:
        listed = self._list(["api", "--paginate",
                             f"repos/{repo}/issues?labels={quote(label, safe='')}&state=open&per_page=100"])
        return sorted({i["number"] for i in listed if "pull_request" not in i and type(i.get("number")) is int})

    def create_issue(self, repo: str, title: str, body: str, labels: list[str]) -> int:
        answer = self._answer(["api", "-X", "POST", f"repos/{repo}/issues"],
                              {"title": title, "body": body, "labels": labels})
        if type(answer.get("number")) is not int:
            raise GithubError("gh_bad_output", f"{repo}: a new issue with no number")
        return answer["number"]

    def close_issue(self, repo: str, number: int, reason: str) -> None:
        self._send(["api", "-X", "PATCH", f"repos/{repo}/issues/{number}"],
                   {"state": "closed", "state_reason": reason})

    # --- workflow watch (#411) ----------------------------------------------------------------

    def workflow_files(self, repo: str) -> list[str]:
        return [workflow.path for workflow in self._workflows(repo)]

    def latest_workflow_runs(self, repo: str) -> list[WorkflowRun]:
        """The latest run of each workflow on the default branch that is not skipped by its `if:`: the newest
        `RUNS_SCANNED` completed runs are read, and the first one not skipped is the workflow's run."""

        branch = quote(self._default_branch(repo), safe="")
        found = []
        for workflow in self._workflows(repo):
            runs = self._pages(["api", f"repos/{repo}/actions/workflows/{workflow.id}/runs?status=completed"
                                f"&branch={branch}&per_page={RUNS_SCANNED}"], "workflow_runs")
            for run in runs:  # a workflow with no completed run on the default branch is left out
                built = self._workflow_run(repo, workflow.path, run)
                if not skipped(built):
                    found.append(built)
                    break
        return found

    def _workflow_run(self, repo: str, path: str, run: dict) -> WorkflowRun:
        run_id = run.get("id")
        if type(run_id) is not int or not _text(run.get("updated_at")):
            raise GithubError("gh_bad_output", f"{repo}: workflow `{path}` has a run with no id or end time")
        conclusion = run.get("conclusion")
        # a run GitHub already concluded `skipped` is passed over without reading its jobs
        jobs = [] if conclusion == "skipped" else self._pages(
            ["api", "--paginate", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"], "jobs")
        return WorkflowRun(
            workflow=path, run_id=run_id, url=_text(run.get("html_url")), ended_at=run["updated_at"],
            outcome=conclusion if conclusion in RUN_OUTCOMES else "other",
            jobs_started=sum(1 for job in jobs
                             if job.get("started_at") is not None and job.get("conclusion") != "skipped"),
            jobs_skipped=sum(1 for job in jobs if job.get("conclusion") == "skipped"))

    def watch_issues(self, repo: str) -> list[WatchIssue]:
        """The open issues (no pull requests) whose body holds a workflow watch mark, whatever labels they carry."""

        found = []
        listed = self._list(["api", "--paginate", f"repos/{repo}/issues?state=open&per_page=100"])
        for issue in listed:
            if "pull_request" in issue or type(issue.get("number")) is not int:
                continue
            body = _text(issue.get("body"))
            workflow = workflow_of(body)
            if workflow is None:
                continue
            texts = [body, *(c.body for c in self.comments(repo, issue.get("number")))]
            found.append(WatchIssue(issue["number"], workflow, last_run_id(texts)))
        return found

    def _default_branch(self, repo: str) -> str:
        found = self._object(["api", f"repos/{repo}"])
        branch = _text(found.get("default_branch")) if found else ""
        if not branch:
            raise GithubError("gh_bad_output", f"{repo}: a repo with no default branch")
        return branch

    def _workflows(self, repo: str) -> list[_Workflow]:
        """The id and file path of every workflow GitHub lists; a workflow is named by its file, never its name."""

        found = []
        for workflow in self._pages(["api", "--paginate", f"repos/{repo}/actions/workflows?per_page=100"],
                                    "workflows"):
            path = _text(workflow.get("path"))
            if type(workflow.get("id")) is not int or not path:
                raise GithubError("gh_bad_output", f"{repo}: a workflow with no id or file path")
            found.append(_Workflow(workflow["id"], path))
        return found

    def actions_enabled(self, repo: str) -> bool:
        """Whether Actions is on for the repo. The workflow's own token holds no administration scope, so GitHub
        refuses it this read with a 403 saying "Resource not accessible by integration": that one answer reads as
        on, because a sweep running on a workflow token is running on Actions. Any other failure, a 403 of another
        kind (a rate limit, a person's login without admin) included, raises."""

        try:
            found = self._object(["api", f"repos/{repo}/actions/permissions"], missing_ok=True)
        except GithubError as e:
            if "HTTP 403" in str(e) and WORKFLOW_TOKEN_REFUSAL in str(e):
                return True
            raise
        if found is None:
            return False
        if not isinstance(found.get("enabled"), bool):
            raise GithubError("gh_bad_output", f"{repo}: actions/permissions with no `enabled`")
        return found["enabled"]

    def event_artifacts(self, repo: str) -> list[tuple[str, str]]:
        found = []
        for page in range(1, EVENT_ARTIFACT_PAGES + 1):
            items = self._pages(["api", f"repos/{repo}/actions/artifacts?per_page=100&page={page}"], "artifacts")
            for item in items:
                run = item.get("workflow_run") if isinstance(item.get("workflow_run"), dict) else {}
                if (_text(item.get("name")).startswith(EVENTS_ARTIFACT_PREFIX) and not item.get("expired")
                        and type(run.get("id")) is int):
                    found.append((str(run["id"]), item["name"]))
            if len(items) < 100:
                break
        return found

    def download_artifact(self, repo: str, run_id: str, name: str, folder: Path) -> None:
        self.gh(["run", "download", str(run_id), "--repo", repo, "-n", name, "-D", str(folder)])

    # --- helpers ----------------------------------------------------------------------------

    def _pages(self, args: list[str], key: str) -> list[dict]:
        """The objects under `key` of every page of an answer that wraps its list in an object."""

        items = []
        pages = json_values(self._read(args))
        if not pages:
            raise GithubError("gh_bad_output", f"{' '.join(args[:3])}: no answer")
        for page in pages:
            found = page.get(key) if isinstance(page, dict) else None
            if not isinstance(found, list):
                raise GithubError("gh_bad_output", f"{' '.join(args[:3])}: no list `{key}`")
            items += [item for item in found if isinstance(item, dict)]
        return items

    def _answer(self, args: list[str], payload: dict) -> dict:
        """A write like `_send` whose answer (one JSON object) is read."""

        folder = tempfile.mkdtemp(prefix="orch-it-gh-")
        path = Path(folder) / "input.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
            printed = self.gh([*args, "--input", str(path)])
        finally:
            path.unlink(missing_ok=True)
            os.rmdir(folder)
        values = json_values(printed)
        if len(values) != 1 or not isinstance(values[0], dict):
            raise GithubError("gh_bad_output", f"{' '.join(args[:3])}: expected one object")
        return values[0]


def _auto_merge(repo: str, value: object) -> str | None:
    if value is None:
        return None
    method = value.get("merge_method") if isinstance(value, dict) else None
    if not isinstance(method, str) or method.lower() not in METHODS:
        raise GithubError("gh_bad_output", f"{repo}: an auto_merge with no merge_method")
    return method.lower()


def _state(run: dict) -> str:
    """A check run's or a job's state: not finished is pending, `success` passed, `failure` failed,
    anything else (cancelled, timed out, skipped, stale, neutral, ...) no result."""

    if run.get("status") != "completed":
        return PENDING
    return {"success": PASSED, "failure": FAILED}.get(run.get("conclusion"), NO_RESULT)


def _job_id(job: dict) -> int:
    return job["id"] if type(job.get("id")) is int else 0


def _note(item: dict, time_key: str) -> list[Comment]:
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    at = _moment(item.get(time_key))
    if at is None:
        return []
    return [Comment(_text(user.get("login")) or "ghost", user.get("type") == "Bot", _text(item.get("body")), at)]


def _latest_reports(names: list[str]) -> list[str]:
    """`junit-<job>-<attempt>` artifacts: only the latest attempt of each job (an earlier attempt
    of a job that was run again says nothing of it now); any other name is kept."""

    latest: dict[str, tuple[int, str]] = defaultdict(lambda: (-1, ""))
    for name in names:
        match = _JUNIT_NAME.fullmatch(name)
        job, attempt = (match.group(1), int(match.group(2))) if match else (name, 0)
        if attempt > latest[job][0]:
            latest[job] = (attempt, name)
    return [name for _, name in latest.values()]

