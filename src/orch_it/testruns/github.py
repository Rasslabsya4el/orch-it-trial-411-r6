"""The GitHub importer (#394 decision 7): the runs of the `tests` workflow become events.

For each completed run, and for each job named in the setting `github_required_jobs` (default
`linux` and `windows`) in every attempt of it, the importer writes a `test_run` step pair
(place `github`), the queue time as a `wait` pair with reason `github_queue`, and one
`test.result` per case of the job's JUnit artifact. The artifact is read by `outcome.judge`
like any other report; the job's GitHub conclusion decides the outcome:

- `success`: passed, always (the report only gives the rows and the counts);
- `failure`: failed `red` when the report holds a red case, else no_result `summary_missing`;
- `cancelled`: no_result `cut` when the job ran for at least the setting
  `github_job_timeout_minutes` (default 15; GitHub may call a job that hit the workflow's
  `timeout-minutes` cancelled), else stopped `superseded` when a newer run superseded its run,
  else stopped with no reason. A newer run supersedes a run when it was created after it (and
  before `now`), has the same event and another head commit, and, for pull request runs, the
  same pull request number (else the same head branch);
- `timed_out`: no_result `cut`;
- `skipped`: no test run at all, a skipped job is not a run;
- anything else (`neutral`, `stale`, `startup_failure`, `action_required`): no_result
  `summary_missing`.

A completed run with no job of a required name in any attempt up to its latest gets, for that
name, a `job_missing` pair, or, when the run was cancelled, a `stopped` pair with the run's
reason. A run still in progress is left for the next import. In the `newest` mode (one unpaginated
list of the newest completed runs) only those listed runs can supersede a cancelled one, so a
cancelled run whose superseder is still running gets reason None.

Every event's id is a hash of (run id, job id or job name, attempt, kind), so a second import
writes nothing. A run whose latest attempt already has an end stored for every required job is
skipped before any `gh` call, and a job whose `step.ended` is stored is skipped before anything
is downloaded. A run's events are written only once everything the run needs from `gh` has been
read, and its `test.result` rows before its `step.ended`, so a run `gh` failed on is skipped
whole (counted in `runs_failed`) and tried again by the next import, and an interrupted write is
finished by the next import. Only a failure of the list of runs itself raises.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from orch_it import settings
from orch_it.events import schema
from orch_it.events.ids import import_id
from orch_it.events.store import Store
from orch_it.testruns import outcome

SOURCE = "github"
WORKFLOW_FILE = "tests.yml"
REQUIRED_JOBS_SETTING = "github_required_jobs"
DEFAULT_REQUIRED_JOBS = ["linux", "windows"]
JOB_TIMEOUT_SETTING = "github_job_timeout_minutes"
DEFAULT_JOB_TIMEOUT_MINUTES = 15
NEWEST_RUNS_SETTING = "github_live_newest_runs"  # how many of the newest runs the live import test judges
DEFAULT_NEWEST_RUNS = 20
MAX_PER_PAGE = 100  # GitHub's largest page: an unpaginated list of more runs would silently return 100
TEST_STEP = "pytest"  # the step of a job that runs the tests
REPORT_FILE = "junit.xml"
NIGHTLY_EVENTS = ("schedule", "workflow_dispatch")
API_TIMEOUT_S = 120
RUNS_LIST_TIMEOUT_S = 600  # every run of the workflow, page after page
DOWNLOAD_TIMEOUT_S = 300
ERROR_TEXT_CHARS = 300
GH_ERROR_CODES = ("gh_failed", "gh_timeout", "gh_bad_output", "bad_since", "bad_now", "bad_repo")
_STORED_RUN = re.compile(r"github:(\d+)/.+/(\d+)")

Gh = Callable[[list[str]], str]


class GithubError(Exception):
    """`code`: `gh_failed` (`gh` is missing or answered with a failure), `gh_timeout` (it did
    not answer in time), `gh_bad_output` (its answer is no JSON of the shape asked for),
    `bad_since`, `bad_now` (an argument that is no time, or has no zone), `bad_repo` (not
    `owner/name`)."""

    def __init__(self, code: str, detail: str = ""):
        if code not in GH_ERROR_CODES:
            raise ValueError(f"not a github error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass
class ImportReport:
    """`runs_read`: runs created since `since`, in progress ones too. `jobs`: test runs looked
    at, written or already stored (skipped jobs are none, a missing job is one). `written`,
    `already_stored`: events. `reports_missing`: jobs read that had no JUnit artifact to
    download (never uploaded, expired or listed under no name asked for). `runs_failed`: runs
    skipped because a `gh` call about them failed or answered badly, `failure_codes`: the
    codes seen, in order."""

    runs_read: int = 0
    jobs: int = 0
    written: int = 0
    already_stored: int = 0
    reports_missing: int = 0
    runs_failed: int = 0
    failure_codes: list[str] = field(default_factory=list)


def run_gh(args: list[str]) -> str:
    """The default `gh` runner: `gh <args>` bounded by a timeout, its stdout as text."""

    if args[:1] == ["run"]:
        timeout = DOWNLOAD_TIMEOUT_S
    elif any(f"/workflows/{WORKFLOW_FILE}/runs" in arg for arg in args):
        timeout = RUNS_LIST_TIMEOUT_S
    else:
        timeout = API_TIMEOUT_S
    try:
        done = subprocess.run(
            ["gh", *args], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
            env={**os.environ, "GH_PROMPT_DISABLED": "1"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise GithubError("gh_timeout", " ".join(args[:3])) from None
    except OSError as e:
        raise GithubError("gh_failed", f"gh could not be started: {e}") from None
    if done.returncode != 0:
        said = done.stderr.decode("utf-8", "replace").strip()[:ERROR_TEXT_CHARS]
        raise GithubError("gh_failed", f"{' '.join(args[:3])}: {said}")
    return done.stdout.decode("utf-8", "replace")


def import_github(store: Store, repo: str, *, gh: Gh | None = None, since: str | None = None,
                  now: datetime | None = None, newest: int | None = None) -> ImportReport:
    """Write the events of `repo`'s `tests` workflow runs the store does not hold yet.

    `gh` takes the argument list of one `gh` call and returns its stdout (default: the real
    CLI). `since`: only runs created at or after it (ISO 8601, a zone-less one is UTC). `now`
    bounds which runs can have superseded a cancelled one. Bad arguments and settings raise
    before `gh` is called; so does a failure of the list of runs. A failure about one run only
    skips that run (see `ImportReport.runs_failed`). `newest`: only the newest that many completed
    runs (created at or after `since`), a whole number from 1 to `MAX_PER_PAGE`, read with one
    unpaginated list call, so the number of `gh` calls is the same whatever the history holds;
    None: every run, page after page. In that mode only the listed completed runs can supersede
    a cancelled one: a cancelled run whose superseder is still running gets reason None."""

    gh = gh or run_gh
    if not schema.is_repo(repo):
        raise GithubError("bad_repo", str(repo))
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise GithubError("bad_now", str(now))
    floor = _since(since)
    if newest is not None:
        _check_newest(newest, "newest")
    config = settings.load()
    required = _required_jobs(config)
    job_bound = _job_timeout_minutes(config)

    every_run = _items(gh(_list_runs_args(repo, floor, newest)), "workflow_runs")
    for run in every_run:
        _need_id(run, "run")
        if _moment(run.get("created_at")) is None:
            raise GithubError("gh_bad_output", f"run {run['id']} has no created_at")
    runs = sorted((r for r in every_run if floor is None or _created(r) >= floor),
                  key=lambda r: (_created(r), r["id"]))
    report = ImportReport(runs_read=len(runs))
    stored = _stored_ends(store, repo)
    importer = _Importer(store, repo, gh, required, job_bound, every_run, now, report)
    with store.writer() as writer:
        for run in runs:
            if run.get("status") != "completed":
                continue
            done = stored.get((run["id"], _attempt(run)), 0)
            if done >= len(required):  # every required job has its end: nothing to ask gh
                report.jobs += done
                continue
            tally = (report.jobs, report.already_stored, report.reports_missing)
            try:
                importer.import_run(run, writer)
            except GithubError as e:
                report.jobs, report.already_stored, report.reports_missing = tally
                report.runs_failed += 1
                if e.code not in report.failure_codes:
                    report.failure_codes.append(e.code)
    return report


def _list_runs_args(repo: str, floor: datetime | None, newest: int | None) -> list[str]:
    listing = f"repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
    if newest is None:
        return ["api", "--paginate", f"{listing}?per_page=100"]
    since = f"&created=>={floor.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}" if floor else ""
    return ["api", f"{listing}?status=completed&per_page={newest}{since}"]


def newest_runs(config: dict | None = None) -> int:
    """The setting `github_live_newest_runs`: how many of the newest completed runs the live
    import test judges (default 20)."""

    count = (settings.load() if config is None else config).get(NEWEST_RUNS_SETTING, DEFAULT_NEWEST_RUNS)
    _check_newest(count, NEWEST_RUNS_SETTING)
    return count


def _check_newest(count: object, name: str) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_PER_PAGE:
        raise settings.SettingsError("bad_setting", f"{name} must be a whole number from 1 to {MAX_PER_PAGE}")


def _stored_ends(store: Store, repo: str) -> Counter:
    """(run id, attempt) -> how many `test_run` ends of the github role the store holds."""

    found = Counter()
    for (run,) in store.query(
            "SELECT run FROM events WHERE kind = 'step.ended' AND step = 'test_run'"
            " AND role = 'github' AND repo = ?", (repo,)).rows:
        match = _STORED_RUN.fullmatch(run or "")
        if match:
            found[(int(match.group(1)), int(match.group(2)))] += 1
    return found


def _since(since: str | None) -> datetime | None:
    if since is None:
        return None
    try:
        moment = datetime.fromisoformat(since)
    except (TypeError, ValueError):
        raise GithubError("bad_since", str(since)) from None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _required_jobs(config: dict) -> list[str]:
    names = config.get(REQUIRED_JOBS_SETTING, DEFAULT_REQUIRED_JOBS)
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name and name == name.strip() for name in names)):
        raise settings.SettingsError("bad_setting", REQUIRED_JOBS_SETTING)
    return list(dict.fromkeys(names))


def _job_timeout_minutes(config: dict) -> float:
    minutes = config.get(JOB_TIMEOUT_SETTING, DEFAULT_JOB_TIMEOUT_MINUTES)
    if (isinstance(minutes, bool) or not isinstance(minutes, (int, float))
            or not math.isfinite(minutes) or minutes <= 0):
        raise settings.SettingsError("bad_setting", JOB_TIMEOUT_SETTING)
    return minutes


def json_values(text: str) -> list:
    """The JSON values `gh api --paginate` printed one after the other (one per page)."""

    decoder, values, at = json.JSONDecoder(), [], 0
    while True:
        while at < len(text) and text[at].isspace():
            at += 1
        if at >= len(text):
            return values
        try:
            value, at = decoder.raw_decode(text, at)
        except ValueError:
            raise GithubError("gh_bad_output", text[at:at + 80]) from None
        values.append(value)


def _items(text: str, key: str) -> list[dict]:
    """The objects under `key` of every page; anything else in the answer is a bad output."""

    items = []
    for page in json_values(text):
        found = page.get(key) if isinstance(page, dict) else page
        if not isinstance(found, list) or not all(isinstance(item, dict) for item in found):
            raise GithubError("gh_bad_output", f"no list of objects `{key}`")
        items += found
    return items


def _objects(value: object, what: str) -> list[dict]:
    """`value` as a list of objects (None: an empty list), else a bad output."""

    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise GithubError("gh_bad_output", f"{what} is no list of objects")
    return value


def _need_id(item: dict, what: str) -> int:
    if type(item.get("id")) is not int:
        raise GithubError("gh_bad_output", f"a {what} with no id")
    return item["id"]


def _attempt(item: dict) -> int:
    attempt = item.get("run_attempt")
    return attempt if type(attempt) is int and attempt > 0 else 1


def _moment(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _created(run: dict) -> datetime:
    return _moment(run.get("created_at")) or datetime.min.replace(tzinfo=UTC)


def _local(moment: datetime) -> str:
    return moment.astimezone().isoformat(timespec="microseconds")


def _minutes(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max((end - start).total_seconds(), 0.0) / 60


def _os_of(job: dict) -> str | None:
    """`linux`, `windows` or `macos`, from the job's labels, else its name."""

    labels = [label for label in job.get("labels") or [] if isinstance(label, str)]
    for text in [*labels, str(job.get("name") or "")]:
        text = text.lower()
        if "ubuntu" in text or "linux" in text:
            return "linux"
        if "windows" in text:
            return "windows"
        if "macos" in text or text.startswith("mac"):
            return "macos"
    return None


@dataclass(frozen=True)
class _RunFacts:
    """What decides whether a newer run supersedes a run."""

    event: str | None
    sha: str | None
    branch: str | None
    pull_request: int | None
    created: datetime | None


def _facts(run: dict) -> _RunFacts:
    numbers = [p.get("number") for p in _objects(run.get("pull_requests"), "pull_requests")
               if type(p.get("number")) is int]
    return _RunFacts(run.get("event"), run.get("head_sha"), run.get("head_branch"),
                     numbers[0] if numbers else None, _moment(run.get("created_at")))


@dataclass
class _Target:
    """One test run to write: a real job (`job` set) or a required job the run lacks."""

    key: str  # `<run id>/<job id or job name>/<attempt>`
    name: str
    attempt: int
    job: dict | None

    @property
    def run(self) -> str:
        return f"{SOURCE}:{self.key}"

    def id(self, kind: str) -> str:
        return import_id(SOURCE, self.key, kind)

    def known_ids(self) -> list[str]:
        return [self.id(k) for k in ("step.started", "step.ended", "wait.started", "wait.ended")]


class _Importer:
    def __init__(self, store, repo, gh, required, job_bound, every_run, now, report):
        self.store, self.repo, self.gh = store, repo, gh
        self.required, self.job_bound, self.now, self.report = required, job_bound, now, report
        self.facts_of_run = {run["id"]: _facts(run) for run in every_run}

    def import_run(self, run: dict, writer) -> None:
        run_id, attempt = run["id"], _attempt(run)
        jobs = _items(self.gh(["api", "--paginate",
                               f"repos/{self.repo}/actions/runs/{run_id}/jobs?filter=all&per_page=100"]),
                      "jobs")
        targets, present = [], set()
        for job in jobs:
            name = job.get("name")
            if not isinstance(name, str) or name.strip() not in self.required:
                continue
            name, job_attempt = name.strip(), _attempt(job)
            job_id = _need_id(job, "job")
            _objects(job.get("steps"), "steps")
            if job_attempt <= attempt:
                present.add(name)
            if job.get("conclusion") != "skipped":
                targets.append(_Target(f"{run_id}/{job_id}/{job_attempt}", name, job_attempt, job))
        targets += [_Target(f"{run_id}/{name}/{attempt}", name, attempt, None)
                    for name in self.required if name not in present]

        stored = self.store.existing_ids([i for t in targets for i in t.known_ids()])
        pending = []
        for target in targets:
            self.report.jobs += 1
            if target.id("step.ended") in stored:
                self.report.already_stored += sum(i in stored for i in target.known_ids())
            else:
                pending.append(target)
        if not pending:
            return

        events = self._events(run, pending)
        stored = self.store.existing_ids([e["id"] for e in events])
        self.report.already_stored += len(stored)
        for event in events:
            if event["id"] not in stored:
                writer.write(event)
                self.report.written += 1

    # --- reading what a run needs from gh --------------------------------------------------

    def _events(self, run: dict, pending: list[_Target]) -> list[dict]:
        listed, billed = set(), {}
        if any(t.job is not None for t in pending):
            artifacts = _items(self.gh(["api", "--paginate",
                                        f"repos/{self.repo}/actions/runs/{run['id']}/artifacts?per_page=100"]),
                               "artifacts")
            listed = {a.get("name") for a in artifacts if not a.get("expired")}
            billed = self._billed(run["id"])
        events = []
        for target in pending:
            events += self._target_events(run, target, listed, billed)
        return events

    def _billed(self, run_id: int) -> dict[int, int]:
        """Job id -> billable minutes (a job is billed by the started minute: rounded up)."""

        found = {}
        for page in json_values(self.gh(["api", f"repos/{self.repo}/actions/runs/{run_id}/timing"])):
            if not isinstance(page, dict) or not isinstance(page.get("billable") or {}, dict):
                raise GithubError("gh_bad_output", "timing")
            for system in (page.get("billable") or {}).values():
                if not isinstance(system, dict):
                    raise GithubError("gh_bad_output", "timing")
                for job in _objects(system.get("job_runs"), "job_runs"):
                    ms = job.get("duration_ms")
                    if isinstance(ms, (int, float)) and not isinstance(ms, bool) and "job_id" in job:
                        found[job["job_id"]] = math.ceil(ms / 60000)
        return found

    def _judge(self, run: dict, target: _Target, listed: set, cut: bool) -> tuple[outcome.RunEnd, bool]:
        """How the job's report reads, and whether it had one to download."""

        name = next((n for n in (f"junit-{target.name}-{target.attempt}", f"junit-{target.name}")
                     if n in listed), None)
        if name is None:
            return outcome.judge(None, cut=cut), False
        with tempfile.TemporaryDirectory(prefix="orch-it-gh-") as folder:
            self.gh(["run", "download", str(run["id"]), "-R", self.repo, "-n", name, "-D", folder])
            report = next(iter(sorted(Path(folder).rglob(REPORT_FILE))), None)
            return outcome.judge(report, cut=cut), report is not None

    # --- the events of one job -------------------------------------------------------------

    def _target_events(self, run: dict, target: _Target, listed: set, billed: dict) -> list[dict]:
        job = target.job
        created = _moment(run.get("created_at"))
        if job is None:  # a required job the run never had
            if run.get("conclusion") == "cancelled":
                end = outcome.RunEnd("stopped", self._superseded_reason(run), (), False)
            else:
                end = outcome.RunEnd("no_result", "job_missing", (), False)
            started_at, ended_at = created, _moment(run.get("updated_at")) or created
            run_os = _os_of({"name": target.name})
            minutes = {"billed_minutes": 0, "start_minutes": None, "run_minutes": None}
            wait = None
        else:
            job_created, job_started = _moment(job.get("created_at")), _moment(job.get("started_at"))
            job_ended = _moment(job.get("completed_at"))
            hit_bound = self._ran_to_the_bound(job, job_started, job_ended)
            end, found = self._judge(run, target, listed, hit_bound or job.get("conclusion") == "timed_out")
            if not found:
                self.report.reports_missing += 1
            end = self._by_conclusion(run, job, end, hit_bound)
            started_at = job_started or job_created or created
            ended_at = job_ended or _moment(run.get("updated_at")) or started_at
            run_os = _os_of(job)
            step = next((s for s in _objects(job.get("steps"), "steps") if s.get("name") == TEST_STEP), {})
            step_start, step_end = _moment(step.get("started_at")), _moment(step.get("completed_at"))
            minutes = {"billed_minutes": billed.get(job.get("id"), 0),
                       "start_minutes": _minutes(job_created, step_start),
                       "run_minutes": _minutes(step_start, step_end)}
            # a clock that put the start before the creation gives a wait of 0, never a negative one
            wait = (job_created, max(job_started, job_created)) if job_created and job_started else None

        pr = _facts(run).pull_request
        event_name = run.get("event")
        envelope = {"repo": self.repo, "ticket": None, "role": "github", "session": None,
                    "parent_session": None, "run": target.run, "note": None}
        events = [self._event("step.started", target.id("step.started"), started_at,
                              envelope | {"step": "test_run"}, {
                                  "place": "github", "chosen": "github", "reason": None,
                                  "run_os": run_os, "commit": run.get("head_sha"),
                                  "branch": run.get("head_branch"), "pr": pr,
                                  "selection": "nightly" if event_name in NIGHTLY_EVENTS else "full",
                                  "event": event_name})]
        if wait is not None:
            events += [
                self._event("step.started", target.id("wait.started"), wait[0],
                            envelope | {"step": "wait"},
                            {"reason": "github_queue", "on": target.run}),
                self._event("step.ended", target.id("wait.ended"), wait[1],
                            envelope | {"step": "wait"}, {"outcome": "passed", "reason": None})]
        for number, detail in enumerate(end.result_details(), start=1):
            events.append(self._event("test.result", target.id(f"test.result/{number}"), ended_at,
                                      envelope | {"step": "test_run"}, detail))
        events.append(self._event("step.ended", target.id("step.ended"), ended_at,
                                  envelope | {"step": "test_run"}, end.ended_detail() | minutes))
        return events

    def _ran_to_the_bound(self, job: dict, started: datetime | None, ended: datetime | None) -> bool:
        """A cancelled job that ran for the bound of the workflow's `timeout-minutes` or more:
        GitHub may call such a job cancelled rather than timed out."""

        minutes = _minutes(started, ended)
        return job.get("conclusion") == "cancelled" and minutes is not None and minutes >= self.job_bound

    def _by_conclusion(self, run: dict, job: dict, end: outcome.RunEnd, hit_bound: bool) -> outcome.RunEnd:
        conclusion = job.get("conclusion")
        if conclusion == "success":
            return replace(end, outcome="passed", reason=None)
        if conclusion == "failure":
            if end.outcome == "failed":
                return end
            return replace(end, outcome="no_result", reason="summary_missing")
        if conclusion == "cancelled":
            if hit_bound:
                return end  # judged as cut
            return replace(end, outcome="stopped", reason=self._superseded_reason(run))
        if conclusion == "timed_out":
            return end  # judged as cut
        return replace(end, outcome="no_result", reason="summary_missing")

    def _superseded_reason(self, run: dict) -> str | None:
        return "superseded" if self._superseded(run) else None

    def _superseded(self, run: dict) -> bool:
        mine = self.facts_of_run[run["id"]]
        if mine.created is None:
            return False
        for other_id, other in self.facts_of_run.items():
            if (other_id == run["id"] or other.created is None or not mine.created < other.created <= self.now
                    or other.event != mine.event or other.sha == mine.sha):
                continue
            if mine.event == "pull_request" and mine.pull_request is not None and other.pull_request is not None:
                same_line = other.pull_request == mine.pull_request
            else:
                same_line = mine.branch is not None and other.branch == mine.branch
            if same_line:
                return True
        return False

    @staticmethod
    def _event(kind: str, event_id: str, at: datetime, envelope: dict, detail: dict) -> dict:
        return {"id": event_id, "kind": kind, "at": _local(at), "turn": None, **envelope,
                "detail": detail}
