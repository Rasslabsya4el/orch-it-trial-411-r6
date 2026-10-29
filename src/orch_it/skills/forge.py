"""The forge of `skills update` and `skills report`: GitHub through `gh` (#397 decision 4).

`GithubForge` is the `Forge` of `orch_it.skills.update`, on the repo's one `gh` runner
(`orch_it.testruns.github.run_gh`: bounded, no prompt, no console window) or any `Gh` given: the
pull requests of `skills-upstream/` branches for `update`, the open issues for the daily `report`.
Any refusal is `SkillsError("forge_failed")`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime

from orch_it import labels, settings
from orch_it.merge import marks
from orch_it.merge.commands import required_jobs
from orch_it.merge.gh import GhPullHub
from orch_it.merge.pr_open import after_open
from orch_it.merge.settings import MergeSettings
from orch_it.settings import SettingsError
from orch_it.skills.copies import SkillsError
from orch_it.skills.update import BRANCH_PREFIX
from orch_it.testruns.github import Gh, GithubError, run_gh

LABEL_LIST_LIMIT = "500"
_PULL_NUMBER = re.compile(r"(?:.*/pull/)?([1-9][0-9]*)/?")  # a pull request's url, or its bare number


class GithubForge:
    def __init__(self, repo: str, *, gh: Gh | None = None, base: str = "main"):
        self.repo = repo
        self.base = base
        self._run = gh or run_gh

    def _api_lines(self, path: str, pick: str) -> list[dict]:
        """Every page of a list the API gives, each item `pick` keeps as one JSON line."""

        try:
            said = self._run(["api", "--paginate", path, "--jq", pick])
        except GithubError as e:
            raise SkillsError("forge_failed", str(e)) from None
        try:
            return [json.loads(line) for line in said.splitlines() if line.strip()]
        except ValueError as e:
            raise SkillsError("forge_failed", f"gh api {path}: {e}") from None

    def _with_body(self, body: str, *args: str) -> str:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "body.md")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
            return self._gh(*args, "--body-file", path)

    def _labels(self) -> set[str]:
        return {item["name"] for item in json.loads(self._gh("label", "list", "--limit", LABEL_LIST_LIMIT,
                                                             "--json", "name") or "[]")}

    def _ensure_label(self, label: str) -> None:
        if label in self._labels():
            return
        want = labels.label_set()[label]  # the layer's own set says how it looks (#398 decision 6)
        try:
            self._gh("label", "create", label, "--color", want["color"], "--description", want["description"])
        except SkillsError:
            if label not in self._labels():  # not made by another run at the same moment
                raise

    def _gh(self, *args: str) -> str:
        try:
            return self._run([*args, "--repo", self.repo])
        except GithubError as e:
            raise SkillsError("forge_failed", str(e)) from None

    def update_prs(self) -> list[dict]:
        """Every pull request of a skills-update branch, of any state and age: every page of the
        repo's pull requests, newest first, filtered by branch."""

        pick = (f'.[] | select(.head.ref | startswith("{BRANCH_PREFIX}")) | {{url: .html_url, branch: .head.ref,'
                ' state: (if .merged_at then "MERGED" elif .state == "open" then "OPEN" else "CLOSED" end)}')
        return self._api_lines(f"repos/{self.repo}/pulls?state=all&per_page=100", pick)

    def create_pr(self, branch: str, title: str, body: str) -> str:
        said = self._with_body(body, "pr", "create", "--base", self.base, "--head", branch, "--title", title)
        return said.strip().splitlines()[-1]

    def add_label(self, pr: str, label: str) -> None:
        self._ensure_label(label)
        self._gh("pr", "edit", pr, "--add-label", label)

    def update_branch(self, pr: str) -> None:
        self._gh("pr", "update-branch", pr)

    def mark_ready(self, pr: str) -> None:
        """The ready mark and auto-merge of a pull request `update` opened or brought up to date (#424): it enters
        the merge line. Already in the line at its head, it keeps its one mark and only auto-merge is set again."""

        found = _PULL_NUMBER.fullmatch(pr.strip())
        if found is None:
            raise SkillsError("forge_failed", f"{pr[:120]}: no pull request number")
        number = int(found.group(1))
        try:
            hub = GhPullHub(gh=self._run)
            pull = hub.pull(self.repo, number)
            if pull is None:
                raise SkillsError("forge_failed", f"{self.repo}#{number}: no such pull request")
            raw = settings.load()
            method, required = MergeSettings.from_settings(raw, self.repo).merge_method, required_jobs(raw)
            ready = marks.ready_since(marks.of_layer(hub.pull_comments(self.repo, number), self.repo.split("/")[0]))
            marked = ready is not None and ready.head == pull.head_sha  # a re-used pull request already marked
            after_open(hub, self.repo, number, pull.head_sha, None, datetime.now(UTC), method, pull.base, required,
                       mark_written=marked)
        except (GithubError, SettingsError) as e:
            raise SkillsError("forge_failed", str(e)) from None

    def close_pr(self, pr: str, comment: str) -> None:
        self._gh("pr", "close", pr, "--comment", comment)

    def update_issues(self) -> list[dict]:
        """Every issue of any state (pull requests left out): url, title, label names, state."""

        pick = ('.[] | select(.pull_request | not) | {url: .html_url, title: .title, labels: [.labels[].name],'
                ' state: (if .state == "open" then "OPEN" else "CLOSED" end)}')
        return self._api_lines(f"repos/{self.repo}/issues?state=all&per_page=100", pick)

    def create_issue(self, title: str, body: str, label: str) -> str:
        self._ensure_label(label)
        return self._with_body(body, "issue", "create", "--title", title, "--label", label).strip().splitlines()[-1]

    def edit_issue(self, url: str, body: str) -> None:
        self._with_body(body, "issue", "edit", url)

    def close_issue(self, url: str, comment: str) -> None:
        self._gh("issue", "close", url, "--comment", comment)
