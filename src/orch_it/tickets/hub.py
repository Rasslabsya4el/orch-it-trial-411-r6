"""What the ticket commands need of GitHub (#398 decisions 4 and 5), in the tickets' own terms.

`Hub` is the seam: `GhHub` answers it through `gh api` for a real repo; the tests answer it
from memory. A call that fails raises `orch_it.testruns.github.GithubError`; a thing that is
not there (an issue, a run, a file) is `None`, never an error.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from orch_it.testruns.github import Gh, GithubError, json_values, run_gh

# the hidden mark of every comment the layer writes: such a comment is never a person's (PARK-9)
LAYER_MARK = "<!-- orch-it -->"


@dataclass(frozen=True)
class Issue:
    repo: str
    number: int
    title: str
    body: str
    labels: tuple[str, ...]  # names as GitHub holds them
    open: bool
    state_reason: str | None  # of a closed issue: `completed`, `not_planned`, ...
    is_pr: bool
    merged: bool  # a pull request merged


@dataclass(frozen=True)
class Comment:
    author: str
    bot: bool  # written by a bot account (a GitHub App, `github-actions`)
    body: str
    at: datetime

    @property
    def by_a_person(self) -> bool:
        return not self.bot and LAYER_MARK not in self.body


@dataclass(frozen=True)
class RunState:
    ended: bool
    passed: bool  # ended green


class Hub(Protocol):
    def issue(self, repo: str, number: int) -> Issue | None: ...

    def parent(self, repo: str, number: int) -> tuple[str, int] | None:
        """The ticket's parent as (repo, number), from GitHub's sub-issues."""

    def parked(self, repo: str) -> list[int]:
        """The numbers of the repo's open issues carrying the `parked` label or a `Parked:` line
        in their body (a line with no label is a park the reader must see, PARK-4)."""

    def write(self, repo: str, number: int, body: str, labels: list[str]) -> None:
        """Set the issue's body and its whole label set in one request."""

    def remove_label(self, repo: str, number: int, name: str) -> None:
        """Take one label off the issue, the others untouched; a label not on it is no error."""

    def comment(self, repo: str, number: int, body: str) -> None: ...

    def comments(self, repo: str, number: int) -> list[Comment]: ...

    def parked_at(self, repo: str, number: int) -> datetime | None:
        """When the `parked` label was last put on the issue."""

    def run(self, repo: str, run_id: str) -> RunState | None:
        """A GitHub Actions run of the repo."""

    def layer_version(self, repo: str) -> str | None:
        """The orch-it version the repo's default branch runs, `X.Y.Z`."""


class ListingHub(Hub, Protocol):
    def listed(self, repo: str, label: str) -> list[Issue]:
        """The repo's open issues (no pull requests) carrying the label."""


VERSION_FILE = ".orch-it/version"  # the version an installed repo runs (part 8 writes it)
_PROJECT_NAME = re.compile(r'^name\s*=\s*"orch-it"\s*$', re.MULTILINE)
_PROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"\n]+)"\s*$', re.MULTILINE)
_PARENT_QUERY = ("query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) "
                 "{ issue(number: $number) { parent { number repository { nameWithOwner } } } } }")
_PINNED_QUERY = ("query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) "
                 "{ pinnedIssues(first: 10) { nodes { issue { number } } } } }")
_PIN_MUTATION = "mutation($id: ID!) { pinIssue(input: {issueId: $id}) { issue { number } } }"
READ_RETRIES = 2  # a read that times out is asked again this many times, then its timeout is raised
READ_RETRY_PAUSE_S = 1  # seconds between two tries of one read


def _is_read(args: list[str]) -> bool:
    """Whether a `gh` call only reads: a `gh api` call whose method is GET, named by `-X`, `-XGET`, `--method` or
    `--method=GET`; with no method named, GET unless a field or an input makes `gh` send a POST. Any other
    spelling of a method that is not GET is a write."""

    if args[:1] != ["api"]:
        return False
    for i, arg in enumerate(args):
        if arg in ("-X", "--method"):
            return args[i + 1:i + 2] == ["GET"]
        if arg.startswith("--method="):
            return arg[len("--method="):] == "GET"
        if arg.startswith("-X"):
            return arg[2:] == "GET"
    return not any(arg in ("-f", "-F", "--field", "--raw-field", "--input") for arg in args)


class GhHub:
    """The `Hub` over `gh api` (the user's own `gh` login). Every call is bounded by
    `run_gh`'s timeout; a 404 is "not there", any other failure raises `GithubError`. A read
    that times out is asked again (`READ_RETRIES`, `READ_RETRY_PAUSE_S` apart); a write is
    called once."""

    def __init__(self, gh: Gh | None = None):
        self.gh = gh or run_gh
        self.sleep: Callable[[float], None] = time.sleep  # the pause between two tries of a read

    def _read(self, args: list[str]) -> str:
        """The answer of one `gh` call. A read (`_is_read`) that times out is tried again after the
        pause, `READ_RETRIES` times; its last timeout is raised as it is. Any other call is tried once."""

        if not _is_read(args):
            return self.gh(args)
        for _ in range(READ_RETRIES):
            try:
                return self.gh(args)
            except GithubError as e:
                if e.code != "gh_timeout":
                    raise
            self.sleep(READ_RETRY_PAUSE_S)
        return self.gh(args)

    def issue(self, repo: str, number: int) -> Issue | None:
        found = self._object(["api", f"repos/{repo}/issues/{number}"], missing_ok=True)
        return None if found is None else _issue(repo, found)

    def listed(self, repo: str, label: str) -> list[Issue]:
        """The repo's open issues carrying the label, pull requests left out (the brain's hand-off
        home and the queues read it)."""

        items = self._list(["api", "--paginate",
                            f"repos/{repo}/issues?labels={quote(label, safe='')}&state=open&per_page=100"])
        return [_issue(repo, item) for item in items if "pull_request" not in item]

    def create(self, repo: str, title: str, body: str, labels: list[str]) -> int:
        made = json_values(self._send(["api", "-X", "POST", f"repos/{repo}/issues"],
                                      {"title": title, "body": body, "labels": labels}))
        if len(made) != 1 or not isinstance(made[0], dict) or not isinstance(made[0].get("number"), int):
            raise GithubError("gh_bad_output", f"{repo}: the new issue has no number")
        return made[0]["number"]

    def pinned(self, repo: str) -> list[int]:
        owner, name = repo.split("/", 1)
        found = self._object(["api", "graphql", "-f", f"query={_PINNED_QUERY}", "-f", f"owner={owner}",
                              "-f", f"name={name}"])
        nodes = ((((found.get("data") or {}).get("repository") or {}).get("pinnedIssues") or {}).get("nodes") or [])
        return [node["issue"]["number"] for node in nodes if isinstance(node, dict)
                and isinstance(node.get("issue"), dict) and isinstance(node["issue"].get("number"), int)]

    def pin(self, repo: str, number: int) -> None:
        found = self._object(["api", f"repos/{repo}/issues/{number}"])
        node = found.get("node_id")
        if not isinstance(node, str) or not node:
            raise GithubError("gh_bad_output", f"{repo}#{number}: no node id")
        self._object(["api", "graphql", "-f", f"query={_PIN_MUTATION}", "-f", f"id={node}"])

    def ensure_label(self, repo: str, name: str) -> bool:
        from orch_it import labels  # the label module reads its set; imported when a label is wanted

        return labels.ensure(repo, name, gh=self.gh)

    def parent(self, repo: str, number: int) -> tuple[str, int] | None:
        owner, name = repo.split("/", 1)
        found = self._object(["api", "graphql", "-f", f"query={_PARENT_QUERY}", "-f", f"owner={owner}",
                              "-f", f"name={name}", "-F", f"number={number}"])
        issue = ((found.get("data") or {}).get("repository") or {}).get("issue") or {}
        parent = issue.get("parent")
        if not isinstance(parent, dict):
            return None
        return parent["repository"]["nameWithOwner"], int(parent["number"])

    def parked(self, repo: str) -> list[int]:
        items = self._list(["api", "--paginate", f"repos/{repo}/issues?labels=parked&state=open&per_page=100"])
        # GitHub's search finds the body lines; it may lag a write by a minute, the next run sees it
        found = self._object(["api", "--method", "GET", "search/issues", "-f",
                              f'q=repo:{repo} is:issue is:open in:body "Parked:"', "-f", "per_page=100"])
        if not isinstance(found.get("items", []), list):
            raise GithubError("gh_bad_output", "search/issues: no items")
        items += [item for item in found.get("items", []) if isinstance(item, dict)]
        return sorted({item["number"] for item in items if "pull_request" not in item
                       and isinstance(item.get("number"), int)})

    def write(self, repo: str, number: int, body: str, labels: list[str]) -> None:
        self._send(["api", "-X", "PATCH", f"repos/{repo}/issues/{number}"], {"body": body, "labels": labels})

    def remove_label(self, repo: str, number: int, name: str) -> None:
        try:  # GitHub answers with the labels left: nothing here to read
            self.gh(["api", "-X", "DELETE", f"repos/{repo}/issues/{number}/labels/{quote(name, safe='')}"])
        except GithubError as e:
            if "HTTP 404" not in str(e):
                raise

    def comment(self, repo: str, number: int, body: str) -> None:
        self._send(["api", "-X", "POST", f"repos/{repo}/issues/{number}/comments"], {"body": body})

    def comments(self, repo: str, number: int) -> list[Comment]:
        said = []
        for item in self._list(["api", "--paginate", f"repos/{repo}/issues/{number}/comments?per_page=100"]):
            user = item.get("user") if isinstance(item.get("user"), dict) else {}
            at = _moment(item.get("created_at"))
            if at is not None:
                said.append(Comment(_text(user.get("login")) or "ghost", user.get("type") == "Bot",
                                    _text(item.get("body")), at))
        return said

    def parked_at(self, repo: str, number: int) -> datetime | None:
        times = [_moment(item.get("created_at"))
                 for item in self._list(["api", "--paginate", f"repos/{repo}/issues/{number}/events?per_page=100"])
                 if item.get("event") == "labeled" and isinstance(item.get("label"), dict)
                 and _text(item["label"].get("name")).lower() == "parked"]
        times = [moment for moment in times if moment is not None]
        return max(times) if times else None

    def run(self, repo: str, run_id: str) -> RunState | None:
        found = self._object(["api", f"repos/{repo}/actions/runs/{run_id}"], missing_ok=True)
        if found is None:
            return None
        ended = found.get("status") == "completed"
        return RunState(ended=ended, passed=ended and found.get("conclusion") == "success")

    def layer_version(self, repo: str) -> str | None:
        text = self._file(repo, VERSION_FILE)
        if text is not None and text.strip():
            return text.strip()
        project = self._file(repo, "pyproject.toml") or ""
        version = _PROJECT_VERSION.search(project)
        return version.group(1) if _PROJECT_NAME.search(project) and version else None

    def _file(self, repo: str, path: str) -> str | None:
        found = self._object(["api", f"repos/{repo}/contents/{path}"], missing_ok=True)
        if found is None or found.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(found.get("content") or "").decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            raise GithubError("gh_bad_output", f"{repo}:{path}: not text") from None

    def _send(self, args: list[str], payload: dict) -> str:
        """A write whose JSON goes in a file of its own, so no text is bounded by a command line;
        GitHub's answer."""

        folder = tempfile.mkdtemp(prefix="orch-it-gh-")
        path = Path(folder) / "input.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
            return self.gh([*args, "--input", str(path)])
        finally:
            path.unlink(missing_ok=True)
            os.rmdir(folder)

    def _object(self, args: list[str], *, missing_ok: bool = False) -> dict | None:
        try:
            values = json_values(self._read(args))
        except GithubError as e:
            if missing_ok and "HTTP 404" in str(e):
                return None
            raise
        if len(values) != 1 or not isinstance(values[0], dict):
            raise GithubError("gh_bad_output", f"{' '.join(args[:3])}: expected one object")
        return values[0]

    def _list(self, args: list[str]) -> list[dict]:
        items = []
        for page in json_values(self._read(args)):
            if not isinstance(page, list):
                raise GithubError("gh_bad_output", f"{' '.join(args[:3])}: expected a list")
            items += [item for item in page if isinstance(item, dict)]
        return items


def _issue(repo: str, found: dict) -> Issue:
    number = found.get("number")
    if not isinstance(number, int) or not isinstance(found.get("labels", []), list):
        raise GithubError("gh_bad_output", f"{repo}#{number}: no issue")
    pull = found.get("pull_request")
    return Issue(
        repo=repo, number=number, title=_text(found.get("title")), body=_text(found.get("body")),
        labels=tuple(label["name"] for label in found.get("labels") or []
                     if isinstance(label, dict) and isinstance(label.get("name"), str)),
        open=found.get("state") == "open", state_reason=found.get("state_reason"),
        is_pr=isinstance(pull, dict), merged=isinstance(pull, dict) and bool(pull.get("merged_at")))


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _moment(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
