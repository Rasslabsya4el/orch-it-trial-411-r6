"""The commit identity of orch-it's own commits and clones.

The repo's setting `repos.<owner/name>.commit_identity` is `noreply` (the default, also when there is
no settings file) or `any`. Under `noreply` the commit that `skills update` makes and the clone the
test machine makes carry the GitHub noreply identity of the gh user, read at run time and never
written here; under `any` git's own identity applies. `guards scan --range` refuses a commit whose author
or committer e-mail is no noreply address while its check is on (the cli reads the same setting; `--all`
is never checked).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

from orch_it import settings
from orch_it.guards import scanner
from orch_it.guards.prepush import github_repo_of
from orch_it.testruns import github

KEY = "commit_identity"
MODES = ("noreply", "any")
DEFAULT_MODE = "noreply"
NOREPLY_DOMAIN = "users.noreply.github.com"
_LOGIN = re.compile(r"[A-Za-z0-9-]+")
# `<digits>+<login>@users.noreply.github.com`, `<login>@users.noreply.github.com` and the web merge's
# committer `noreply@github.com`; nothing else is a noreply address
_NOREPLY = re.compile(rf"(?:\d+\+)?[A-Za-z0-9-]+@{re.escape(NOREPLY_DOMAIN)}|noreply@github\.com",
                      re.IGNORECASE)


class IdentityError(Exception):
    """`code`: `gh_failed` (`gh api user` failed or is no JSON object), `bad_user` (the gh user has
    no login or id of the shape asked for)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def mode_of(repo: str, raw: dict | None = None) -> str:
    """The repo's `commit_identity`: `noreply` when absent, else the value; a value of another kind
    raises `bad_setting`. `raw` is the settings object (default: the installation's file)."""

    raw = settings.load() if raw is None else raw
    value = settings.repo_section(raw, repo).get(KEY, DEFAULT_MODE)
    if value not in MODES:
        raise settings.SettingsError("bad_setting", f"repos.{repo}.{KEY}: one of {', '.join(MODES)}")
    return value


def current_user() -> dict:
    """The authenticated gh account now (`gh api user`), through the code base's gh runner."""

    try:
        user = json.loads(github.run_gh(["api", "user"]))
    except (github.GithubError, ValueError) as e:
        raise IdentityError("gh_failed", f"gh api user: {e}") from None
    if not isinstance(user, dict):
        raise IdentityError("gh_failed", "gh api user: no JSON object")
    return user


def noreply_identity(gh_user: dict) -> tuple[str, str]:
    """(name, e-mail) of the GitHub account: the login, and `<id>+<login>@users.noreply.github.com`."""

    login = gh_user.get("login") if isinstance(gh_user, dict) else None
    user_id = gh_user.get("id") if isinstance(gh_user, dict) else None
    if not (isinstance(login, str) and _LOGIN.fullmatch(login)
            and type(user_id) is int and user_id > 0):
        raise IdentityError("bad_user", "the gh user has no login or id")
    return login, f"{user_id}+{login}@{NOREPLY_DOMAIN}"


def is_noreply(email) -> bool:
    """Whether `email` is one of the three noreply forms of GitHub, and nothing else."""

    return isinstance(email, str) and _NOREPLY.fullmatch(email) is not None


def for_repo(repo: str) -> tuple[str, str] | None:
    """The identity orch-it's commits in `repo` carry: the gh user's noreply identity under
    `noreply`, None under `any` (git's own identity then applies)."""

    if mode_of(repo) != DEFAULT_MODE:
        return None
    return noreply_identity(current_user())


def for_origin(origin: str) -> dict | None:
    """The identity a clone of `origin` gets, as a request's `identity` (`name`, `email`): None for
    an origin that is no GitHub repo, else `for_repo` of it."""

    repo = github_repo_of(origin)
    found = for_repo(repo) if repo else None
    return None if found is None else {"name": found[0], "email": found[1]}


def commit_env(found: tuple[str, str] | None) -> dict | None:
    """The environment of a git call that commits as `found` (author and committer), or None for
    git's own identity."""

    if found is None:
        return None
    name, email = found
    return {**os.environ, "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}


def noreply_failures(work_tree: Path | str, commits: Iterable[str]) -> list[tuple[str, str]]:
    """(sha, `author` | `committer`) of each commit whose author or committer e-mail is no noreply
    address, in the order of `commits`; the e-mails themselves are never returned."""

    failures = []
    for sha, author, committer in scanner.emails_of_commits(Path(work_tree), commits):
        if not is_noreply(author):
            failures.append((sha, "author"))
        if not is_noreply(committer):
            failures.append((sha, "committer"))
    return failures
