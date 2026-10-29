"""The brain's settings of one repo (orchestration-layer#403 decisions 2 and 3), in the section
`repos.<owner/name>` of the installation's settings file:

- `brain_chat_title` (default `Brain: <repo name>`, the GitHub repo's name): the title of the
  owner's session that is the brain (the brain title); the session hook finds the brain by it;
- `brain_handoff_issue` (no default): the number of the repo's pinned hand-off issue, written by
  `python -m orch_it brain handoff ensure`.

A value of the wrong shape raises `SettingsError("bad_setting")` naming it.
"""

from __future__ import annotations

from dataclasses import dataclass

from orch_it.settings import SettingsError, repo_section

TITLE_SETTING = "brain_chat_title"
HANDOFF_SETTING = "brain_handoff_issue"
TITLE_CHARS = 200  # a chat title longer than this is no title


@dataclass(frozen=True)
class BrainSettings:
    chat_title: str
    handoff_issue: int | None

    @classmethod
    def of(cls, raw: dict, repo: str) -> BrainSettings:
        own = repo_section(raw, repo)
        title = own.get(TITLE_SETTING, default_title(repo))
        if (not isinstance(title, str) or not title.strip() or len(title) > TITLE_CHARS
                or any(c in title for c in "\r\n")):
            raise SettingsError("bad_setting", f"repos.{repo}.{TITLE_SETTING}: one line of text")
        return cls(title, handoff_issue_of(raw, repo))


def handoff_issue_of(raw: dict, repo: str) -> int | None:
    """The setting `brain_handoff_issue` alone: a reader of it never fails on the title."""

    number = repo_section(raw, repo).get(HANDOFF_SETTING)
    if number is not None and (isinstance(number, bool) or not isinstance(number, int) or number < 1):
        raise SettingsError("bad_setting", f"repos.{repo}.{HANDOFF_SETTING}: an issue number")
    return number


def default_title(repo: str) -> str:
    return f"Brain: {repo.rsplit('/', 1)[-1]}"
