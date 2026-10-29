"""A personal skill of the same name as a repo skill (#397 decision 3).

Claude Code loads a personal skill over a project skill of the same name, so a skill in the user's
own folder (`$CLAUDE_CONFIG_DIR/skills`, default `~/.claude/skills`) silently replaces the repo's
copy or ours. At session start a hook compares the two folders' skill names (letter case aside, as
Windows folders are) and writes one `skill.shadowed` event per clash; it never blocks the session
and prints nothing into it. A resumed or compacted session is the same session: no new events.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orch_it.events.store import Store
from orch_it.skills.copies import NO_WINDOW, SKILL_FILE, SKILLS_FOLDER, Manifest, SkillsError

SAME_SESSION_SOURCES = ("resume", "compact")
HOOK_GIT_TIMEOUT_SECONDS = 5  # a session waits for this hook: a git that hangs must not hold it


@dataclass(frozen=True)
class Clash:
    skill: str
    copy: bool | None       # his copy (in the manifest), a skill of ours, or None when the manifest is unreadable
    same_text: bool         # the personal SKILL.md is byte for byte the repo's


def personal_folder() -> Path:
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(config) if config else Path.home() / ".claude") / "skills"


def _skills(folder: Path) -> dict[str, Path]:
    """Each skill folder's name, folded to lower case, with its SKILL.md."""

    try:
        return {path.name.casefold(): path / SKILL_FILE for path in folder.iterdir() if (path / SKILL_FILE).is_file()}
    except OSError:  # no folder, or one we may not read: no skills
        return {}


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def clashes(repo: Path, personal: Path) -> list[Clash]:
    """The repo's skills that a personal skill of the same name shadows, sorted by name."""

    repo_skills = _skills(repo / SKILLS_FOLDER)
    personal_skills = _skills(personal)
    try:
        manifest = Manifest.load(repo)
        copied = {name.casefold() for name in manifest.skills} if manifest else set()
    except SkillsError:
        copied = None
    found = []
    for name in sorted(repo_skills.keys() & personal_skills.keys()):
        mine, theirs = _read(repo_skills[name]), _read(personal_skills[name])
        found.append(Clash(repo_skills[name].parent.name, None if copied is None else name in copied,
                           mine is not None and mine == theirs))
    return found


def write_shadowed(store: Store, *, repo: Path, personal: Path, repo_name: str, session: str | None) -> list[Clash]:
    found = clashes(repo, personal)
    if found:
        with store.writer() as writer:
            for clash in found:
                writer.write({
                    "kind": "skill.shadowed", "repo": repo_name, "ticket": None, "role": "hook",
                    "session": session, "parent_session": None, "turn": None, "step": None, "run": None,
                    "note": None,
                    "detail": {"skill": clash.skill, "copy": clash.copy, "same_text": clash.same_text}})
    return found


def project_root(cwd: str) -> Path:
    """The session's project: `CLAUDE_PROJECT_DIR` as Claude Code sets it for hooks, else the git work
    tree around `cwd` (a session may stand in a subfolder), else `cwd`."""

    if os.environ.get("CLAUDE_PROJECT_DIR"):
        return Path(os.environ["CLAUDE_PROJECT_DIR"])
    try:
        done = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True,
                              timeout=HOOK_GIT_TIMEOUT_SECONDS, creationflags=NO_WINDOW)
        if done.returncode == 0 and done.stdout.strip():
            return Path(done.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return Path(cwd)


def run_hook(raw: bytes, *, open_store: Callable[[], Store], repo_of: Callable[[Path], str | None]) -> None:
    """The SessionStart hook's work on its stdin `raw`: nothing for input it cannot read, for a
    resumed or compacted session, or for a project with no repo name (`repo_of` gives None)."""

    try:
        hook = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        return
    if not isinstance(hook, dict) or hook.get("source") in SAME_SESSION_SOURCES:
        return
    cwd = hook.get("cwd") if isinstance(hook.get("cwd"), str) and hook.get("cwd") else "."
    session = hook.get("session_id") if isinstance(hook.get("session_id"), str) else None
    root = project_root(cwd)
    repo_name = repo_of(root)
    if repo_name is None:
        return
    write_shadowed(open_store(), repo=root, personal=personal_folder(), repo_name=repo_name, session=session)
