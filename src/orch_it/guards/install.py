"""`guards install` and the facts it and `guards status` read about the hooks (#395 decision 6, row 9).

`install` sets git's `core.hooksPath` to the repo's tracked hooks folder (`hooks`), so the tracked
`hooks/pre-push` runs on every push. It never overwrites: another `core.hooksPath`, at any level
of git's configuration, is refused with a coded error naming it (S09-041); our own folder is
installed already, and installing again changes nothing. Each call writes one `check` step
`guard_install` with the facts. The Claude Code half of the wiring is a tracked file
(`.claude/settings.json`), so it is only read: present or not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from orch_it.events.ids import live_id
from orch_it.guards import hook, scanner

HOOKS_FOLDER = "hooks"
PRE_PUSH_FILE = "pre-push"
CLAUDE_SETTINGS = (".claude", "settings.json")
CLAUDE_HOOK_COMMAND = "orch_it hook pre-tool-use"
INSTALL_ERROR_CODES = ("hooks_path_taken", "hook_file_missing", "not_a_repo", "git_failed")
GIT_SECONDS = 30
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CHECK_ROLE = "script"


class InstallError(Exception):
    """An install that cannot be done. `code` is one of `INSTALL_ERROR_CODES`: `hooks_path_taken`
    (another `core.hooksPath` is set; the detail names it and it is left as it was),
    `hook_file_missing` (the repo tracks no `hooks/pre-push`), `not_a_repo` (the folder is in no
    git work tree), `git_failed` (git could not be run or answered a failure)."""

    def __init__(self, code: str, detail: str = ""):
        if code not in INSTALL_ERROR_CODES:
            raise ValueError(f"unknown install error code: {code!r}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class Installed:
    """What `install` did. `hooks_path`: the value of `core.hooksPath` now; `changed`: False when
    our folder was set already; `claude_hook`: whether `.claude/settings.json` holds the
    PreToolUse entry."""

    hooks_path: str
    changed: bool
    claude_hook: bool


def _git(work_tree: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(work_tree), *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=GIT_SECONDS,
                              stdin=subprocess.DEVNULL, env=scanner._git_env(), creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as e:
        raise InstallError("git_failed", type(e).__name__) from None


def top_level(work_tree: Path) -> Path:
    """The root of the git work tree around `work_tree`, or `InstallError("not_a_repo")`."""

    done = _git(work_tree, "rev-parse", "--show-toplevel")
    if done.returncode != 0 or not done.stdout.strip():
        raise InstallError("not_a_repo", "the folder is in no git work tree")
    return Path(done.stdout.strip())


def read_hooks_path(root: Path) -> str | None:
    """The `core.hooksPath` git sees in `root` (local, user or system configuration), or None."""

    done = _git(root, "config", "--get", "core.hooksPath")
    if done.returncode == 1:
        return None
    if done.returncode != 0:
        raise InstallError("git_failed", (done.stderr.strip().splitlines() or ["git config failed"])[0])
    return done.stdout.strip() or None


def is_ours(value: str | None, root: Path) -> bool:
    """Whether a `core.hooksPath` value names the repo's tracked hooks folder: `hooks` in any
    spelling git reads, or the absolute path of it."""

    if not value:
        return False
    plain = value.strip().replace("\\", "/")
    while plain.startswith("./"):
        plain = plain[2:]
    if plain.rstrip("/") == HOOKS_FOLDER:
        return True
    try:
        return os.path.normcase(os.path.realpath(value)) == os.path.normcase(
            os.path.realpath(root / HOOKS_FOLDER)) and os.path.isabs(value)
    except (OSError, ValueError):
        return False


def pre_push_present(root: Path) -> bool:
    return (root / HOOKS_FOLDER / PRE_PUSH_FILE).is_file()


def claude_hook_present(root: Path) -> bool:
    """Whether `.claude/settings.json` wires `python -m orch_it hook pre-tool-use` on PreToolUse.
    A missing, unreadable or differently shaped file holds no hook."""

    try:
        settings = json.loads(root.joinpath(*CLAUDE_SETTINGS).read_text("utf-8-sig"))
        entries = settings["hooks"]["PreToolUse"]
        return any(isinstance(h, dict) and CLAUDE_HOOK_COMMAND in str(h.get("command", ""))
                   for entry in entries if isinstance(entry, dict)
                   for h in (entry.get("hooks") or []))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def write_check(store, repo: str, check: str, target: str, outcome: str, findings: list,
                facts: dict) -> None:
    """One `check` step, started and ended: `check` is one of the `check` codes, `findings` what
    is wrong (codes), `facts` what was read. Never raises: a store that cannot be written loses
    the event, not the command."""

    run = live_id()

    def event(kind: str, detail: dict) -> dict:
        return {**hook._envelope(kind, repo, None, detail), "role": CHECK_ROLE, "step": "check",
                "run": run}

    try:
        with store.writer() as writer:
            writer.write(event("step.started", {"check": check, "target": target}))
            writer.write(event("step.ended", {"outcome": outcome, "findings": findings, "facts": facts}))
    except Exception as e:
        print(f"orch-it guards: check event not written: {type(e).__name__}", file=sys.stderr)


def install(work_tree: Path | str, store, *, repo: str) -> Installed:
    """Point the repo's `core.hooksPath` at its tracked hooks folder. Raises `InstallError`:
    `hooks_path_taken` when another value is set (left unchanged), `hook_file_missing` when
    `hooks/pre-push` is not in the work tree, `not_a_repo`. Writes the `guard_install` check
    for a done install and for a refusal alike."""

    root = top_level(Path(work_tree))
    claude = claude_hook_present(root)
    current = read_hooks_path(root)
    facts = {"hooks_path": current, "changed": False, "pre_push_hook": False, "claude_hook": claude}
    target = repo
    if current is not None and not is_ours(current, root):
        write_check(store, repo, "guard_install", target, "refused", ["hooks_path_taken"], facts)
        raise InstallError("hooks_path_taken",
                           f"core.hooksPath is already {current!r}; it is not overwritten")
    if not pre_push_present(root):
        write_check(store, repo, "guard_install", target, "refused", ["hook_file_missing"], facts)
        raise InstallError("hook_file_missing", f"{HOOKS_FOLDER}/{PRE_PUSH_FILE} is not in the work tree")
    changed = current is None
    if changed:
        done = _git(root, "config", "--local", "core.hooksPath", HOOKS_FOLDER)
        if done.returncode != 0:
            raise InstallError("git_failed", (done.stderr.strip().splitlines() or ["git config failed"])[0])
    facts = {"hooks_path": current if not changed else HOOKS_FOLDER, "changed": changed,
             "pre_push_hook": True, "claude_hook": claude}
    write_check(store, repo, "guard_install", target, "passed" if claude else "failed",
                [] if claude else ["claude_hook_missing"], facts)
    return Installed(facts["hooks_path"], changed, claude)
