"""git for the merge flow (#400): one bounded runner, never asking for a password, never opening a
console window, as `orch_it.testruns.github.run_gh` runs `gh`; a branch name is checked before it
reaches a command line, so no name reads as an option."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_S = 300
ERROR_CHARS = 300
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# git's own rules for a branch name (`git check-ref-format --branch`), kept to what a ticket's branch
# needs: no leading `-` or `/`, no `..`, `//`, `@{`, no space or control character, no `.lock` end
_BRANCH = re.compile(r"(?!-)(?!/)(?!.*(?:\.\.|//|@\{|\.lock$|/$|\.$))[A-Za-z0-9._/+-]{1,200}")


class GitError(RuntimeError):
    """`code`: `git_failed` (git could not be started, gave no answer in time, or failed)."""

    def __init__(self, code: str, detail: str = ""):
        if code != "git_failed":
            raise ValueError(f"not a git error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class GitDone:
    code: int
    out: str
    err: str


def run(args: list[str]) -> GitDone:
    """`git <args>`; its exit code is the caller's to read. A start failure or no answer in
    `TIMEOUT_S` is `GitError`."""

    try:
        done = subprocess.run(["git", *args], capture_output=True, timeout=TIMEOUT_S, stdin=subprocess.DEVNULL,
                              env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise GitError("git_failed", f"git {' '.join(args[:3])}: no answer in {TIMEOUT_S} s") from None
    except OSError as e:
        raise GitError("git_failed", f"git could not be started: {e}") from None
    return GitDone(done.returncode, done.stdout.decode("utf-8", "replace"),
                   done.stderr.decode("utf-8", "replace").strip()[:ERROR_CHARS])


def work_tree(path: Path, runner: Callable[[list[str]], GitDone] = run) -> Callable[[list[str]], str]:
    """`git -C <path> <args>` giving its stdout; a non-zero exit is `GitError`."""

    def git(args: list[str]) -> str:
        done = runner(["-C", str(path), *args])
        if done.code != 0:
            raise GitError("git_failed", f"git {' '.join(args[:2])}: {done.err}")
        return done.out

    return git


def branch_ok(name: str) -> bool:
    return isinstance(name, str) and bool(_BRANCH.fullmatch(name)) and name != "HEAD"
