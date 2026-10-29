"""The closing check (the home repo's ticket): a closing word before a reference (`fixes <URL>`, `Resolves #N`,
the word and the reference across a line break too) in the text a pull request lands with is refused before
GitHub reads it.
GitHub closes the ticket when such a text reaches the default branch: by a commit's message, or by the title
and body that become the merge commit's text.

The words are found by `marks.closing_keywords` (the rule `pr open` uses, #400); this module only reads the
texts: the messages of the commits `origin/<base>..<head>` and, in a workflow, the title and body that the
workflow passes in the environment (`PR_TITLE`, `PR_BODY`), never on the shell line. A text the environment
does not carry is not checked.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from orch_it.merge import git as merge_git
from orch_it.merge import marks

TITLE_ENV = "PR_TITLE"
BODY_ENV = "PR_BODY"
SEP = "\x1f"  # between a commit's sha and its message in `git log`; no message holds one
_WORD = re.compile(r"[A-Za-z]+")


class ClosingError(RuntimeError):
    """`code`: `git_failed` (the commits of the pull request could not be read)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class Hit:
    where: str  # a commit's short sha, `title` or `body`
    line: int  # the line of the text the closing word starts on
    phrase: str  # the closing word and its reference as written, the spaces and line breaks folded
    reference: str  # the reference alone: what `Refs` takes


@dataclass(frozen=True)
class Done:
    hits: list[Hit]
    sections: list[str]  # what was read: `N commits`, then `title` and `body` when the environment gave them


def hits_in(where: str, text: str) -> list[Hit]:
    """Every closing word before a reference in `text`, in order, with the line it starts on."""

    hits, cursor = [], 0
    for found in marks.closing_keywords(text):
        start = text.index(found, cursor)  # the matches come in order and do not overlap
        cursor = start + len(found)
        word = _WORD.match(found).group(0)
        hits.append(Hit(where=where, line=text.count("\n", 0, start) + 1, phrase=" ".join(found.split()),
                        reference=found[len(word):].lstrip(": \t\r\n")))
    return hits


def commit_texts(git: Callable[[list[str]], str], base: str, head: str) -> list[tuple[str, str]]:
    """(short sha, message) of every commit of `origin/<base>..<head>`, newest first."""

    out = git(["log", f"--format=%H{SEP}%B%x00", f"origin/{base}..{head}", "--"])
    texts = []
    for record in out.split("\x00"):
        sha, sep, message = record.lstrip("\n").partition(SEP)
        if sep:
            texts.append((sha[:12], message))
    return texts


def check(git: Callable[[list[str]], str], base: str, head: str, environ: Mapping[str, str]) -> Done:
    """The closing words of the commits of the pull request, and of its title and body when the
    environment carries them. A commit that cannot be read is `ClosingError`."""

    try:
        commits = commit_texts(git, base, head)
    except merge_git.GitError as e:
        raise ClosingError("git_failed", str(e)) from None
    texts = list(commits)
    texts += [(name, environ[env]) for name, env in (("title", TITLE_ENV), ("body", BODY_ENV)) if env in environ]
    hits = [hit for where, text in texts for hit in hits_in(where, text)]
    return Done(hits=hits, sections=[f"{len(commits)} commits"] + [where for where, _ in texts[len(commits):]])


def render(done: Done) -> str:
    """One line per hit, naming the commit or `title`/`body`, then the line `Refs` is written as; or the
    summary of a clean read."""

    if not done.hits:
        return f"closing check: {', '.join(done.sections)}: 0 hits\n"
    lines = [f"{hit.where} line {hit.line}: \"{hit.phrase}\" makes GitHub close the ticket on merge; "
             f"write `Refs {hit.reference}` instead" for hit in done.hits]
    count = len(done.hits)
    lines.append(f"closing check: {count} {'hit' if count == 1 else 'hits'}")
    return "\n".join(lines) + "\n"
