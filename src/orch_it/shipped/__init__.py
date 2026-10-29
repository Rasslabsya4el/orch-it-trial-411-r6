"""The instructions every agent in a repo gets, shipped with the package.

`CLAUDE.md` and `CODING_STANDARDS.md` in this folder hold the rule that a value differing per repo,
account, machine or OS is a setting with its default, never a literal. The repo's root `CLAUDE.md`
imports the shipped `CLAUDE.md` by its import line; the root `CODING_STANDARDS.md` is a byte copy of
the shipped one, because the reviewer's Read tool does not follow import lines. `problems` says
what has drifted from that: one line per problem, none when the copies are fine.
"""

from __future__ import annotations

from pathlib import Path

RULE_FIRST_SENTENCE = (
    "A value that differs per repo, account, machine or OS is a setting with its default, "
    "never a literal in code, tests, rules or tickets."
)
SHIPPED_REL = Path("src/orch_it/shipped")
IMPORT_LINE = "@src/orch_it/shipped/CLAUDE.md"


def _bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _text(data: bytes | None) -> str | None:
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def problems(root: Path) -> list[str]:
    """One short line per drift of the instruction files in `root`; empty when they are fine."""

    root = Path(root)
    found: list[str] = []
    shipped = root / SHIPPED_REL
    for name in ("CLAUDE.md", "CODING_STANDARDS.md"):
        rel = (SHIPPED_REL / name).as_posix()
        text = _text(_bytes(shipped / name))
        if text is None:
            found.append(f"{rel}: missing or not utf-8")
        elif RULE_FIRST_SENTENCE not in text:
            found.append(f"{rel}: the rule's first sentence is missing")

    claude = _text(_bytes(root / "CLAUDE.md"))
    if claude is None:
        found.append("CLAUDE.md: missing or not utf-8")
    elif IMPORT_LINE not in claude.splitlines():
        found.append(f"CLAUDE.md: no line exactly {IMPORT_LINE}")

    shipped_standards = _bytes(shipped / "CODING_STANDARDS.md")
    root_standards = _bytes(root / "CODING_STANDARDS.md")
    if root_standards is None:
        found.append("CODING_STANDARDS.md: missing")
    elif shipped_standards is not None and root_standards != shipped_standards:
        found.append("CODING_STANDARDS.md: not byte-identical to src/orch_it/shipped/CODING_STANDARDS.md")
    return found
