"""How a text names a skill, and the check that every text of the repo names only skills it holds
(#397 decisions 4 and 5).

A text names a skill in three forms:
- a call: a sentence naming the Skill tool (any letter case, `Skill` or **Skill** too) and then
  the skill in quotes or backticks, his form ("Call the Skill tool with `tdd`", "Call the Skill
  tool twice, for "grilling" and "domain-modeling""): the first run of quoted names after the
  tool, so "with `tdd`, then run `pytest`" calls `tdd` alone; the sentence may run across lines
  of one paragraph, and "e.g." does not end it;
- a preload: an item of the `skills:` field in the frontmatter of a role file (`.claude/agents/`);
- a naming: a quoted name followed by the word "skill" ("the `research` skill");
- a slash command, `/name`, counted only when `name` is one of his skills the repo does not hold
  (left out, or gone upstream), since `/clear` or a `/settings` route are no skill.

`check` reads every Markdown file of the repo (role files, skills, docs) and returns one line per
name that is neither a copy of his nor ours (a folder of `.claude/skills/` with a `SKILL.md`), and
per use of the namespaced form of his plugin (`mattpocock-skills:<name>`), which the repo's plugin
setting switches off.
"""

from __future__ import annotations

import bisect
import os
import re
from pathlib import Path

from collections.abc import Iterator

from orch_it.skills.copies import MANIFEST_NAME, SKILL_FILE, SKILLS_FOLDER, Manifest, SkillsError

NAMESPACE = "mattpocock-skills"
AGENTS_FOLDER = ".claude/agents/"
# `docs/research`: records that quote what they studied verbatim (the old layer's texts included), never rules
NOT_SCANNED_FOLDERS = (".git", "venv", ".claude/worktrees", "build", "dist", "tests/fixtures", "docs/research")
NOT_SCANNED_NAMES = (".venv", "node_modules", ".pytest_cache", "__pycache__")
_SKILL_TOOL = re.compile(r"\b[`*_]{0,2}skill[`*_]{0,2}\s+tool\b", re.IGNORECASE)
_QUOTED_NAME = re.compile(r"[\"`“]([a-z0-9][a-z0-9:-]*)[\"`”]")
_LEAD_IN = re.compile(r"\b(?:with|for|named|calling)\b|[(:]", re.IGNORECASE)
_NEXT_NAME = re.compile(r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+)(?=[\"`“])")
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)[.!?;](?=\s|$)")
_NAMESPACED = re.compile(re.escape(NAMESPACE) + r":([a-z0-9][a-z0-9._-]*)?")
_NAMED_SKILL = re.compile(r"[\"`“]/?([a-z0-9][a-z0-9-]*)[\"`”]\s+skill\b(?!s)")
_SLASH = re.compile(r"(?<![\w/.~:-])/([a-z0-9][a-z0-9-]*)(?![\w/-]|\.\w)")
_PRELOAD_ITEM = re.compile(r"[ \t]*-[ \t]+[\"']?([^\"'\s#,\]]+)")
NOT_HELD = "which the repo holds neither as a copy nor as ours"


def _names_after(rest: str) -> list[str]:
    """The run of quoted names the sentence `rest` gives first: `"a"`, `"a" and "b"`, `"a", "b" or "c"`."""

    end = _SENTENCE_END.search(rest)
    clause = rest[:end.start()] if end else rest
    first = _QUOTED_NAME.search(clause)
    if first is None or not _LEAD_IN.search(clause, 0, first.start()):
        return []
    names, at = [first.group(1)], first.end()
    while (joint := _NEXT_NAME.match(clause, at)) and (name := _QUOTED_NAME.match(clause, joint.end())):
        names.append(name.group(1))
        at = name.end()
    return names


def _calls(text: str) -> list[tuple[int, str]]:
    """(line number, name) of each Skill tool call in `text`, the line being the one naming the tool."""

    found = []
    paragraph: list[tuple[int, str]] = []
    for number, line in [*enumerate(text.split("\n"), 1), (0, "")]:
        if line.strip():
            paragraph.append((number, line))
            continue
        if paragraph:
            parts, starts, length = [], [], 0
            for _, part in paragraph:
                starts.append(length)
                parts.append(" ".join(part.split()) + " ")
                length += len(parts[-1])
            joined = "".join(parts)
            for match in _SKILL_TOOL.finditer(joined):
                at_line = paragraph[bisect.bisect_right(starts, match.start()) - 1][0]
                found += [(at_line, name) for name in _names_after(joined[match.end():match.end() + 2000])]
        paragraph = []
    return found


def skill_calls(text: str) -> set[str]:
    """The names a text calls through the Skill tool."""

    return {name for _, name in _calls(text)}


def _preloads(text: str) -> list[tuple[int, str]]:
    """(line number, name) of each item of the frontmatter's `skills:` field, inline or a list."""

    lines = text.split("\n")
    if not lines or lines[0].removeprefix("\ufeff").strip() != "---":
        return []
    found, inside, flow = [], False, False
    for number, line in enumerate(lines[1:], 2):
        if line.strip() == "---":
            break
        if line.startswith("skills:"):
            inside = True
            inline = line[len("skills:"):].split("#", 1)[0].strip()
            flow = inline.startswith("[") and "]" not in inline
            found += [(number, item.strip().strip("\"'")) for item in inline.strip("[]").split(",") if item.strip()]
        elif inside and flow:  # a flow list `[` over several lines, to its `]`
            body = line.split("#", 1)[0]
            flow = "]" not in body
            found += [(number, item.strip().strip("\"'")) for item in body.strip().strip("[]").split(",")
                      if item.strip()]
        elif inside and (item := _PRELOAD_ITEM.match(line)):
            found.append((number, item.group(1)))
        elif line[:1] not in (" ", "\t"):
            inside = False
    return found


def _markdown_files(repo: Path, unreadable: list[str]) -> Iterator[Path]:
    """Every Markdown file of the repo but those under `NOT_SCANNED_FOLDERS` or in a folder named in
    `NOT_SCANNED_NAMES` at any depth; a folder that cannot be listed goes to `unreadable`."""

    def refused(error: OSError) -> None:
        unreadable.append(Path(error.filename).relative_to(repo).as_posix() if error.filename else str(error))

    for folder, dirs, files in os.walk(repo, onerror=refused):
        here = Path(folder).relative_to(repo).as_posix()
        dirs[:] = sorted(d for d in dirs if not d.endswith(".egg-info") and d not in NOT_SCANNED_NAMES
                         and (d if here == "." else f"{here}/{d}") not in NOT_SCANNED_FOLDERS)
        for name in sorted(files):
            if name.lower().endswith(".md"):
                yield Path(folder) / name


def held(repo: Path) -> set[str]:
    """The skills the repo holds: every folder of `.claude/skills/` with a `SKILL.md`."""

    root = repo / SKILLS_FOLDER
    return {p.name for p in root.iterdir() if (p / SKILL_FILE).is_file()} if root.is_dir() else set()


def check(repo: str | Path) -> list[str]:
    """One line per name the repo's texts use for a skill it does not hold, sorted by file and line."""

    repo = Path(repo)
    holds = held(repo)
    problems = []
    try:
        manifest = Manifest.load(repo)
    except SkillsError as e:  # said, not raised: the test names what is wrong
        problems.append(("", 0, f"{MANIFEST_NAME}: {e}"))
        manifest = None
    gone = manifest.gone if manifest else {}
    his_left_out = {name: entry["why"] for name, entry in (manifest.not_copied if manifest else {}).items()} | gone
    unreadable: list[str] = []
    for path in _markdown_files(repo, unreadable):
        rel = path.relative_to(repo).as_posix()
        try:
            text = path.read_bytes().decode("utf-8").removeprefix("\ufeff").replace("\r\n", "\n")
        except (OSError, UnicodeDecodeError):
            problems.append((rel, 0, f"{rel}: not UTF-8 text"))
            continue
        for number, line in enumerate(text.split("\n"), 1):
            for name in _NAMESPACED.findall(line):
                problems.append((rel, number, f"{rel}:{number}: the namespaced form `{NAMESPACE}:{name}`;"
                                              + (f" name the copy `{name}`" if name else " name the copy bare")))
            for name in _SLASH.findall(line):
                if name in his_left_out and name not in holds:
                    problems.append((rel, number, f"{rel}:{number}: names `/{name}`, his skill the repo does"
                                                  f" not copy ({his_left_out[name]})"))
        uses = [("calls", n, name) for n, name in _calls(text) if ":" not in name]
        uses += [("names", n, name) for n, line in enumerate(text.split("\n"), 1) for name in _NAMED_SKILL.findall(line)]
        if rel.startswith(AGENTS_FOLDER):
            uses += [("preloads", n, name) for n, name in _preloads(text)]
        for verb, number, name in uses:
            if name not in holds:
                why = f" ({gone[name]})" if name in gone else ""
                problems.append((rel, number, f"{rel}:{number}: {verb} `{name}`, {NOT_HELD}{why}"))
    problems += [(folder, 0, f"{folder}: a folder that cannot be read") for folder in unreadable]
    return [line for _, _, line in sorted(problems)]
