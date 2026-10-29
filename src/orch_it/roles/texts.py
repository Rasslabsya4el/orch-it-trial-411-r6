"""Rules held as checks (#399 decision 4): `check(repo)` reads every text an agent of the layer works
from and returns one `Finding` per break of the kinds #382 counted in the old layer's 30 files.

The texts read are whatever exist of: `.claude/agents/*.md`, every Markdown file of a skill of ours
under `.claude/skills/` (a folder the manifest `pocock.json` does not name as his copy), every
`CLAUDE.md` of the repo, and every Markdown file under `docs/` but the research records of
`docs/research/` (they quote what they studied, the old layer's texts included, verbatim; no agent
works from them as rules). Nothing is listed by name, so a role file added later is read with no edit
here.

What is code in a text: an inline code span (`` `...` ``) and a line of a fenced block (closed by a
fence of its own kind, at least as long). A path is read from inline code and Markdown links only
(a fenced block is an example or a command, and a span of several words is a command, where only a
variable's value is read): a word that ends in a file extension or a `/`, the value of a variable
(`PYTHONPATH=src/x`), or a word whose first folder is one of the repo's own; a bare `name.md` must be
a file somewhere in the repo. A placeholder of the root (`<path>/src`, `<repo>/x`, `<worktree>/x`)
stands for the repo. Any other placeholder (`<n>`, `$X`, `*`, `~`), a URL or a domain makes a word
no path; a path of another repo is written `<repo name>:<path>` and is not read. A line suffix
(`:12`, `:12-14`, `:12:5`) and a test id (`::test_a`) are no part of the path. A section is named as
`path#anchor` (GitHub's heading anchor) or as `` `path` section "Title" `` (or `` `path` "Title" ``,
or `section "Title" of `path``).

Codes:
- `path_missing`: a path that is neither under the repo root nor beside the text;
- `section_missing`: a named section with no heading of that title or anchor in its file;
- `stub`: a pointer to a file, or a section, that holds no text but headings, or only a TODO,
  TBD or stub mark;
- `command_unknown`, `flag_unknown`: a `python -m orch_it ...` command (in code) whose command,
  value or option the command line does not have, or a `python -m orch_it.<module>` that is no
  module; a placeholder (`<n>`, `PATH`, `...`) stands for any value;
- `label_unknown`: a label named (a `gh issue`/`gh pr` label option, "the `x` label", "label `x`")
  that is not in the label set of `orch_it.labels`;
- `role_unknown`: a role named ("the `x` role", "role `x`") that is not in the roles settings;
- `part_removed`: a part the layer removed (`REMOVED`, the unblocker and the autopilot of #378),
  named in any form: a rule that hands work to it hands it to nobody;
- `skill_unknown`: a skill named that the repo holds neither as his copy nor as ours (#397's
  `orch_it.skills.references.check`, reused as it is: its line is parsed back into a finding);
- `rule_deferred`: a rule handed to a file that may not be there: a local file (`local/`,
  `*.local.*`), or a pointer said to hold "if it exists", "if present", "the lead's own part of
  ..." and the like;
- `rule_twice`: one rule in two places, of two files or of one: two normative sentences (with
  `must`, `never`, `only` ...; prose, not a table row) that share most of their content words, or
  one inside the other; or one term defined twice (a list item led by `**term**`, "`term` means").
  Two rules on one object worded apart are not seen; the ADRs of `docs/adr/` record why a rule is
  so and are left out of this one;
- `model_literal`, `effort_literal`: a model (`claude-...`, "Opus 4.8", `--model fable`,
  `model: sonnet`) or an effort (`--effort high`, "at high effort", "effort: high") written into a
  text; they are settings (decision 5), and only an agent file's frontmatter, which `roles render`
  writes, carries them;
- `text_unreadable`: a text that is not UTF-8.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

CODES = ("path_missing", "section_missing", "stub", "command_unknown", "flag_unknown", "label_unknown", "role_unknown",
         "part_removed", "skill_unknown", "rule_deferred", "rule_twice", "model_literal", "effort_literal",
         "text_unreadable")
AGENTS = ".claude/agents"
SKILLS = ".claude/skills"
RECORDS = "docs/adr/"
RESEARCH = "docs/research"  # records that quote other texts verbatim: never read as rules
# the parts #378 (resolution 6397583416) took out of the layer; a text naming one names nobody
REMOVED = {"unblocker": "#378", "autopilot": "#378"}
FILE_EXTENSIONS = ("md", "py", "json", "jsonl", "toml", "yml", "yaml", "sql", "txt", "sh", "ps1", "cfg", "ini", "xml")


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    code: str
    detail: str

    def __post_init__(self):
        if self.code not in CODES:
            raise ValueError(f"not a role-text code: {self.code}")

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code}: {self.detail}"


# ---- reading Markdown -------------------------------------------------------------------------

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")
_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*))?$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_STUB_MARK = re.compile(r"^\W*(todo|tbd|stub|to be written|coming soon)\W*$", re.IGNORECASE)


@dataclass(frozen=True)
class _Line:
    number: int
    text: str
    fenced: bool  # the whole line is code
    spans: tuple[str, ...]  # its inline code spans (none on a fenced line)

    def code(self) -> list[str]:
        return [self.text] if self.fenced else list(self.spans)


@dataclass(frozen=True)
class _Text:
    rel: str
    lines: tuple[_Line, ...]

    def finding(self, line: int, code: str, detail: str) -> Finding:
        return Finding(self.rel, line, code, detail)


def _fenced(lines: list[str]) -> list[bool | None]:
    """Per line: True inside a fenced block, None for a fence line itself, False outside. A fence
    closes only at a fence of its own character at least as long (so a ```` block may hold ```)."""

    out: list[bool | None] = []
    fence = None
    for line in lines:
        opened = _FENCE.match(line)
        if fence is None and opened:
            fence = opened.group(1)
            out.append(None)
        elif fence is not None:
            closing = _FENCE.match(line)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence) \
                    and not line.strip()[len(closing.group(1)):].strip():
                fence = None
                out.append(None)
            else:
                out.append(True)
        else:
            out.append(False)
    return out


def _lines(text: str) -> tuple[_Line, ...]:
    raw = text.split("\n")
    out = []
    for number, (line, inside) in enumerate(zip(raw, _fenced(raw)), 1):
        if inside is None:
            continue
        spans = () if inside else tuple(m.group(2).strip() for m in _SPAN.finditer(line))
        out.append(_Line(number, line, bool(inside), spans))
    return tuple(out)


def _plain(title: str) -> str:
    """A heading's text as GitHub shows it: links as their text, no emphasis or code marks."""

    title = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", title)
    return re.sub(r"[`*]|(?<!\w)_|_(?!\w)", "", title).strip()


def slug(title: str) -> str:
    """GitHub's anchor of a heading: its plain text in lower case, punctuation but `-` and `_`
    gone, each space a hyphen."""

    return re.sub(r" ", "-", re.sub(r"[^\w\- ]", "", _plain(title).lower()))


@dataclass(frozen=True)
class _Section:
    title: str
    anchor: str
    empty: bool


def _sections(text: str) -> tuple[list[_Section], bool]:
    """(the headings of a Markdown text, each with whether it has no text before the next heading
    of its level or above; whether the whole text is a stub)."""

    lines = text.removeprefix("﻿").replace("\r\n", "\n").split("\n")
    inside = _fenced(lines)
    heads: list[tuple[int, str, int]] = []
    body: list[int] = []
    for number, line in enumerate(lines):
        if inside[number] is not False:
            body.append(number)
            continue
        atx = _ATX.match(line)
        setext = _SETEXT.match(line) if number else None
        if atx:
            heads.append((len(atx.group(1)), re.sub(r"[ \t]+#+[ \t]*$", "", (atx.group(2) or "").strip()), number))
        elif setext and lines[number - 1].strip() and inside[number - 1] is False \
                and not _ATX.match(lines[number - 1]) and (number - 1) in body:
            body.remove(number - 1)  # `Title` over `===` or `---` is a heading, not text
            heads.append((1 if setext.group(1)[0] == "=" else 2, lines[number - 1].strip(), number - 1))
        elif line.strip() and not _STUB_MARK.match(line) and line.strip() != "---":
            body.append(number)
    heads.sort(key=lambda h: h[2])
    sections, seen = [], {}
    for i, (level, title, at) in enumerate(heads):
        end = next((h[2] for h in heads[i + 1:] if h[0] <= level), len(lines))
        anchor = slug(title)
        if anchor in seen:  # GitHub numbers a repeated anchor: `x`, `x-1`, ...
            seen[anchor] += 1
            anchor = f"{anchor}-{seen[anchor]}"
        else:
            seen[anchor] = 0
        sections.append(_Section(_plain(title), anchor, not any(at < n < end for n in body)))
    return sections, not body


def _ours(repo: Path) -> list[Path]:
    """The skill folders of ours: every folder of `.claude/skills/` the manifest does not name as
    his copy (all of them when the manifest cannot be read; #397's check reports it)."""

    from orch_it.skills.copies import Manifest, SkillsError

    root = repo / SKILLS
    if not root.is_dir():
        return []
    try:
        manifest = Manifest.load(repo)
        his = set(manifest.skills) if manifest else set()
    except SkillsError:
        his = set()
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name not in his)


def text_files(repo: str | Path) -> list[Path]:
    """Every text the check reads, sorted."""

    repo = Path(repo)
    found = list((repo / AGENTS).glob("*.md")) if (repo / AGENTS).is_dir() else []
    for folder in [*_ours(repo), repo / "docs"]:
        if folder.is_dir():
            found += [Path(dirpath) / name for dirpath, _, names in os.walk(folder)
                      for name in names if name.lower().endswith(".md")
                      and not f"{Path(dirpath).relative_to(repo).as_posix()}/".startswith(f"{RESEARCH}/")]
    for dirpath, dirs, names in os.walk(repo):  # every CLAUDE.md: Claude Code reads the one of a folder it works in
        rel = Path(dirpath).relative_to(repo).as_posix()
        dirs[:] = [d for d in dirs if not _skipped(d if rel == "." else f"{rel}/{d}")]
        found += [Path(dirpath) / "CLAUDE.md"] if "CLAUDE.md" in names else []
    return sorted(p for p in found if p.is_file())


def _skipped(rel: str) -> bool:
    """A folder no text of the layer lives in: the other worktrees, the test fixtures (the role-text
    tests' own broken texts), environments and caches."""

    parts = set(rel.split("/"))
    return (bool({".git", ".venv", "venv", "node_modules", "__pycache__"} & parts)
            or rel.startswith((".claude/worktrees", "tests/fixtures")))


# ---- paths and sections -----------------------------------------------------------------------

_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\s*\)")
_REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)")
_TITLED = re.compile(r"`([^`\s]+)`,?\s+(?:section\s+)?[\"“]([^\"”]+)[\"”]", re.IGNORECASE)
_SECTION_OF = re.compile(r"section\s+[\"“]([^\"”]+)[\"”]\s+of\s+`([^`\s]+)`", re.IGNORECASE)
_SUFFIX = re.compile(r"(?:::[\w\[\]-]+)+$|(?::\d+(?:-\d+)?)+(?:,\d+(?:-\d+)?)*$")
_ROOT = re.compile(r"^<(?:path|repo|root|worktree|work tree|work-tree)>/", re.IGNORECASE)
_NO_PATH = re.compile(r"[<>*?$%{}~@\\|]|://|^[A-Za-z]:|^-|^/")
_EXTENSION = re.compile(r"\.(" + "|".join(FILE_EXTENSIONS) + r")$")
_DOMAIN = re.compile(r"^(?:[a-z0-9-]+\.)+(?:com|org|net|io|dev|ai|app)(?:/|$)", re.IGNORECASE)
_QUOTES = "\"'()[]"
_TRAILING = ",;.!:"
_ASSIGNED = re.compile(r"[A-Z][A-Z0-9_]*=(.+)")


class _Repo:
    """What the texts are judged against: the repo's files."""

    def __init__(self, repo: Path):
        self.root = repo
        self.top = {p.name for p in repo.iterdir()}
        self.md_files: dict[str, list[Path]] = {}
        for dirpath, dirs, names in os.walk(repo):
            rel = Path(dirpath).relative_to(repo).as_posix()
            dirs[:] = [d for d in dirs if not _skipped(f"{rel}/{d}" if rel != "." else d)]
            for name in names:
                if name.endswith(".md"):
                    self.md_files.setdefault(name, []).append(Path(dirpath) / name)

    def locate(self, path: str, beside: str) -> Path | None:
        if "/" not in path.rstrip("/") and path.endswith(".md"):
            if (self.root / beside / path).is_file():
                return self.root / beside / path
            found = self.md_files.get(path)
            return found[0] if found else None
        for base in (self.root, self.root / beside):
            target = base / path
            if target.exists() and (not path.endswith("/") or target.is_dir()):
                return target
        return None


@cache
def _read_sections(path: Path) -> tuple[list[_Section], bool]:
    try:
        return _sections(path.read_bytes().decode("utf-8", "replace"))
    except OSError:
        return [], False


def _path_of(word: str, repo: _Repo) -> tuple[str, str | None] | None:
    """(the path `word` names, its anchor), or None when it is no path of this repo."""

    word = word.strip(_QUOTES).rstrip(_TRAILING)
    assigned = _ASSIGNED.match(word)
    if assigned:  # `PYTHONPATH=src/x`: the value of a variable is a path
        word = assigned.group(1).strip(_QUOTES).rstrip(_TRAILING)
    rooted = bool(_ROOT.match(word))
    word = _ROOT.sub("", word)  # `<path>/src`: the repo's own `src`
    word, _, anchor = word.partition("#")
    word = _SUFFIX.sub("", word)
    if not word or _NO_PATH.search(word) or ":" in word or "=" in word or _DOMAIN.match(word):
        return None
    if rooted:
        return word, anchor or None
    if "/" not in word.rstrip("/"):
        return (word, anchor or None) if word.endswith(".md") else None
    if assigned or _EXTENSION.search(word) or word.endswith("/") or word.split("/", 1)[0] in repo.top:
        return word, anchor or None
    return None


def _pointers(line: _Line) -> list[tuple[str, str | None]]:
    """(word, section title or None) of every pointer of a prose line."""

    found = [(word, None) for span in line.spans for word in span.split()
             if len(span.split()) == 1 or _ASSIGNED.match(word)]  # a command's words are no pointers
    targets = _LINK.findall(line.text) + _REFERENCE.findall(line.text)
    found += [(t, None) for t in targets if not re.match(r"[a-z][a-z0-9+.-]*:|#", t, re.IGNORECASE)]
    found += [(path, title) for path, title in _TITLED.findall(line.text)]
    found += [(path, title) for title, path in _SECTION_OF.findall(line.text)]
    return found


def _paths(text: _Text, repo: _Repo) -> list[Finding]:
    beside = str(Path(text.rel).parent)
    found = []
    for line in text.lines:
        if line.fenced:  # a fenced block is an example or a command, never a pointer
            continue
        for word, title in _pointers(line):
            named = _path_of(word, repo)
            if named is None:
                continue
            path, anchor = named
            target = repo.locate(path, beside)
            if target is None:
                found.append(text.finding(line.number, "path_missing", f"`{path}` does not exist"))
                continue
            if not target.is_file() or not target.name.endswith(".md"):
                continue
            sections, stub = _read_sections(target)
            if anchor is None and title is None:
                if stub:
                    found.append(text.finding(line.number, "stub", f"`{path}` holds no text"))
                continue
            section = next((s for s in sections if (anchor and s.anchor == anchor.lower())
                            or (title and s.title.lower() == title.strip().lower())), None)
            name = f"#{anchor}" if anchor else f'"{title}"'
            if section is None:
                found.append(text.finding(line.number, "section_missing", f"`{path}` has no section {name}"))
            elif section.empty:
                found.append(text.finding(line.number, "stub", f"`{path}` section {name} holds no text"))
    return found


# ---- commands ---------------------------------------------------------------------------------

_COMMAND = re.compile(r"(?:^|(?<=[\s`(]))(?:python[0-9.]*|py)(?:\.exe)?((?:\s+-(?!m\b)\w\S*(?:\s+(?!-)[^\s-]\S*)?)*)"
                      r"\s+-m\s+(orch_it(?:\.[A-Za-z_][\w.]*)?)(?![\w.])(.*)")
_SEPARATOR = re.compile(r"\s(?:\||&&|\|\||;|>|2>|<)\s|`")
_PLACEHOLDER = re.compile(r"[<{$]|\.\.\.|^[A-Z][A-Z0-9_.=-]*$|^#")


def _placeholder(word: str) -> bool:
    return bool(_PLACEHOLDER.search(word))


@cache
def _parser() -> argparse.ArgumentParser:
    from orch_it import cli

    return cli.parser()


def _option(parser: argparse.ArgumentParser, name: str) -> argparse.Action | None:
    """The action of option `name`, or of the one long option it is a prefix of (argparse takes
    an unambiguous prefix). Reads argparse's own tables: the only place this module does."""

    actions = parser._option_string_actions  # noqa: SLF001 - argparse keeps no public table
    if name in actions:
        return actions[name]
    longer = {actions[o] for o in actions if name.startswith("--") and o.startswith(name)}
    return longer.pop() if len(longer) == 1 else None


def _judge_command(module: str, rest: str) -> tuple[str, str] | None:
    """(code, detail) of why `python -m <module> <rest>` is no command of orch-it, or None."""

    if module != "orch_it":
        try:
            known = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            known = False
        return None if known else ("command_unknown", f"no module `{module}`")
    try:
        words = shlex.split(rest)
    except ValueError:
        words = rest.split()
    words = [w.strip("[]()") for w in words]
    words = [w for w in words if w and w not in ("|", "\\")]
    parser, said = _parser(), "python -m orch_it"
    positionals = [a for a in parser._actions if not a.option_strings]  # noqa: SLF001
    i = 0
    while i < len(words):
        word = words[i]
        if word.startswith("-") and len(word) > 1 and not _placeholder(word):
            name, equals, value = word.partition("=")
            action = _option(parser, name)
            if action is None:
                return "flag_unknown", f"`{said}`: no option `{name}`"
            if action.nargs == 0:
                i += 1
                continue
            value = value if equals else (words[i + 1] if i + 1 < len(words) else None)
            if value is not None and action.choices and not _placeholder(value) and value not in action.choices:
                return "command_unknown", f"`{said}`: `{value}` is not one of its `{name}` values"
            i += 1 if equals else 2
            continue
        if not positionals:
            return None  # more words than the command takes: prose after it
        action = positionals[0]
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            if _placeholder(word):
                return None  # `events <command>`: any command of the group
            if word not in action.choices:
                return "command_unknown", f"`{said}`: no command `{word}`"
            parser, said = action.choices[word], f"{said} {word}"
            positionals = [a for a in parser._actions if not a.option_strings]  # noqa: SLF001
        else:
            if action.choices and not _placeholder(word) and word not in action.choices:
                return "command_unknown", f"`{said}`: `{word}` is not one of its `{action.dest}` values"
            if action.nargs not in ("*", "+", argparse.REMAINDER):
                positionals = positionals[1:]
        i += 1
    return None


def _code_pieces(text: _Text) -> list[tuple[int, str]]:
    """(line, code) of every span and fenced line, a fenced line ending in `\\` joined to the next."""

    pieces, carry = [], None
    for line in text.lines:
        for code in line.code():
            if carry is not None and line.fenced:
                code = f"{carry[1]} {code}"
                number = carry[0]
            else:
                number = line.number
            carry = None
            if line.fenced and code.rstrip().endswith("\\"):
                carry = (number, code.rstrip()[:-1])
                continue
            pieces.append((number, code))
    if carry:
        pieces.append(carry)
    return pieces


def _commands(text: _Text) -> list[Finding]:
    found = []
    for number, code in _code_pieces(text):
        for part in _SEPARATOR.split(f" {code} "):
            for match in _COMMAND.finditer(part):
                why = _judge_command(match.group(2), match.group(3))
                if why:
                    found.append(text.finding(number, *why))
    return found


# ---- labels, roles, removed parts -------------------------------------------------------------

_LABEL_OPTIONS = ("--add-label", "--remove-label", "--label", "-l")
_GH_ISSUE = re.compile(r"\bgh\s+(?:issue|pr)\b")
_NAMED_LABEL = re.compile(r"`([^`\s]+(?: [^`\s]+)?)`\s+label\b|\blabel\s+`([^`\s]+(?: [^`\s]+)?)`", re.IGNORECASE)
_LISTED_LABELS = re.compile(r"((?:`[^`\s]+`\s*(?:,|and|or)\s*)+`[^`\s]+`)\s+labels\b", re.IGNORECASE)
_NAMED_ROLE = re.compile(r"\bthe\s+`([^`\s]+)`\s+role\b|\brole\s+`([^`\s]+)`", re.IGNORECASE)
_REMOVED = re.compile(r"\b(" + "|".join(REMOVED) + r")s?\b", re.IGNORECASE)


@cache
def _labels() -> frozenset[str]:
    from orch_it import labels

    return frozenset(labels.label_set())


@cache
def _roles() -> frozenset[str]:
    from orch_it.roles import RoleSet

    return frozenset(RoleSet.load().roles)


def _label_options(code: str) -> list[str]:
    """The labels a `gh issue` / `gh pr` command's label options name, read from its shell words."""

    try:
        words = shlex.split(code)
    except ValueError:
        words = code.split()
    found = []
    for i, word in enumerate(words):
        name, equals, value = word.partition("=")
        if name in _LABEL_OPTIONS and (equals or i + 1 < len(words)):
            found += [v.strip() for v in (value if equals else words[i + 1]).split(",") if v.strip()]
    return found


def _names(text: _Text) -> list[Finding]:
    """Labels a text names against the label set, roles against the roles settings, and any part
    the layer removed."""

    found = []
    records = text.rel.startswith(RECORDS)
    for line in text.lines:
        labels = [name for code in line.code() if _GH_ISSUE.search(code) for name in _label_options(code)]
        roles: list[str] = []
        if not line.fenced:
            labels += [a or b for a, b in _NAMED_LABEL.findall(line.text)]
            labels += [name for listed in _LISTED_LABELS.findall(line.text)
                       for name in re.findall(r"`([^`\s]+)`", listed)]
            roles += [a or b for a, b in _NAMED_ROLE.findall(line.text)]
        for name in dict.fromkeys(labels):
            if name not in _labels() and not _placeholder(name):
                found.append(text.finding(line.number, "label_unknown", f"`{name}` is no label of the label set"))
        for name in dict.fromkeys(roles):
            if name not in _roles() and not _placeholder(name):
                found.append(text.finding(line.number, "role_unknown", f"`{name}` is no role of the roles settings"))
        if not records:
            for name in dict.fromkeys(m.lower() for m in _REMOVED.findall(line.text)):
                why = f"`{name}`: removed from the layer ({REMOVED[name]}); name what does its job now"
                found.append(text.finding(line.number, "part_removed", why))
    return found


# ---- literals ---------------------------------------------------------------------------------

_ALIASES = r"(?:opus|sonnet|haiku|fable|mythos)"
_MODEL = re.compile(r"\bclaude-(?:[\d.-]+-)?" + _ALIASES + r"(?:-\d[\w.-]*)?\b"
                    r"|\b(?i:opus|sonnet|haiku|fable|mythos)\s+\d+(?:\.\d+)?\b(?!\s+\d)"
                    r"|--model[ =]" + _ALIASES + r"\b|\bmodel:\s*" + _ALIASES + r"\b", re.IGNORECASE)
_LEVELS = r"(?:low|medium|high|x-?high|max)"
_EFFORT = re.compile(r"--effort[ =]" + _LEVELS + r"\b|\beffort(?:\s+level)?\s*[:=]?\s*(?:of\s+|at\s+|is\s+)?" + _LEVELS
                     + r"\b|\b(?:at|on|with)\s+" + _LEVELS + r"\s+effort\b", re.IGNORECASE)


def _frontmatter_end(text: _Text) -> int:
    """The last line of an agent file's frontmatter (rendered from the settings), else 0."""

    if not text.rel.startswith(AGENTS + "/") or not text.lines or text.lines[0].text.strip() != "---":
        return 0
    return next((line.number for line in text.lines[1:] if line.text.strip() == "---"), 0)


def _literals(text: _Text) -> list[Finding]:
    """A model or effort written into a text: they are settings (decision 5), read from there."""

    found = []
    skip = _frontmatter_end(text)
    for line in text.lines:
        if line.number <= skip:
            continue
        for match in _MODEL.finditer(line.text):
            found.append(text.finding(line.number, "model_literal",
                                      f"`{match.group(0)}`: a role's model is a setting of the roles settings"))
        for match in _EFFORT.finditer(line.text):
            found.append(text.finding(line.number, "effort_literal",
                                      f"`{match.group(0)}`: a role's effort is a setting of the roles settings"))
    return found


# ---- rules deferred ---------------------------------------------------------------------------

_LOCAL_PATH = re.compile(r"(?:^|/)local(?:/|$)|\.local\.")
_OPTIONAL = re.compile(r"\b(?:if (?:it|one|that file|the file|such a file|they) (?:exists?|is there|is present)"
                       r"|if present|when present|where present|if any|optional (?:file|doc|notes|part)"
                       r"|(?:\w+'s|its|their) own (?:part|notes|section|file) of)\b", re.IGNORECASE)


def _deferred(text: _Text, repo: _Repo) -> list[Finding]:
    """A rule handed to a file that may not be there: a local file (`local/`, `*.local.*`), or a
    pointer said to hold only if the file exists or in someone's own part of it. A rule is written
    once, in a file every machine has."""

    found = []
    for line in text.lines:
        if line.fenced:
            continue
        pointed = [named[0] for word, _ in _pointers(line) if (named := _path_of(word, repo))]
        local = [path for path in pointed if _LOCAL_PATH.search(path)]
        optional = _OPTIONAL.search(line.text)
        if local:
            found.append(text.finding(line.number, "rule_deferred",
                                      f"`{local[0]}` is a local file; write the rule where every machine has it"))
        elif optional and (pointed or re.search(r"\b[\w.-]+\.md\b", line.text)):
            found.append(text.finding(line.number, "rule_deferred",
                                      f"a rule left to a file that may not be there ({optional.group(0)!r})"))
    return found


# ---- skills -----------------------------------------------------------------------------------

def _skills(repo: Path) -> list[Finding]:
    """#397's check, reused as it is: each of its `path:line: text` lines becomes a finding."""

    from orch_it.skills.references import check as skill_check

    found = []
    for line in skill_check(repo):
        where, _, detail = line.partition(": ")
        path, _, number = where.partition(":")
        found.append(Finding(path, int(number) if number.isdigit() else 0, "skill_unknown", detail or line))
    return found


# ---- one rule in one file ---------------------------------------------------------------------

_NORMATIVE = re.compile(r"\b(?:must|never|always|only|do not|don't|may not|cannot|is refused|are refused|refuses)\b",
                        re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z`\"(*_])")
_WORD = re.compile(r"[a-z0-9][a-z0-9_]+")
_DEFINED = re.compile(r"^\*\*([^*]{0,59}[^*.:])\*\*|^`([^`]{1,40})`\s+means\b")  # `**Step.**` is a label, no term
_STOP = frozenset("""a an the and or but nor of to in on at by for with from into onto as is are was were be been
being it its this that these those there here which who whom whose what when where why how than then so if
else not no any each every all both either neither one two their his her our your my we you they he she
i me us them do does did done has have had having can could would should will shall may might must never
always only also just very more most less least such same other another own per via about after before
between over under up down out off again once while until because though although yet still even""".split())
_GENERIC = frozenset({"ticket", "issue", "label", "pull", "request", "file"})  # a defined term without them
RULE_WORDS_MIN = 5  # a shorter sentence is too general to be one rule said twice
RULE_LIKENESS = 0.6  # the share of all their content words two sentences hold in common
RULE_INSIDE = 0.7  # the share of the smaller sentence's words the larger one holds
RULE_INSIDE_MIN = 6  # one sentence inside another counts only from this many words
COMMON_WORD_RULES = 200  # a word in more rules than this finds no candidate pair on its own


def _stem(word: str) -> str:
    for ending in ("ing", "ed", "es", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            word = word[: -len(ending)]
            break
    return word.removesuffix("e") if len(word) > 4 else word


@dataclass(frozen=True)
class _Rule:
    rel: str
    line: int
    words: frozenset[str]
    term: bool = False  # a defined term, not a sentence


def _rules(text: _Text) -> list[_Rule]:
    """The normative sentences and defined terms of a text's prose (not fenced, not a heading or a
    table row), each with the line it starts on and its content words."""

    rules: list[_Rule] = []
    paragraph: list[_Line] = []
    skip = _frontmatter_end(text)

    def words_of(sentence: str) -> frozenset[str]:
        return frozenset(_stem(w) for w in _WORD.findall(sentence.lower().replace("`", "")) if w not in _STOP)

    def flush() -> None:
        if not paragraph:
            return
        joined, starts = "", []
        for line in paragraph:
            starts.append((len(joined), line.number))
            joined += re.sub(r"^\s*(?:[-*>]|\d+\.)\s+", "", line.text).strip() + " "
        defined = _DEFINED.match(joined)
        if defined:
            term = words_of(defined.group(1) or defined.group(2)) - _GENERIC
            if term:
                rules.append(_Rule(text.rel, paragraph[0].number, frozenset(term), term=True))
        at = 0
        for sentence in _SENTENCE_END.split(joined):
            begin = joined.find(sentence, at)
            at = begin + len(sentence)
            if _NORMATIVE.search(sentence) and len(words := words_of(sentence)) >= RULE_WORDS_MIN:
                rules.append(_Rule(text.rel, max(n for offset, n in starts if offset <= begin), words))
        paragraph.clear()

    for line in text.lines:
        stripped = line.text.strip()
        if line.number <= skip or line.fenced or not stripped or _ATX.match(line.text) or stripped.startswith("|"):
            flush()
            continue
        if re.match(r"^\s*(?:[-*]|\d+\.)\s", line.text):
            flush()
        paragraph.append(line)
    flush()
    return rules


def _alike(first: _Rule, second: _Rule) -> bool:
    if first.term or second.term:
        return first.term and second.term and first.words == second.words
    common = len(first.words & second.words)
    small = min(len(first.words), len(second.words))
    return (common / len(first.words | second.words) >= RULE_LIKENESS
            or (small >= RULE_INSIDE_MIN and common / small >= RULE_INSIDE))


def _twice(rules: list[_Rule]) -> list[Finding]:
    """One finding per rule said again: at its later place, naming the first place that says it.
    Candidate pairs come from the words two rules share, so the compare stays near linear."""

    found = []
    ordered = sorted(rules, key=lambda r: (r.rel, r.line))
    postings: dict[str, list[int]] = defaultdict(list)
    first_of: dict[tuple[bool, frozenset[str]], int] = {}
    for index, rule in enumerate(ordered):
        same = first_of.get((rule.term, rule.words))
        if same is not None:
            match = ordered[same]
        else:
            counts: dict[int, int] = defaultdict(int)
            for word in rule.words:
                if len(postings[word]) <= COMMON_WORD_RULES:
                    for earlier in postings[word]:
                        counts[earlier] += 1
            need = 1 if rule.term else 2
            match = next((ordered[e] for e in sorted(counts) if counts[e] >= need and _alike(ordered[e], rule)), None)
            first_of[(rule.term, rule.words)] = index
        for word in rule.words:
            postings[word].append(index)
        if match is not None:
            what = "the term defined" if rule.term else "the rule"
            found.append(Finding(rule.rel, rule.line, "rule_twice",
                                 f"{what} at {match.rel}:{match.line}, said again; write it once and link to it"))
    return found


# ---- the check --------------------------------------------------------------------------------

def check(repo: str | Path) -> list[Finding]:
    """Every finding of the repo's texts, sorted by file and line, each once."""

    repo = Path(repo)
    judge = _Repo(repo)
    _read_sections.cache_clear()
    found: set[Finding] = set(_skills(repo))
    rules: list[_Rule] = []
    for path in text_files(repo):
        rel = path.relative_to(repo).as_posix()
        try:
            content = path.read_bytes().decode("utf-8").removeprefix("﻿").replace("\r\n", "\n")
        except (OSError, UnicodeDecodeError):
            found.add(Finding(rel, 0, "text_unreadable", "not UTF-8 text"))
            continue
        text = _Text(rel, _lines(content))
        for judged in (_paths(text, judge), _commands(text), _names(text), _literals(text), _deferred(text, judge)):
            found.update(judged)
        if not rel.startswith(RECORDS):  # an ADR records why a rule is so: it says the rule again by design
            rules += _rules(text)
    found.update(_twice(rules))
    return sorted(found)
