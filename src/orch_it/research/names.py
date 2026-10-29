"""The names the research bridge's text must not carry (#412 unit 5, 9): a provider or model name is
refused when the bridge's settings name it at run time, or when it is one of the known families
the caller passes in (the list lives in the test, not here).

`refused_names(settings_raw, known)` reads the `research` object of a settings dict, and takes only
the values of the provider and model NAME fields (never an alias, a host or another option):
- the parts of each `models` value and of each provider model id (`acme-llm/acme-1` gives `acme-llm`
  and `acme-1`);
- `provider.id` and the parts of `provider.npm`, whole;
- `provider.name`, whole, and each word of four characters or more;
- each `name` inside the dict of a `provider.models` entry, whole, and each model id of
  `provider.models`.
Names under three characters and the neutral words `research`, `model` and `provider` are dropped.

A pull request cannot set the installation's settings, so the required checks read the settings
file the work tree carries at its root (`settings.json`, the installation's own format and name):
`refused_names_of_tree(root, known)`. With no such file it is the known families alone.

`find_names(files, refuse)` returns (path, line number, name) for each whole-word,
case-insensitive hit; a line carrying the marker `_Avoid_` is skipped (a glossary's list of
words to avoid names them on purpose).

`bridge_texts(root)` reads the files the check covers: the bridge's `.mcp.json`, its MCP front,
the module docstring of its settings, the agent and skill files, every Markdown file under `docs/`
(not its `research` and `adr` records), `GLOSSARY.md` and `README.md`, and the `research`
subcommands' `help`, `description` and `epilog` strings in the command line. Those parts keep
their line numbers.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path

from orch_it import settings as installation
from orch_it.research.settings import ResearchSettings
from orch_it.settings import SettingsError

NEUTRAL = frozenset({"research", "model", "provider"})
MIN_LENGTH = 3  # a whole name
WORD_MIN_LENGTH = 4  # a word of a name
TREE_SETTINGS = "settings.json"  # the settings file a work tree carries at its root
MARKER = "_Avoid_"
WHOLE_FILES = (".mcp.json", "src/orch_it/research/mcp.py", "GLOSSARY.md", "README.md",
               ".claude/agents/*.md", ".claude/skills/**/SKILL.md", "docs/**/*.md")
RECORDS = ("docs/research/", "docs/adr/")
SETTINGS_DOCSTRING = "src/orch_it/research/settings.py"
COMMAND_LINE = "src/orch_it/cli.py"
SUBCOMMAND = "research"  # the subcommands named `research` or `research-*` are the bridge's
HELP_KEYS = ("help", "description", "epilog")


def refused_names(settings_raw: dict | None, known: Iterable[str] = ()) -> set[str]:
    """The lower-cased names to refuse: the given fixed families and the names the `research`
    object of `settings_raw` names; `settings_raw` None names none. A malformed `research` object
    raises `SettingsError`, as the bridge's own reading of it does."""

    names = {_clean(name) for name in known}
    if settings_raw is not None:
        research = ResearchSettings.from_settings(settings_raw)
        for model in research.models.values():
            names.update(_parts(model))
        provider = research.provider or {}
        for key in ("id", "npm"):
            if isinstance(provider.get(key), str):
                names.update(_parts(provider[key]))
        if isinstance(provider.get("name"), str):
            names.update(_parts(provider["name"]))
            names.update(_words(provider["name"]))
        for model_id, entry in (provider.get("models") or {}).items():
            names.update(_parts(model_id))
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.update(_parts(entry["name"]))
    return {name for name in names if len(name) >= MIN_LENGTH and name not in NEUTRAL}


def find_names(files: dict[str, str], refuse: set[str]) -> list[tuple[str, int, str]]:
    """(path, line number, name) for each refused name whole-word in each line, case-insensitive;
    the lines marked `_Avoid_` are skipped."""

    patterns = {name: re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE) for name in sorted(refuse)}
    found = []
    for path in sorted(files):
        for number, line in enumerate(files[path].splitlines(), 1):
            if MARKER in line:
                continue
            found.extend((path, number, name) for name, pattern in patterns.items() if pattern.search(line))
    return found


def refused_names_from_installation(known: Iterable[str] = (), work_tree: Path | None = None) -> set[str]:
    """`refused_names` of the installation's settings file, read through its path, together with the
    settings file of `work_tree` when one is given and carries it (the run-time entry point; the tests
    pass a dict instead)."""

    names = refused_names(installation.load(), known)
    if work_tree is not None:
        names |= refused_names_of_tree(work_tree)
    return names


def tree_settings(root: Path) -> dict | None:
    """The settings file a work tree carries at its root, parsed; None when it has none. A file that is
    no JSON object raises `SettingsError`, as the installation's does."""

    path = Path(root) / TREE_SETTINGS
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text("utf-8-sig"))
    except (OSError, ValueError) as e:
        raise SettingsError("settings_unreadable", f"{path}: {e}") from None
    if not isinstance(raw, dict):
        raise SettingsError("settings_unreadable", f"{path}: not an object")
    return raw


def refused_names_of_tree(root: Path, known: Iterable[str] = ()) -> set[str]:
    """`refused_names` of the settings file the work tree carries: what the required checks use."""

    return refused_names(tree_settings(root), known)


def bridge_texts(root: Path) -> dict[str, str]:
    """The text of each file the check covers, keyed by its path relative to `root` in posix form.
    Files that do not exist are skipped."""

    root = Path(root)
    texts: dict[str, str] = {}
    for pattern in WHOLE_FILES:
        for path in sorted(root.glob(pattern)):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and not rel.startswith(RECORDS):
                texts[rel] = path.read_text("utf-8")
    settings_py = root / SETTINGS_DOCSTRING
    if settings_py.is_file():
        texts[SETTINGS_DOCSTRING] = _docstring_text(settings_py)
    cli = root / COMMAND_LINE
    if cli.is_file():
        texts[COMMAND_LINE] = _help_text(cli)
    return texts


def _clean(name: str) -> str:
    return name.strip().lower()


def _parts(value: str) -> set[str]:
    """The names a value carries: its parts split at `/` and `@` (an npm scope), else the value whole."""

    return {part for part in (_clean(p) for p in re.split(r"[/@]", value)) if part}


def _words(value: str, separator: str = r"[^A-Za-z0-9]+") -> set[str]:
    """The words of a value of four characters or more that are no neutral word."""

    words = (_clean(word) for word in re.split(separator, value))
    return {word for word in words if len(word) >= WORD_MIN_LENGTH and word not in NEUTRAL}


def _docstring_text(path: Path) -> str:
    """The module docstring at its own line numbers, the rest of the file left out."""

    tree = ast.parse(path.read_text("utf-8"))
    doc = ast.get_docstring(tree, clean=False)
    if doc is None:
        return ""
    return "\n" * (tree.body[0].lineno - 1) + doc


def _help_text(path: Path) -> str:
    """The `help`, `description` and `epilog` strings of the `research` subcommands at their own
    line numbers: the command line's help text, the code left out."""

    source = path.read_text("utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    start, end = _research_span(calls)
    lines = [""] * max(source.count("\n") + 1, 1)
    for call in calls:
        if start <= call.lineno <= end:
            for keyword in call.keywords:
                if keyword.arg in HELP_KEYS:
                    text = " ".join(part.value for part in ast.walk(keyword.value)
                                    if isinstance(part, ast.Constant) and isinstance(part.value, str))
                    lines[keyword.value.lineno - 1] = (lines[keyword.value.lineno - 1] + " " + text).strip()
    return "\n".join(lines)


def _subcommand(call: ast.Call) -> str | None:
    """The name of the subcommand an `add_parser` call registers, else None."""

    if (isinstance(call.func, ast.Attribute) and call.func.attr == "add_parser" and call.args
            and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
        return call.args[0].value
    return None


def _research_span(calls: list[ast.Call]) -> tuple[int, int]:
    """The first line of the `research` subcommand and the last line of any `research-*` one."""

    named = [(_subcommand(call), call.lineno) for call in calls]
    bridge = [(name, line) for name, line in named if name and (name == SUBCOMMAND or name.startswith(SUBCOMMAND + "-"))]
    starts = [line for name, line in bridge if name == SUBCOMMAND]
    if not starts:
        raise LookupError(f"no subcommand {SUBCOMMAND!r} in the command line")
    return min(starts), max(line for _, line in bridge)
