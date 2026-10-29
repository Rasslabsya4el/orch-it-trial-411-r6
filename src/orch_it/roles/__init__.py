"""The roles of the layer and their settings (#399 decision 5).

`roles.json` beside this module is the roles settings: per role its `model` and `effort`, and the
`fallback` model a launcher takes when the model's safeguard pauses a session (the brain's answer
on #399: a fallback per role, Opus 4.8 for the Opus roles). Like the label set (`orch_it.labels`)
it ships with orch-it, so every repo gets the same roles; a test or a person passes another file
with `RoleSet.load(path)` or `roles render --settings`.

A model is a full model id (`claude-...`), never an alias such as `sonnet`, so its price row can
be checked (`orch_it.events.pricing.missing_rates`). `effort` is one of Claude Code's levels, or
absent for the model's own default.

`render` writes each agent file's `model:` and `effort:` frontmatter lines from the settings and
leaves every other byte of the file as it was; `drift` names each agent file whose frontmatter
differs from the settings, or that names no role of them. A role with no agent file yet (the
brain's, until the owner merges it) is no drift.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from orch_it.events import pricing

ROLES_FILE = Path(__file__).with_name("roles.json")
AGENTS_FOLDER = Path(".claude") / "agents"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
ROLE_KEYS = ("model", "effort", "fallback", "inputs", "report")
REPORT_KEYS = ("cap_lines", "owes", "context_line")
# each report line a role may owe, and the skill a line of it claims (#399 decision 6)
SKILL_OF_LINE = {"tdd": "tdd", "diagnosis": "diagnosing-bugs", "code-review": "code-review", "research": "research",
                 "writing-for-agents": "writing-for-agents", "domain-modeling": "domain-modeling",
                 "codebase-design": "codebase-design"}
OWED_LINES = tuple(SKILL_OF_LINE)
# what a spawn fills for a role's subagent (#399 decision 3); a role with none is a session, never spawned
INPUTS = ("issue", "unit", "diff", "worktree", "scratchpad")
ERROR_CODES = ("settings_unreadable", "bad_setting", "frontmatter_missing", "agent_unreadable", "agent_unwritable",
               "work_tree_missing")
_ROLE = re.compile(r"[a-z][a-z0-9-]{0,63}")
_MODEL = re.compile(r"claude-[a-z0-9][a-z0-9.-]{0,63}")
_FRONT_LINE = {"model": re.compile(r"model[ \t]*:"), "effort": re.compile(r"effort[ \t]*:")}


class RolesError(ValueError):
    """`code`: `settings_unreadable` (no JSON object), `bad_setting` (`field` names the part, such
    as `roles.worker.model`), `frontmatter_missing` and `agent_unreadable` (an agent file of a role
    that `render` cannot write: no frontmatter, or not UTF-8 text; `field` is its path)."""

    def __init__(self, code: str, field: str = ""):
        if code not in ERROR_CODES:
            raise ValueError(f"not a roles error code: {code}")
        self.code = code
        self.field = field
        super().__init__(f"{code}: {field}" if field else code)


@dataclass(frozen=True)
class Report:
    """What a role's final report is held to at its stop (#399 decision 6): at most `cap_lines`
    lines, the lines it `owes`, and the `context_line` in tokens past which it must name its
    table file (survey's 100k hand-back)."""

    cap_lines: int | None = None
    owes: tuple[str, ...] = ()
    context_line: int | None = None


@dataclass(frozen=True)
class Role:
    name: str
    model: str
    effort: str | None
    fallback: str
    inputs: tuple[str, ...] = ()
    report: Report | None = None  # None: the stop hold reads nothing of this role


@dataclass(frozen=True)
class RoleSet:
    roles: dict[str, Role]

    @classmethod
    def load(cls, path: str | Path | None = None) -> RoleSet:
        path = Path(path) if path is not None else ROLES_FILE
        try:
            raw = json.loads(path.read_bytes().decode("utf-8-sig"), object_pairs_hook=_Object)
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as e:
            raise RolesError("settings_unreadable", f"{path}: {type(e).__name__}: {e}"[:300]) from None
        if not isinstance(raw, dict):
            raise RolesError("settings_unreadable", f"{path}: not an object")
        roles = raw.get("roles")
        if not isinstance(roles, dict) or not roles or "roles" in raw.repeated:
            raise RolesError("bad_setting", "roles")
        if roles.repeated:
            raise RolesError("bad_setting", f"roles.{sorted(roles.repeated)[0]}")
        return cls({name: _role(name, value) for name, value in roles.items()})

    def unpriced(self) -> list[str]:
        """One line per model the settings name, primary or fallback, that the price table
        cannot price every turn of (the brain's answer on #399), sorted."""

        found = []
        for role in self.roles.values():
            for key, model in (("model", role.model), ("fallback", role.fallback)):
                missing = pricing.missing_rates(model)
                if missing == ["model"]:
                    found.append(f"roles.{role.name}.{key}: {model} has no row in the price table")
                elif missing:
                    found.append(f"roles.{role.name}.{key}: {model} has no rate for {', '.join(missing)}")
        return sorted(found)


class _Object(dict):
    """A JSON object that remembers the keys it was given more than once (the last one wins in
    plain JSON, so a repeated role would silently drop the first)."""

    def __init__(self, pairs: list[tuple[str, object]]):
        super().__init__(pairs)
        keys = [key for key, _ in pairs]
        self.repeated = {key for key in keys if keys.count(key) > 1}


def _role(name: object, value: object) -> Role:
    where = f"roles.{name}"
    if not isinstance(name, str) or not _ROLE.fullmatch(name) or not isinstance(value, dict):
        raise RolesError("bad_setting", where)
    if value.repeated:
        raise RolesError("bad_setting", f"{where}.{sorted(value.repeated)[0]}")
    for key in value:
        if key not in ROLE_KEYS:
            raise RolesError("bad_setting", f"{where}.{key}")
    model, effort, fallback = value.get("model"), value.get("effort"), value.get("fallback")
    if not isinstance(model, str) or not _MODEL.fullmatch(model):
        raise RolesError("bad_setting", f"{where}.model")
    if effort is not None and effort not in EFFORTS:
        raise RolesError("bad_setting", f"{where}.effort")
    if not isinstance(fallback, str) or not _MODEL.fullmatch(fallback) or fallback == model:
        raise RolesError("bad_setting", f"{where}.fallback")
    inputs = value.get("inputs", [])
    if (not isinstance(inputs, list) or any(item not in INPUTS for item in inputs)
            or len(set(inputs)) != len(inputs)):
        raise RolesError("bad_setting", f"{where}.inputs")
    return Role(name, model, effort, fallback, tuple(inputs), _report(where, value.get("report")))


def _report(where: str, value: object) -> Report | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.repeated:
        raise RolesError("bad_setting", f"{where}.report")
    for key in value:
        if key not in REPORT_KEYS:
            raise RolesError("bad_setting", f"{where}.report.{key}")
    for key in ("cap_lines", "context_line"):
        number = value.get(key)
        if number is not None and (type(number) is not int or number < 1):
            raise RolesError("bad_setting", f"{where}.report.{key}")
    owes = value.get("owes", [])
    if not isinstance(owes, list) or any(line not in OWED_LINES for line in owes) or len(set(owes)) != len(owes):
        raise RolesError("bad_setting", f"{where}.report.owes")
    return Report(value.get("cap_lines"), tuple(owes), value.get("context_line"))


def _split(text: str) -> tuple[list[str], list[str]] | None:
    """(frontmatter lines, the rest) of an agent file, or None when it opens with no
    frontmatter closed by `---`."""

    lines = text.split("\n")
    if not lines or lines[0].removeprefix("﻿").rstrip("\r") != "---":
        return None
    for end in range(1, len(lines)):
        if lines[end].rstrip("\r") == "---":
            return lines[1:end], lines[end:]
    return None


def _line_value(line: str) -> str | None:
    """The value of a `key: value` line, its quotes and a trailing comment aside."""

    return line.split(":", 1)[1].split("#", 1)[0].strip().strip("\"'") or None


def _values(front: list[str], key: str) -> list[str | None]:
    return [_line_value(line) for line in front if _FRONT_LINE[key].match(line)]


def _rendered(front: list[str], role: Role, eol: str) -> list[str]:
    """The frontmatter with its `model:` and `effort:` lines as the settings say, each in the
    place it had (a new one after `model:`, else after `name:`, else at the end); `eol` is the
    file's `\\r` or nothing, so a CRLF file stays CRLF."""

    wanted = {"model": role.model, "effort": role.effort}
    out: list[str] = []
    for line in front:
        key = next((k for k, pattern in _FRONT_LINE.items() if pattern.match(line)), None)
        if key is None:
            out.append(line)
        elif wanted.get(key) is not None:
            value = wanted.pop(key)
            out.append(line if _line_value(line) == value else f"{key}: {value}{eol}")
        else:
            wanted.pop(key, None)  # a second line of the key, or an effort the settings leave out
    missing = [f"{key}: {value}{eol}" for key, value in wanted.items() if value is not None]
    after = next((i + 1 for i, line in enumerate(out) if _FRONT_LINE["model"].match(line)), None)
    if after is None:
        after = next((i + 1 for i, line in enumerate(out) if line.startswith("name:")), len(out))
    return out[:after] + missing + out[after:]


def _agent_files(work_tree: Path) -> list[Path]:
    folder = work_tree / AGENTS_FOLDER
    return sorted(p for p in folder.glob("*.md") if p.is_file()) if folder.is_dir() else []


PATIENT_TRIES = 20  # Windows refuses a read or a replace of a file another writer is replacing


def _patient(call):
    """`call()`, tried again a few times (about 2 s in all) while Windows refuses it with a
    `PermissionError` because another writer replaces the same file at that moment."""

    for attempt in range(PATIENT_TRIES):
        try:
            return call()
        except PermissionError:
            if attempt == PATIENT_TRIES - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


def _read(path: Path, rel: str) -> tuple[str, list[str], list[str]]:
    """(text, frontmatter lines, the rest) of an agent file, or `agent_unreadable` /
    `frontmatter_missing` naming it."""

    try:
        text = _patient(path.read_bytes).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise RolesError("agent_unreadable", rel) from None
    parts = _split(text)
    if parts is None:
        raise RolesError("frontmatter_missing", rel)
    return text, *parts


def drift(work_tree: str | Path, roles: RoleSet) -> list[str]:
    """One line per agent file of `work_tree` out of step with `roles`: a model or effort the
    settings do not say, no role of the file's name, no frontmatter, not UTF-8 text. The test
    `tests/test_roles_settings.py` fails on any line for the repo's own agent files."""

    work_tree = Path(work_tree)
    found = []
    for path in _agent_files(work_tree):
        rel = path.relative_to(work_tree).as_posix()
        role = roles.roles.get(path.stem)
        if role is None:
            found.append(f"{rel}: no role `{path.stem}` in the roles settings")
            continue
        try:
            _, front, _ = _read(path, rel)
        except RolesError as e:
            found.append(f"{rel}: {'no frontmatter' if e.code == 'frontmatter_missing' else 'not UTF-8 text'}")
            continue
        for key, wanted in (("model", role.model), ("effort", role.effort)):
            have = _values(front, key)
            if len(have) > 1:
                found.append(f"{rel}: {key} is given {len(have)} times")
            elif (have[0] if have else None) != wanted:
                found.append(f"{rel}: {key} is {have[0] if have else 'not set'},"
                             f" the roles settings say {wanted or 'not set'}")
    return found


def _replace(source: str, target: Path) -> None:
    _patient(lambda: os.replace(source, target))


def render(work_tree: str | Path, roles: RoleSet) -> list[tuple[str, str]]:
    """Write the `model:` and `effort:` lines of every agent file of a role from `roles`;
    (path, `changed` or `same`) per file. Nothing is written when any file of a role cannot be
    read (`agent_unreadable`, `frontmatter_missing`); a file of no role is left alone. Each file is
    replaced whole through a temporary file of its own; one that cannot be written is
    `agent_unwritable` with its temporary file removed (the files before it are already written)."""

    work_tree = Path(work_tree)
    if not work_tree.is_dir():
        raise RolesError("work_tree_missing", str(work_tree))
    planned = []
    for path in _agent_files(work_tree):
        role = roles.roles.get(path.stem)
        if role is None:
            continue
        text, front, rest = _read(path, path.relative_to(work_tree).as_posix())
        first = text.split("\n", 1)[0]
        eol = "\r" if first.endswith("\r") else ""
        planned.append((path, text, "\n".join([first, *_rendered(front, role, eol), *rest])))
    done = []
    for path, before, after in planned:
        rel = path.relative_to(work_tree).as_posix()
        if after == before:
            done.append((rel, "same"))
            continue
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(after.encode("utf-8"))
            _replace(temporary, path)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise RolesError("agent_unwritable", rel) from None
        done.append((rel, "changed"))
    return done
