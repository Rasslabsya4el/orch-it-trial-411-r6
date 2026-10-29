"""The update classifier (#397 decision 4): the skills of the pinned upstream tree and of his newer
one, compared file by file, each skill given one class.

- `same`: every file equal, byte for byte.
- `added`: a skill folder the pinned tree had not.
- `renamed`: a folder gone from the pinned tree that comes back under a new path: with the same
  files, or by a rename git reports between the two commits (`renames`, old path -> new path).
  A move to another group is a new path too.
- `removed`: a folder gone with nothing to match it.
- `contract_changed`: what the skill takes or returns changed: any field of its frontmatter
  (`name`, `description`, `disable-model-invocation`, `allowed-tools`, ...); a section whose heading names what
  it takes or returns (inputs, outputs, invocation, arguments, files, formats, templates...), or the
  whole body of a skill with no section below its title; a `<...-template>` block; the skills it
  calls through the Skill tool; a file of the folder added or removed, or a file of it that shapes
  what it writes changed (formats, templates, report and brief shapes, the tracker, label and
  domain files his setup writes, configs); a `SKILL.md` with no readable frontmatter.
- `text_changed`: any other difference, down to one line inside a paragraph.

`why` names what decided the class: the fields, sections, templates, calls or files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from orch_it.skills.copies import SKILL_FILE, Tree, blob_hash, skill_name
from orch_it.skills.references import skill_calls

CLASSES = ("same", "added", "renamed", "removed", "text_changed", "contract_changed")
# a heading that names what the skill takes or returns; a verb counts only as "it reads/writes/produces/
# returns/takes" (third person), so a step telling what to do ("4. Write", "Write history, not the present") is text
_CONTRACT_HEADING = re.compile(
    r"\b(inputs?|outputs?|returns|return value|arguments?|parameters?|invocation|usage|preconditions?|templates?"
    r"|formats?|files?|deliverables?|definition of done|acceptance criteria|reads|writes|produces|takes|results?"
    r"|options?|interface|args?|requirements?)\b")
# support files that shape what the skill writes: his formats, templates, report and brief shapes, the files his
# setup writes (tracker, labels, domain) and configs it installs
_CONTRACT_FILE = re.compile(r"format|template|report|brief|scope|label|tracker|domain|config", re.IGNORECASE)
_HEADING = re.compile(r" {0,3}(#{1,6})[ \t]+(\S.*)")
_SETEXT = re.compile(r" {0,3}(=+|-+)[ \t]*")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_TEMPLATE_OPEN = re.compile(r"<([A-Za-z0-9_-]+[-_]template)\b[^>]*>", re.IGNORECASE)
_KEY = re.compile(r"([A-Za-z0-9_-]+):(.*)$")


@dataclass(frozen=True)
class Change:
    name: str                 # the skill's name in the newer tree, or in the pinned one when removed
    cls: str                  # one of CLASSES
    old_path: str | None
    new_path: str | None
    why: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.cls not in CLASSES:
            raise ValueError(f"not a class: {self.cls}")

    @property
    def old_name(self) -> str | None:
        return skill_name(self.old_path) if self.old_path else None


def classify(old: Tree, new: Tree, *, renames: dict[str, str] | None = None) -> list[Change]:
    """One change per skill of either tree, sorted by name."""

    renames = renames or {}
    changes = []
    for path in sorted(old.keys() & new.keys()):
        changes.append(_compare(skill_name(path), path, path, old[path], new[path]))
    gone = sorted(old.keys() - new.keys())
    fresh = sorted(new.keys() - old.keys())
    for path in gone:
        target = renames.get(path)
        if target not in fresh:
            target = next((n for n in fresh if _identity(new[n]) == _identity(old[path])), None)
        if target is None:
            changes.append(Change(skill_name(path), "removed", path, None))
            continue
        fresh.remove(target)
        compared = _compare(skill_name(target), path, target, old[path], new[target])
        changes.append(Change(skill_name(target), "renamed", path, target, compared.why))
    changes += [Change(skill_name(path), "added", None, path) for path in fresh]
    return sorted(changes, key=lambda c: (c.name, c.old_path or ""))


def _identity(files: dict[str, bytes]) -> frozenset:
    return frozenset((rel, blob_hash(data)) for rel, data in files.items())


def _compare(name: str, old_path: str, new_path: str, old: dict[str, bytes], new: dict[str, bytes]) -> Change:
    if old == new:
        return Change(name, "same", old_path, new_path)
    changed = sorted(rel for rel in old.keys() & new.keys() if old[rel] != new[rel])
    contract = [f"files: +{rel}" for rel in sorted(new.keys() - old.keys())]
    contract += [f"files: -{rel}" for rel in sorted(old.keys() - new.keys())]
    for rel in changed:
        if rel == SKILL_FILE:
            contract += _skill_contract(old[rel], new[rel])
        elif _CONTRACT_FILE.search(rel):
            contract.append(rel)
    if contract:
        return Change(name, "contract_changed", old_path, new_path, contract)
    return Change(name, "text_changed", old_path, new_path, changed)


def _skill_contract(old: bytes, new: bytes) -> list[str]:
    try:
        old_text, new_text = (data.decode("utf-8").removeprefix("﻿").replace("\r\n", "\n")
                              for data in (old, new))
    except UnicodeDecodeError:
        return [f"{SKILL_FILE} unreadable"]
    old_front, old_body = _split(old_text)
    new_front, new_body = _split(new_text)
    if old_front is None or new_front is None:
        return ["frontmatter"]
    why = [key for key in [*new_front, *(k for k in old_front if k not in new_front)]
           if old_front.get(key) != new_front.get(key)]
    old_sections, new_sections = _contract_sections(old_body), _contract_sections(new_body)
    for heading in [*new_sections, *(h for h in old_sections if h not in new_sections)]:
        if old_sections.get(heading) != new_sections.get(heading):
            why.append("body" if heading == "body" else f"section {heading}")
    old_templates, new_templates = _templates(old_body), _templates(new_body)
    for tag in sorted(old_templates.keys() | new_templates.keys()):
        if old_templates.get(tag) != new_templates.get(tag):
            why.append(f"template {tag}")
    old_calls, new_calls = skill_calls(old_body), skill_calls(new_body)
    why += [f"calls +{n}" for n in sorted(new_calls - old_calls)]
    why += [f"calls -{n}" for n in sorted(old_calls - new_calls)]
    return why


def _split(text: str) -> tuple[dict[str, list[str]] | None, str]:
    """The frontmatter's fields (each key with every value it is given, in order) and the body, or
    (None, text) when there is no closed frontmatter."""

    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 3)
    if end == -1:
        if not text.endswith("\n---"):
            return None, text
        end = len(text) - 4
    fields: dict[str, list[list[str]]] = {}
    current = None
    for line in text[4:end].split("\n"):
        found = _KEY.match(line)
        if found and not line[:1].isspace():
            current = [found.group(2)]
            fields.setdefault(found.group(1), []).append(current)
        elif current is not None:
            current.append(line)
    return {key: [_value(parts) for parts in values] for key, values in fields.items()}, text[end + 5:]


def _value(parts: list[str]) -> str:
    value = " ".join(part.strip() for part in parts).strip()
    if value[:1] in (">", "|"):
        value = value[1:].lstrip("+-").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].replace('\\"', '"') if value[0] == '"' else value[1:-1].replace("''", "'")
    return " ".join(value.split())


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(first line index, level, title) of each heading outside code fences: ATX (`## x`, up to three
    spaces in) and setext (a line under `===` or `---`)."""

    found = []
    fence = None  # (char, length) of the open fence
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        opened = _FENCE.match(line)
        if fence:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= fence[1] \
                    and not opened.group(2).strip():
                fence = None
            continue
        if opened and not (opened.group(1)[0] == "`" and "`" in opened.group(2)):
            fence = (opened.group(1)[0], len(opened.group(1)))
            continue
        atx = _HEADING.match(line)
        if atx:
            found.append((i, len(atx.group(1)), atx.group(2).rstrip("#").strip()))
            continue
        under = _SETEXT.fullmatch(line)
        previous = lines[i - 1].strip() if i else ""
        if under and previous and not _HEADING.match(lines[i - 1].rstrip()) \
                and not _FENCE.match(lines[i - 1].rstrip()) and not (found and found[-1][0] == i - 1):
            found.append((i - 1, 1 if under.group(1)[0] == "=" else 2, previous))
    return found


def _contract_sections(body: str) -> dict[str, str]:
    """Each section whose heading names what the skill takes or returns, heading -> its text up to
    the next heading of its level or higher; headings inside code fences are text. A body with no
    heading below its title is one section, all of it."""

    lines = body.split("\n")
    headings = _headings(lines)
    if not any(level >= 2 for _, level, _ in headings):
        return {"body": "\n".join(line.rstrip() for line in lines).strip()}
    sections = {}
    for n, (start, level, title) in enumerate(headings):
        if not _CONTRACT_HEADING.search(title.lower()):
            continue
        end = next((i for i, lv, _ in headings[n + 1:] if lv <= level), len(lines))
        text = "\n".join(line.rstrip() for line in lines[start:end]).strip()
        sections[title] = sections.get(title, "") + text
    return sections


def _templates(body: str) -> dict[str, str]:
    """Each `<...-template>` block, any letter case, with attributes or left open to the end."""

    found: dict[str, str] = {}
    at = 0
    while (opened := _TEMPLATE_OPEN.search(body, at)) is not None:
        tag = opened.group(1).lower()
        closed = re.compile(rf"</{re.escape(opened.group(1))}\s*>", re.IGNORECASE).search(body, opened.end())
        end = closed.end() if closed else len(body)
        found[tag] = found.get(tag, "") + body[opened.start():end]
        at = end
    return found
