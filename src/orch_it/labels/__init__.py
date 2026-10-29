"""The label set of the new layer and `labels sync` (#398 decision 6).

`labels.json` beside this module is the set: the five roles, `parked`, `priority` and `brain-handoff`, each with a
colour and a description, in the order the layer shows them. "The label set is a settings file in
orch-it" means this packaged file, shipped with orch-it: every repo gets the same set. It is not
the per-user `settings.json` of `orch_it.settings`.

`sync` makes a repo's labels match it: a missing label is created, one with another colour,
description or letter case of its name is updated, an equal one is kept. A label of the repo that
is not in the set is listed as `foreign` and never changed: sync deletes nothing, so it cannot take
a label from a person or from the old layer. The old layer's `autopilot`, `autopilot-stuck`,
`provisional`, `hard-push` and `flow-fix` are not carried. A write that fails (another sync or a
person changed the label since it was read) reads the labels once more and settles that label from
what is there now; a second failure raises.

`gh` is the seam: a function that takes the argument list of one `gh` call and returns its stdout
(default: the real CLI of the importer), so the whole sync runs in tests without a network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

from orch_it.events import schema
from orch_it.testruns import github
from orch_it.testruns.github import Gh, GithubError

LABELS_FILE = Path(__file__).with_name("labels.json")
NAME_CHARS = 50  # GitHub's own limits
DESCRIPTION_CHARS = 100
LABEL_ERROR_CODES = ("labels_unreadable",)
_COLOUR = re.compile(r"[0-9a-f]{6}")


class LabelError(Exception):
    """`code`: `labels_unreadable` (the label file is missing or is no object of
    `{name: {"color", "description"}}`: a name of 1-50 characters without a comma or an outer
    space and not repeated, a colour of 6 lowercase hex digits, a description of at most 100
    characters on one line; a repeated key is refused too)."""

    def __init__(self, code: str, detail: str = ""):
        if code not in LABEL_ERROR_CODES:
            raise ValueError(f"not a label error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _object_without_repeats(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key is repeated")
    return dict(pairs)


def label_set() -> dict[str, dict[str, str]]:
    """The layer's labels, `{name: {"color": ..., "description": ...}}`, in file order."""

    try:
        found = json.loads(LABELS_FILE.read_text("utf-8"), object_pairs_hook=_object_without_repeats)
    except (OSError, ValueError) as e:
        raise LabelError("labels_unreadable", f"{LABELS_FILE.name}: {e}") from None
    if not isinstance(found, dict) or not found:
        raise LabelError("labels_unreadable", "expected an object of labels")
    seen: set[str] = set()
    for name, label in found.items():
        if (not name or len(name) > NAME_CHARS or "," in name or name != name.strip()
                or name.lower() in seen):
            raise LabelError("labels_unreadable", f"name {name!r} is empty, too long, repeated, "
                                                  f"holds a comma or starts or ends with a space")
        seen.add(name.lower())
        if not isinstance(label, dict) or set(label) != {"color", "description"}:
            raise LabelError("labels_unreadable", f"{name}: expected color and description")
        colour, description = label["color"], label["description"]
        if not isinstance(colour, str) or not _COLOUR.fullmatch(colour):
            raise LabelError("labels_unreadable", f"{name}: color is not 6 lowercase hex digits")
        if (not isinstance(description, str) or len(description) > DESCRIPTION_CHARS
                or "\n" in description or "\r" in description):
            raise LabelError("labels_unreadable", f"{name}: description is no one-line text of at "
                                                  f"most {DESCRIPTION_CHARS} characters")
    return found


def _read(repo: str, gh: Gh) -> list[dict]:
    """Every label of the repo, 100 to a page; pages come printed one after the other. An answer
    with no JSON value at all is bad output, never a repo with no labels."""

    pages = github.json_values(gh(["api", f"repos/{repo}/labels?per_page=100", "--paginate"]))
    if not pages:
        raise GithubError("gh_bad_output", "labels: an empty answer")
    found: list[dict] = []
    for page in pages:
        if not isinstance(page, list):
            raise GithubError("gh_bad_output", "labels: expected a list")
        for label in page:
            if not isinstance(label, dict) or not isinstance(label.get("name"), str):
                raise GithubError("gh_bad_output", "labels: a label without a name")
            found.append(label)
    return found


def _is_wanted(held: dict, name: str, want: dict[str, str]) -> bool:
    """Whether a label of the repo is the wanted one: name to the letter, colour without `#`
    in lowercase, a missing or null description as empty."""

    colour = str(held.get("color") or "").lstrip("#").lower()
    return (held["name"] == name and colour == want["color"]
            and (held.get("description") or "") == want["description"])


def _form_fields(name: str, want: dict[str, str], name_field: str) -> list[str]:
    """The `-f` fields of a write; the label's name goes in `name_field` (`name` of a create,
    `new_name` of an update)."""

    return ["-f", f"{name_field}={name}", "-f", f"color={want['color']}",
            "-f", f"description={want['description']}"]


def _settle(repo: str, name: str, want: dict[str, str], held: dict | None, gh: Gh,
            *, retry: bool = True) -> str:
    """Bring one set label to what is wanted: create it when the repo holds none, keep it when
    equal, else update it (a PATCH names the label as the repo spells it, so a change of letter
    case is a rename). When the write fails, the label may have changed since it was read (the text
    of `gh`'s error is not relied on): read again once and settle it from what is there now. A
    create that failed with the label still absent raises its own error; a second failure raises."""

    try:
        if held is None:
            gh(["api", "-X", "POST", f"repos/{repo}/labels", *_form_fields(name, want, "name")])
            return "created"
        if _is_wanted(held, name, want):
            return "kept"
        gh(["api", "-X", "PATCH", f"repos/{repo}/labels/{quote(held['name'], safe='')}",
            *_form_fields(name, want, "new_name")])
        return "updated"
    except GithubError as failure:
        if not retry:
            raise
        try:
            now = next((label for label in _read(repo, gh)
                        if label["name"].lower() == name.lower()), None)
        except GithubError:
            raise failure from None
        if now is None and held is None:
            raise
        return _settle(repo, name, want, now, gh, retry=False)


def ensure(repo: str, name: str, *, gh: Gh | None = None) -> bool:
    """Create one label of the set in the repo when the repo has none of its name (in any case);
    True when created. A label already there is left as it is (`sync` settles its colour)."""

    want = label_set()[name]
    if not schema.is_repo(repo):
        raise GithubError("bad_repo", str(repo))
    gh = gh or github.run_gh
    if any(label["name"].lower() == name.lower() for label in _read(repo, gh)):
        return False
    return _settle(repo, name, want, None, gh) == "created"


def sync(repo: str, *, gh: Gh | None = None) -> list[tuple[str, str]]:
    """Make the repo's labels match the set; return `(name, action)` rows: the set's labels in
    file order (`created`, `updated` or `kept`), then the repo's other labels sorted by name
    (`foreign`). The label file is read and the repo checked before `gh` is called; nothing is
    deleted."""

    wanted = label_set()
    if not schema.is_repo(repo):
        raise GithubError("bad_repo", str(repo))
    gh = gh or github.run_gh
    held = {label["name"].lower(): label for label in _read(repo, gh)}
    rows = [(name, _settle(repo, name, want, held.pop(name.lower(), None), gh))
            for name, want in wanted.items()]
    rows += [(label["name"], "foreign")
             for label in sorted(held.values(), key=lambda label: label["name"])]
    return rows
