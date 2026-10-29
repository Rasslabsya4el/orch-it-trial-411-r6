"""The closed sets of the event list and the check every event passes before it is stored.

`closed_sets.json` beside this module is the only place a kind, step id, outcome or reason
code is defined (#393 decision 4). `validate` raises `EventError` on a closed-set breach; the
error carries a code from `ERROR_CODES`, never prose to parse.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from functools import cache
from importlib import resources

PAIR_KINDS = ("step.started", "step.ended")
SIDE_OF_PAIR = {"step.started": "started", "step.ended": "ended"}
STEP_KINDS = (*PAIR_KINDS, "step.progressed")
# The generic step.ended reason codes of section 8, each with the one outcome it explains.
GENERIC_REASON_OUTCOME = {"suspended": "no_result", "lease_lost": "lost"}

ERROR_CODES = (
    "not_an_object",
    "unknown_field",
    "missing_field",
    "null_field",
    "bad_type",
    "unknown_kind",
    "unknown_step",
    "step_missing",
    "bad_at",
    "at_without_zone",
    "bad_repo",
    "bad_code",
    "missing_outcome",
    "bad_outcome",
    "too_long",
)

_REPO = re.compile(r"[A-Za-z0-9-]+/(?!\.\.?$)[A-Za-z0-9_.-]+")
_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})")
_AT_NO_ZONE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?")
_MAX_TICKET = 2**63 - 1


def is_repo(text: object) -> bool:
    """Whether `text` is a repo as the envelope writes it: `owner/name`."""

    return isinstance(text, str) and _REPO.fullmatch(text) is not None


class EventError(ValueError):
    """An event outside the closed sets. `code` is one of `ERROR_CODES`; `field` names the
    offending field (dotted for a detail field) or is None. The offending value is never part
    of the message (it may be huge or private)."""

    def __init__(self, code: str, field: str | None = None):
        if code not in ERROR_CODES:
            raise ValueError(f"not an error code: {code}")
        self.code = code
        self.field = field
        super().__init__(code, field)

    def __str__(self) -> str:
        return f"{self.code}: {self.field}" if self.field else self.code


@cache
def _sets() -> dict:
    text = resources.files("orch_it.events").joinpath("closed_sets.json").read_text("utf-8")
    return json.loads(text)


def envelope_fields() -> tuple[str, ...]:
    return tuple(_sets()["envelope"])


def step_ids() -> tuple[str, ...]:
    return tuple(_sets()["steps"])


def point_kinds() -> tuple[str, ...]:
    return tuple(_sets()["kinds"])


def kinds() -> tuple[str, ...]:
    return PAIR_KINDS + point_kinds()


def outcomes() -> list[str]:
    return list(_sets()["codes"]["outcome"])


def fields_of(owner: str) -> dict[str, dict]:
    """The named fields of `owner`: `envelope`, a point kind, or `<step id>.started` /
    `<step id>.ended`. A `<step>.ended` side always names `outcome` and `reason`. Raises
    `unknown_kind` for any other owner."""

    sets = _sets()
    if owner == "envelope":
        return dict(sets["envelope"])
    if owner in sets["kinds"]:
        return dict(sets["kinds"][owner])
    step, _, side = owner.rpartition(".")
    if step in sets["steps"] and side in SIDE_OF_PAIR.values():
        fields = dict(sets["steps"][step][side])
        if side == "ended":
            fields = {"outcome": {"codes": "outcome"}, "reason": {}} | fields
        return fields
    raise EventError("unknown_kind", owner)


def codes_of(owner: str, field: str) -> set[str] | None:
    """The codes `field` of `owner` must take, or None when its value is open.

    A field named `reason` is always coded (the list's rule 5: a reason is a code from a list
    written next to its field): with no list written it takes no code yet, and the part that
    first writes it adds its list to the closed-set file. A step's ended `reason` also takes
    the codes any `step.ended` may carry."""

    spec = fields_of(owner).get(field)
    if spec is None:
        return None
    lists = _sets()["codes"]
    name = spec.get("codes")
    if field == "reason":
        own = set(lists[name]) if name else set()
        return own | set(lists["step_ended_any"]) if owner.endswith(".ended") else own
    return set(lists[name]) if name else None


def detail_field_names() -> list[str]:
    """Every detail field the list names, once, in a stable order (the index's columns)."""

    sets = _sets()
    names: dict[str, None] = {"outcome": None, "reason": None}
    for step in sets["steps"].values():
        for side in SIDE_OF_PAIR.values():
            names.update(dict.fromkeys(step[side]))
    for fields in sets["kinds"].values():
        names.update(dict.fromkeys(fields))
    return list(names)


def validate(event: object) -> dict:
    """Return `event` unchanged when it is inside the closed sets, else raise `EventError`.

    An event is the envelope's fields, every one present (null where the envelope allows it),
    plus `detail`, an object. Every value is plain JSON (string keys, finite numbers, text that
    encodes as UTF-8). A named detail field with a code list must take a code from it; any
    other detail field is open."""

    if not isinstance(event, dict):
        raise EventError("not_an_object")
    envelope = _sets()["envelope"]
    for name in event:
        if name not in envelope and name != "detail":
            raise EventError("unknown_field", str(name))
    for name, spec in envelope.items():
        if name not in event:
            raise EventError("missing_field", name)
        _check_envelope_field(name, spec, event[name])
    if "detail" not in event:
        raise EventError("missing_field", "detail")
    detail = event["detail"]
    if not isinstance(detail, dict):
        raise EventError("bad_type", "detail")
    _check_plain_json(detail)

    kind = event["kind"]
    if kind in STEP_KINDS and event["step"] is None:
        raise EventError("step_missing", "step")
    owner = f"{event['step']}.{SIDE_OF_PAIR[kind]}" if kind in PAIR_KINDS else kind
    fields = fields_of(owner) if kind != "step.progressed" else {}
    if kind == "step.ended":
        _check_ending(detail)
    for name, value in detail.items():
        spec = fields.get(name)
        if spec is None or value is None:
            continue
        if "max_chars" in spec:
            if not isinstance(value, str):
                raise EventError("bad_type", f"detail.{name}")
            if len(value) > spec["max_chars"]:
                raise EventError("too_long", f"detail.{name}")
        codes = codes_of(owner, name)
        if codes is not None:
            for item in _code_items(value):
                if not _is_code(item, codes):
                    raise EventError("bad_code", f"detail.{name}")
    return event


def _check_ending(detail: dict) -> None:
    if "outcome" not in detail:
        raise EventError("missing_outcome", "detail.outcome")
    outcome = detail["outcome"]
    if not isinstance(outcome, str) or outcome not in _sets()["codes"]["outcome"]:
        raise EventError("bad_outcome", "detail.outcome")
    reason = detail.get("reason")
    if isinstance(reason, str) and GENERIC_REASON_OUTCOME.get(reason, outcome) != outcome:
        raise EventError("bad_code", "detail.reason")


def _check_envelope_field(name: str, spec: dict, value: object) -> None:
    if value is None:
        if not spec["nullable"]:
            raise EventError("null_field", name)
        return
    field_type = spec["type"]
    if field_type == "int":
        if type(value) is not int or not 0 < value <= _MAX_TICKET:
            raise EventError("bad_type", name)
        return
    if not isinstance(value, str) or not value:
        raise EventError("bad_type", name)
    _check_text(value, name)
    if field_type == "kind" and value not in kinds():
        raise EventError("unknown_kind", name)
    if field_type == "step" and value not in _sets()["steps"]:
        raise EventError("unknown_step", name)
    if field_type == "repo" and not _REPO.fullmatch(value):
        raise EventError("bad_repo", name)
    if field_type == "code" and value not in _sets()["codes"][spec["codes"]]:
        raise EventError("bad_code", name)
    if field_type == "timestamp":
        parse_at(value)


def parse_at(value: str) -> datetime:
    """An `at` as an aware datetime: ISO 8601 `YYYY-MM-DDTHH:MM:SS[.ffffff]` with `Z` or an
    offset `±HH:MM`. Raises `at_without_zone` for one without a zone (a reader compares in UTC,
    so a local time with no offset is unknown), `bad_at` for anything else (a leap second
    included: no reader can order it)."""

    if not isinstance(value, str) or not _AT.fullmatch(value):
        if isinstance(value, str) and _AT_NO_ZONE.fullmatch(value):
            raise EventError("at_without_zone", "at")
        raise EventError("bad_at", "at")
    try:
        moment = datetime.fromisoformat(value)
        moment.astimezone(UTC)
    except (ValueError, OverflowError):
        raise EventError("bad_at", "at") from None
    return moment


def _check_text(value: str, field: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise EventError("bad_type", field) from None


def _check_plain_json(detail: dict) -> None:
    """Every value under `detail` is JSON a reader gets back as written: objects with string
    keys, lists, UTF-8 text, finite numbers, booleans, null. Walked without recursion."""

    stack: list[tuple[str, object]] = []
    for key, item in detail.items():
        if not isinstance(key, str):
            raise EventError("bad_type", "detail")
        _check_text(key, "detail")
        stack.append((f"detail.{key}", item))
    while stack:
        path, value = stack.pop()  # path: the detail field the value sits under
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise EventError("bad_type", path)
                _check_text(key, path)
                stack.append((path, item))
        elif isinstance(value, list):
            stack.extend((path, item) for item in value)
        elif isinstance(value, str):
            _check_text(value, path)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise EventError("bad_type", path)
        elif value is not None and not isinstance(value, int):
            raise EventError("bad_type", path)


def _code_items(value: object) -> list:
    """The codes a coded field holds: the value itself, each item of a list, or each key of an
    object (a count per code, as `reconcile`'s `repairs`)."""

    if isinstance(value, dict):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _is_code(item: object, codes: set[str]) -> bool:
    """`item` is one of `codes`, or fills a placeholder code `<prefix><name>` such as
    `machine:<id>` with a non-empty value after the prefix."""

    if not isinstance(item, str):
        return False
    if item in codes:
        return True
    for code in codes:
        if code.endswith(">") and "<" in code:
            prefix = code[: code.index("<")]
            if item.startswith(prefix) and len(item) > len(prefix):
                return True
    return False
