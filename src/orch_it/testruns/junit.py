"""The JUnit XML reader (#394 decision 1): a report as written by pytest, jest-junit,
go-junit-report, gotestsum, Maven Surefire and most other runners, read into its test cases.

`read` returns every `<testcase>` under the root `<testsuites>` or `<testsuite>`, at any depth
(a test case inside another one is not a case of its own), in file order. The file is parsed in
chunks, so a huge report never sits in memory whole. A report that is missing, unreadable,
half-written, declares a DTD or an entity, or has another root raises `ReportUnreadable`; the
caller never gets a partial list.

A case's outcome is `failed` with a `<failure>` child, `error` with an `<error>` child (either
one wins over `<skipped>`), `skipped` with a `<skipped>` child, else `passed`.
"""

from __future__ import annotations

import math
import re
import xml.parsers.expat
from dataclasses import dataclass
from pathlib import Path

ROOTS = ("testsuites", "testsuite")
CHUNK_BYTES = 1 << 20
SKIP_REASON_CHARS = 300
NAME_CHARS = 1000  # a classname or name longer than this is cut: test ids stay bounded
MAX_SECONDS = 10.0 ** 7  # about 115 days: a longer `time` is no measurement
SUITE_CASE = "(suite)"
_TIME = re.compile(r"\d+(\.\d*)?([eE][+-]?\d+)?|\d{1,3}(,\d{3})+(\.\d*)?")
REPORT_ERROR_CODES = ("missing", "broken")
_RANK = {"passed": 0, "skipped": 1, "error": 2, "failed": 3}
_CHILD_OUTCOME = {"failure": "failed", "error": "error", "skipped": "skipped"}


class ReportUnreadable(Exception):
    """`code`: `missing` (no file, or the OS would not read it), `broken` (half-written, not
    XML, a DTD or entity in it, or not a JUnit report)."""

    def __init__(self, code: str, detail: str = ""):
        if code not in REPORT_ERROR_CODES:
            raise ValueError(f"not a report error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class Case:
    classname: str
    name: str
    outcome: str  # passed, failed, error or skipped
    seconds: float | None
    skip_reason: str | None  # the `message` (else `type`) of a skipped case's <skipped>

    @property
    def test_id(self) -> str:
        """`<classname>::<name>`, or the name alone when the classname is empty or the same
        (jest-junit writes the full title in both)."""

        if not self.classname or self.classname == self.name:
            return self.name
        return f"{self.classname}::{self.name}"

    def result_detail(self, attempt: int) -> dict:
        """The detail of this case's `test.result` event."""

        return {"test_id": self.test_id, "outcome": self.outcome, "skip_reason": self.skip_reason,
                "ms": None if self.seconds is None else round(self.seconds * 1000),
                "attempt": attempt}


class _Suite:
    def __init__(self, name: str, depth: int, declared: int):
        self.name = name
        self.depth = depth
        self.declared = declared  # its `failures` + `errors` attributes
        self.red = 0  # red cases read inside it


class _Reader:
    def __init__(self):
        self.cases: list[Case] = []
        self.depth = 0
        self.root_seen = False
        self.case_depth = None  # the depth of the open <testcase>, None outside one
        self.open_case: dict | None = None
        self.suites: list[_Suite] = []

    def _suite_red(self, suite: _Suite) -> None:
        """A failure of the suite itself (it did not load, or it declares more red than its
        cases show): one red case named `<suite>::(suite)`, so the run reads red."""

        self._add(Case(suite.name, SUITE_CASE, "error", None, None))

    def _add(self, case: Case) -> None:
        self.cases.append(case)
        if case.outcome in ("failed", "error"):
            for suite in self.suites:
                suite.red += 1

    def start(self, tag: str, attrs: dict) -> None:
        self.depth += 1
        if not self.root_seen:
            self.root_seen = True
            if tag not in ROOTS:
                raise ReportUnreadable("broken", f"root <{tag}>")
        if self.case_depth is None:
            if tag == "testsuite":
                self.suites.append(_Suite((attrs.get("name") or "")[:NAME_CHARS], self.depth,
                                          _count(attrs.get("failures")) + _count(attrs.get("errors"))))
            elif tag == "testcase":
                self.case_depth = self.depth
                self.open_case = {"classname": (attrs.get("classname") or "")[:NAME_CHARS],
                                  "name": (attrs.get("name") or "")[:NAME_CHARS],
                                  "outcome": "passed", "seconds": _seconds(attrs.get("time")),
                                  "skip_reason": None}
            elif tag in ("failure", "error") and self.suites and self.depth == self.suites[-1].depth + 1:
                self._suite_red(self.suites[-1])
        elif self.depth == self.case_depth + 1 and tag in _CHILD_OUTCOME:
            outcome = _CHILD_OUTCOME[tag]
            case = self.open_case
            if _RANK[outcome] > _RANK[case["outcome"]]:
                case["outcome"] = outcome
                text = attrs.get("message") or attrs.get("type") if outcome == "skipped" else None
                case["skip_reason"] = text[:SKIP_REASON_CHARS] if text else None

    def end(self, tag: str) -> None:
        if self.case_depth == self.depth:
            self._add(Case(**self.open_case))
            self.case_depth = None
            self.open_case = None
        elif self.suites and self.suites[-1].depth == self.depth:
            suite = self.suites.pop()
            if suite.declared > suite.red:
                self._suite_red(suite)
        self.depth -= 1


def _count(text: str | None) -> int:
    try:
        value = int(text) if text else 0
    except ValueError:
        return 0
    return max(value, 0)


def _seconds(text: str | None) -> float | None:
    """A `time` attribute in seconds: a plain decimal (`0.25`), or one with thousands commas
    (`1,234.5`). Anything else (`1,5`, `1_0`, `nan`, a negative, more than `MAX_SECONDS`) is
    unknown."""

    if not text or not _TIME.fullmatch(text.strip()):
        return None
    value = float(text.strip().replace(",", ""))
    return value if math.isfinite(value) and 0 <= value <= MAX_SECONDS else None


def _refuse(*_args) -> None:
    raise ReportUnreadable("broken", "a DTD or entity declaration")


def read(path: Path) -> list[Case]:
    reader = _Reader()
    parser = xml.parsers.expat.ParserCreate()
    parser.StartElementHandler = reader.start
    parser.EndElementHandler = reader.end
    parser.StartDoctypeDeclHandler = _refuse
    parser.EntityDeclHandler = _refuse
    try:
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_BYTES):
                parser.Parse(chunk, False)
            parser.Parse(b"", True)
    except FileNotFoundError:
        raise ReportUnreadable("missing", str(path)) from None
    except OSError as e:  # a folder, a file the OS would not open
        raise ReportUnreadable("missing", f"{path}: {e}") from None
    except (xml.parsers.expat.ExpatError, ValueError, LookupError) as e:
        # an encoding expat cannot decode, or (ValueError) a path the OS cannot name
        raise ReportUnreadable("broken", f"{path}: {e}") from None
    return reader.cases
