"""The outcome rules of a test run (#394 decision 2).

`judge` reads a run's JUnit report and says how the run ended, the first rule that holds
deciding:

1. the run hit its bound and was killed: `no_result` `cut` (`suspended` when the machine slept
   across the bound), whatever its report says;
2. no report, or one half-written or no JUnit XML: `no_result` `summary_missing`;
3. 0 test cases: `no_result` `none_collected`;
4. a failed or errored case: `failed` `red`. A red case is red even when a requested id is also
   missing: the ticket's table lets both rows hold, and "red at any required place is red"
   (TEST-7) decides;
5. a requested id with no case: `no_result` `ids_not_collected`;
6. else `passed`.

No `no_result` is ever written as `failed`: a run says red only from a complete report holding
a red case. `RunEnd` also gives the `test_run` end detail and one `test.result` detail per test
case, so every writer of a test run (a place, the GitHub importer) writes the same rows from
the same report.

A requested id matches a case by its test id (`<classname>::<name>`, as `test.result` writes
it), by a pytest node id (`tests/test_x.py::TestC::test_y[a::b]` is
`tests.test_x.TestC::test_y[a::b]`), or, with no `::`, as a file, folder, module, package or
suite: every case whose classname is it or under it. A runner that is not pytest is asked by its
own test ids as `test.result` writes them (a jest file path or a Maven `Class#method` names
nothing its report holds, so it reads as not collected). Blank ids are no ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orch_it.testruns import junit

RED_OUTCOMES = ("failed", "error")


@dataclass(frozen=True)
class RunEnd:
    """How one test run ended. `failed` counts the failed and the errored cases together, as
    the `test_run` end detail's `failed` does; each case's own outcome stays in its row."""

    outcome: str
    reason: str | None
    cases: tuple[junit.Case, ...]
    summary_seen: bool
    missing_ids: tuple[str, ...] = ()

    @property
    def collected(self) -> int:
        return len(self.cases)

    @property
    def failed(self) -> int:
        return sum(case.outcome in RED_OUTCOMES for case in self.cases)

    @property
    def skipped(self) -> int:
        return sum(case.outcome == "skipped" for case in self.cases)

    @property
    def passed(self) -> int:
        return sum(case.outcome == "passed" for case in self.cases)

    def ended_detail(self) -> dict:
        """This end's part of the `test_run` `step.ended` detail."""

        return {"outcome": self.outcome, "reason": self.reason, "collected": self.collected,
                "passed": self.passed, "failed": self.failed, "skipped": self.skipped,
                "summary_seen": self.summary_seen}

    def result_details(self) -> list[dict]:
        """One `test.result` detail per case, in report order; a test id seen again in the
        same report is its next `attempt`."""

        attempts: dict[str, int] = {}
        details = []
        for case in self.cases:
            attempts[case.test_id] = attempts.get(case.test_id, 0) + 1
            details.append(case.result_detail(attempts[case.test_id]))
        return details


def judge(report_path: Path | None, *, requested_ids: list[str] | None = None,
          cut: bool = False, slept: bool = False) -> RunEnd:
    """How the run ended, from its report at `report_path` (None: the run wrote none).
    `requested_ids`: the ids the run was asked to run (None: the full suite). `cut`: the run
    hit its bound and its process tree was killed; `slept`: the machine slept during the run."""

    try:
        cases = tuple(junit.read(report_path)) if report_path is not None else None
    except junit.ReportUnreadable:
        cases = None
    seen = cases is not None
    cases = cases or ()
    if cut:
        return RunEnd("no_result", "suspended" if slept else "cut", cases, seen)
    if not seen:
        return RunEnd("no_result", "summary_missing", (), False)
    if not cases:
        return RunEnd("no_result", "none_collected", cases, True)
    if any(case.outcome in RED_OUTCOMES for case in cases):
        return RunEnd("failed", "red", cases, True)
    missing = _not_collected(requested_ids or [], cases)
    if missing:
        return RunEnd("no_result", "ids_not_collected", cases, True, missing)
    return RunEnd("passed", None, cases, True)


def _not_collected(requested: list[str], cases: tuple[junit.Case, ...]) -> tuple[str, ...]:
    """The requested ids no case answers. The id rule is the `orch_it_ids` plugin's (which
    selects the tests of a pytest run), kept apart because the plugin runs where orch-it is not
    installed; a test runs every id form through both."""

    ids = {case.test_id for case in cases}
    names: dict[str, set[str]] = {}
    for case in cases:
        names.setdefault(case.classname, set()).add(case.name)
    prefixes = set()  # every classname and each dotted head of it: `a.b.c` gives a, a.b, a.b.c
    for classname in names:
        parts = classname.split(".")
        prefixes.update(".".join(parts[:n]) for n in range(1, len(parts) + 1))

    def answered(wanted: str) -> bool:
        if wanted in ids:
            return True
        if "::" not in wanted:
            return _module_of(wanted) in prefixes
        if wanted.partition("::")[0].endswith(".py"):  # a pytest node id, or the start of one
            base, bracket, params = wanted.partition("[")
            parts = base.split("::")
            classname = ".".join([_module_of(parts[0]), *parts[1:-1]])
            name = parts[-1] + bracket + params
            if not bracket and f"{classname}.{name}" in prefixes:  # a class
                return True
        else:  # a JUnit id `<classname>::<name>`
            classname, _, name = wanted.partition("::")
        own = names.get(classname, set())
        return name in own or ("[" not in name and any(n.startswith(name + "[") for n in own))

    missing = []
    for wanted in requested:
        wanted = _normal(wanted)
        if wanted and not answered(wanted):
            missing.append(wanted)
    return tuple(dict.fromkeys(missing))


def _normal(wanted: str) -> str:
    wanted = wanted.strip().replace("\\", "/")
    while wanted.startswith("./"):
        wanted = wanted[2:]
    return wanted


def _module_of(path: str) -> str:
    """A file or folder as pytest's classname of it: `tests/sub/test_x.py` and `tests/sub/` give
    `tests.sub.test_x` and `tests.sub`."""

    path = path.strip("/")
    if path.endswith(".py"):
        path = path[:-3]
    return path.replace("/", ".")
