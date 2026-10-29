"""The tickets the main check opens (#400 decision 4), each written in #398's ticket shape (`build`
and `build_no_result` check it): the red-on-main ticket, one per set of tests that fail on main,
ready for an agent, top priority, its failing test ids in a fenced block under the layer's mark,
the pull request that brought them named; and the ticket to the brain when main's tests gave no
result twice. The follow reads the red-on-main tickets (`open_reds`) to tell a pull request red
only on their tests (it parks on that ticket, not counted) from one red on its own tests.

A test id is compared as `normal` gives it: one line, its runs of white space one space, at most
`ID_CHARS` long, so an id with a line end, a trailing space or a run of backticks reads back as it
was listed. A body lists as many ids as fit `BODY_MAX`; its mark holds the count and a hash of the
whole set, so a ticket that lists only part of a huge failure (`truncated`) still covers all of it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from orch_it.tickets import parse, violations
from orch_it.tickets.model import PRIORITY_LABEL, READY_LABEL
from orch_it.tickets.reader import TRIAGE_LABEL

MARK = "<!-- orch-it red-main -->"
_SET = re.compile(r"<!-- orch-it red-main-set count=(\d+) sha=([0-9a-f]{16}) -->")
ID_CHARS = 300
IDS_IN_EVENT = 200  # the ids a `main.red` event lists; the ticket holds the rest
IDS_HEADING = "Failing tests:"
LABELS = (READY_LABEL, PRIORITY_LABEL)
WEIGHT = 3
BODY_MAX = 60_000  # under GitHub's 65536 for an issue body; failing tests past it are counted, not listed


class RedMainShapeError(AssertionError):
    """A ticket this module wrote breaks a shape rule: a bug of this module, never of the input."""


@dataclass(frozen=True)
class RedTicket:
    number: int
    ids: frozenset[str] | None  # None: its ids block cannot be read (a person edited it)
    truncated: bool  # its body lists part of the failing set

    def covers(self, failing: set[str]) -> bool:
        return self.ids is not None and (self.truncated or failing <= self.ids)


def normal(test_id: str) -> str:
    return " ".join(str(test_id).split())[:ID_CHARS]


def open_reds(hub, repo: str) -> list[RedTicket]:
    """The open red-on-main tickets of `repo` (the `priority` issues with the layer's mark)."""

    found = []
    for number in hub.issues_with_label(repo, PRIORITY_LABEL):
        issue = hub.issue(repo, number)
        if issue is None or not issue.open or MARK not in (issue.body or ""):
            continue
        ids = ids_of(issue.body)
        whole = _SET.search(issue.body)
        found.append(RedTicket(number, ids, bool(ids is not None and whole and int(whole.group(1)) > len(ids))))
    return found


def _checked(title: str, body: str, labels) -> tuple[str, str]:
    broken = violations(parse(body, title=title, labels=labels))
    if broken:
        raise RedMainShapeError(", ".join(v.message for v in broken))
    return title, body


def build(ids: list[str], *, sha: str, run_id: str, pr: int | None) -> tuple[str, str]:
    """(title, body) of the red-on-main ticket for `ids`, failing on main at `sha` in run `run_id`."""

    ids = sorted({normal(test_id) for test_id in ids} - {""})
    count, newline = len(ids), "\n"
    whole = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]
    by = f"пулл-реквеста #{pr}" if pr else "коммита без пулл-реквеста"
    brought = f"#{pr}" if pr else "a commit with no pull request"
    tests = "тест падает" if count == 1 else ("теста падают" if 2 <= count % 10 <= 4 and count % 100 // 10 != 1
                                                 else "тестов падают")
    title = f"Main is red: {count} test{'s' if count != 1 else ''} fail"
    longest = max((len(run) for test_id in ids for run in re.findall(r"`+", test_id)), default=0)
    fence = "`" * max(3, longest + 1)
    listed, size = [], 0
    for test_id in ids:  # the ids that fit; a pull request red on one left out is red on its own
        size += len(test_id) + 1
        if size > BODY_MAX - 4000:
            break
        listed.append(test_id)
    more = len(ids) - len(listed)
    rest = f"\n... and {more} more failing tests of that run, not listed here.\n" if more else ""
    body = f"""{MARK}
<!-- orch-it red-main-set count={count} sha={whole} -->
## For the human
- **What:** main красный: {count} {tests} после слияния {by}.
- **Goal:** main снова зелёный, и пулл-реквесты, красные только на этих тестах, идут дальше сами.
- **Plan:** починить или откатить то, что сломало эти тесты; что выбрать, решает этот тикет.
- **Result:** полный прогон тестов на main зелёный; тикет закрывается сам на первом зелёном прогоне.
- **Undo:** two-way door (один откат); touches каждый пулл-реквест, который ждёт этот тикет.

## For the agent
Facts: the `tests` run {run_id} on main at `{sha}` failed the tests listed below (each one red in the JUnit report
of that run); they came with {brought}.
Rule: make main green again by a fix or a revert; a pull request red only on these tests waits for this ticket.

| row | check | pass |
|---|---|---|
| 1 | each failing test below on main | passes in a full run on main |
| 2 | the full suite on main after the change | green on linux and windows |
| 3 | a test of the list that passes on a rerun with no change | still fixed or marked flaky with its reason |

Outside case: a listed test that passes on a rerun with no change, a flake rather than a break (row 3).
Same object: the follow parks a pull request red only on these tests on this ticket and returns it when this ticket
closes; the main check closes this ticket on the first green run of main.
Not in this ticket: tests that turn red on main later and are not listed here: they get their own ticket.
Weight: {WEIGHT}

{IDS_HEADING}

{fence}text
{newline.join(listed)}
{fence}
{rest}"""
    return _checked(title, body, LABELS)


NO_RESULT_MARK = "<!-- orch-it main-no-result run={run_id} -->"
NO_RESULT_LABELS = (TRIAGE_LABEL,)


def build_no_result(*, sha: str, run_id: str, jobs: dict[str, str]) -> tuple[str, str]:
    """(title, body) of the ticket to the brain when main's tests gave no result twice on one run."""

    named = ", ".join(f"`{name}` {state}" for name, state in sorted(jobs.items()))
    title = "Main's tests gave no result twice"
    body = f"""{NO_RESULT_MARK.format(run_id=run_id)}
## For the human
- **What:** тесты на main дважды подряд не дали результата: прогон обрезан, отменён или без отчёта.
- **Goal:** снова видно, зелёный main или красный, и пулл-реквесты, красные только на нём, понятны.
- **Plan:** мозг смотрит, почему прогон не дал ответа (место, лимит, отчёт), и решает, что чинить.
- **Result:** следующий прогон тестов на main даёт результат на обеих системах.
- **Undo:** two-way door (один откат); touches только проверку main.

## For the agent
Facts: the `tests` run {run_id} on main at `{sha}` ended with no result on its first attempt and again on its
second: {named}. No result is never red: no red-on-main ticket was opened.
Rule: find why the run gives no result and make the next main run give one.

| row | check | pass |
|---|---|---|
| 1 | the next run of `tests` on main | every required job ends passed or failed, with its report |
| 2 | a run cut at its bound | read as no result, never red |

Outside case: a run that ends with a report but no test collected in it (row 1).
Same object: the main check reruns a run with no result once and opens this ticket on the second; the follow reads
no-result runs of a pull request the same way.
Not in this ticket: red tests on main: they get a red-on-main ticket.
Weight: 2
"""
    return _checked(title, body, NO_RESULT_LABELS)


def ids_of(body: str | None) -> frozenset[str] | None:
    """The failing test ids of a red-on-main ticket's body; None when the body is none."""

    text = (body or "").replace("\r\n", "\n")
    if MARK not in text or IDS_HEADING not in text:
        return None
    after = text[text.index(IDS_HEADING) + len(IDS_HEADING):].lstrip("\n")
    fence = after.split("\n", 1)[0]
    mark = fence[:len(fence) - len(fence.lstrip("`"))]
    if len(mark) < 3:
        return None
    lines = after.split("\n")[1:]
    ids = []
    for line in lines:
        if line.strip() == mark:
            return frozenset(ids)
        if line.strip():
            ids.append(line.strip())
    return None
