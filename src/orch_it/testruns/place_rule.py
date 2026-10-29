"""The place rule (#394 decision 5): which test place one run goes to.

`choose` is pure. Its inputs: the OS the run needs (`linux`, `windows`, `macos` or `any`), the
places in tie order (own, machines in settings order, github), the live runs per place and the
measured durations per place, both read from the store by `read_facts`, so a decision replays
from the store alone and reads the same on every OS (load is a count of live runs, never the
OS's CPU load). In order:

1. drop a place without that OS (`os_missing`), switched off (`place_off`), at capacity
   (`busy`), or lost earlier in this same run (`excluded`, e.g. `place_vanished`);
2. for each place left, expected finish = start overhead (median of its measured starts, 0 with
   none) + run time (median of its measured runs). A place with no measured run takes the own
   place's median; when own has none either, the median of every place's measured runs;
3. with nothing measured anywhere the run goes to own, or, when own is dropped, to the first
   place left (`nothing_measured`); else to the earliest expected finish (`earliest_finish`),
   a tie going to the earlier place in tie order;
4. no place left: no choice (the caller writes a `wait` with reason `test_place`).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from orch_it.events import index
from orch_it.events.store import Store, machine_id

TIE_DIGITS = 9  # finishes equal to this many decimals of a minute are a tie
MEASURED_RUNS = 10  # decision 5: the median of the last 10 ended passed / failed runs
LIVE_GRACE_MINUTES = 10  # a started run with no end is live for its bound plus this
DEFAULT_BOUND_MINUTES = 15.0
MAX_MINUTES = 7 * 24 * 60  # a bound or a measurement over a week is no reading
# The drop reasons, each a code of `test_run_reason`, in the order the rule checks them; a place
# left out after it refused or vanished in this run keeps that attempt's own code.
OS_MISSING, PLACE_OFF, BUSY = "os_missing", "place_off", "busy"
DROP_CODES = (OS_MISSING, PLACE_OFF, BUSY)
EARLIEST, NOTHING_MEASURED = "earliest_finish", "nothing_measured"


@dataclass(frozen=True)
class Offer:
    place: str
    oses: tuple[str, ...]
    on: bool
    capacity: int


@dataclass(frozen=True)
class Measured:
    starts: tuple[float, ...] = ()  # start overheads in minutes
    runs: tuple[float, ...] = ()  # run times in minutes (the last ended passed/failed runs)


@dataclass(frozen=True)
class Offered:
    place: str
    expected_minutes: float | None
    dropped: str | None


@dataclass(frozen=True)
class Decision:
    chosen: str | None
    reason: str | None
    offered: tuple[Offered, ...] = field(default_factory=tuple)

    def detail(self) -> list[dict]:
        """`places_offered` of the `test_run` start: each place with its expected finish in
        minutes (2 decimals) or its drop reason."""

        return [{"place": o.place,
                 "expected_minutes": None if o.expected_minutes is None else round(o.expected_minutes, 2),
                 "dropped": o.dropped} for o in self.offered]


def _drop(offer: Offer, need_os: str, live: dict[str, int], excluded: dict[str, str]) -> str | None:
    if need_os != "any" and need_os not in offer.oses:
        return OS_MISSING
    if not offer.on:
        return PLACE_OFF
    if live.get(offer.place, 0) >= offer.capacity:
        return BUSY
    return excluded.get(offer.place)


def choose(need_os: str, places: list[Offer], live: dict[str, int],
           measured: dict[str, Measured], excluded: dict[str, str] | None = None) -> Decision:
    excluded = excluded or {}
    drops = {o.place: _drop(o, need_os, live, excluded) for o in places}
    left = [o for o in places if drops[o.place] is None]
    every_run = [m for place in measured.values() for m in place.runs]
    if not every_run:
        offered = tuple(Offered(o.place, None, drops[o.place]) for o in places)
        if not left:
            return Decision(None, None, offered)
        pick = next((o for o in left if o.place == "own"), left[0])
        return Decision(pick.place, NOTHING_MEASURED, offered)
    own_runs = measured.get("own", Measured()).runs
    reference = statistics.median(own_runs or every_run)
    finish = {}
    for o in left:
        m = measured.get(o.place, Measured())
        start = statistics.median(m.starts) if m.starts else 0.0
        finish[o.place] = start + (statistics.median(m.runs) if m.runs else reference)
    offered = tuple(Offered(o.place, finish.get(o.place), drops[o.place]) for o in places)
    if not left:
        return Decision(None, None, offered)
    best = min(left, key=lambda o: round(finish[o.place], TIE_DIGITS))  # min keeps the first of a tie
    return Decision(best.place, EARLIEST, offered)


@dataclass(frozen=True)
class _Started:
    """A `test_run` start as the rule reads it."""

    place: str
    repo: str
    selection: str | None
    run_os: str | None
    at: str
    machine: str
    bound: float


def read_facts(store: Store, *, repo: str, selection: str, need_os: str, now: datetime,
               machine: str | None = None) -> tuple[dict[str, int], dict[str, Measured]]:
    """The rule's store inputs at `now`: live runs per place (of every repo: a machine's capacity
    is shared), and per place the measured run times (the last `MEASURED_RUNS` ended `passed` /
    `failed` runs of `repo` and `selection`, on `need_os` unless it is `any`) and start
    overheads (the last `MEASURED_RUNS` ended runs at the place with one measured, any repo).

    `own` is this machine (`machine`, default the store's machine id): another machine's own
    runs, in a store merged by copying, are neither live nor measured here. A start dated more
    than the grace ahead of `now` (a skewed clock) is no live run; a bound or a measurement
    that is no number within a week is no reading. The live count is read, not reserved: two
    runs deciding at the same moment may both take a place's last free slot."""

    machine = machine or machine_id()
    started: dict[str, _Started] = {}
    for run, place, run_repo, run_selection, run_os, at, run_machine, detail in store.query(
            "SELECT run, place, repo, selection, run_os, at, machine, detail FROM events"
            " WHERE kind = 'step.started' AND step = 'test_run' AND run IS NOT NULL").rows:
        if not isinstance(place, str) or (place == "own" and run_machine != machine):
            continue
        bound = _minutes(json.loads(detail).get("bound_minutes")) or DEFAULT_BOUND_MINUTES
        started.setdefault(run, _Started(place, run_repo, run_selection, run_os, at, run_machine,
                                         bound))
    ends = store.query(
        "SELECT run, outcome, run_minutes, start_minutes FROM events"
        " WHERE kind = 'step.ended' AND step = 'test_run' AND run IS NOT NULL ORDER BY at DESC, id").rows

    ended_runs = {run for run, *_ in ends}
    latest = _utc(now + timedelta(minutes=LIVE_GRACE_MINUTES))
    live: dict[str, int] = {}
    for run, start in started.items():
        earliest = _utc(now - timedelta(minutes=start.bound + LIVE_GRACE_MINUTES))
        if run not in ended_runs and earliest <= start.at <= latest:
            live[start.place] = live.get(start.place, 0) + 1

    runs: dict[str, list[float]] = {}
    starts: dict[str, list[float]] = {}
    for run, outcome, run_minutes, start_minutes in ends:
        start = started.get(run)
        if start is None or (need_os != "any" and start.run_os != need_os):
            continue
        place = start.place
        if _minutes(start_minutes) is not None and len(starts.setdefault(place, [])) < MEASURED_RUNS:
            starts[place].append(_minutes(start_minutes))
        if (outcome in ("passed", "failed") and start.repo == repo and start.selection == selection
                and _minutes(run_minutes) is not None
                and len(runs.setdefault(place, [])) < MEASURED_RUNS):
            runs[place].append(_minutes(run_minutes))
    measured = {place: Measured(tuple(starts.get(place, ())), tuple(runs.get(place, ())))
                for place in set(starts) | set(runs) if starts.get(place) or runs.get(place)}
    return live, measured


def _minutes(value: object) -> float | None:
    """A number of minutes from the store: finite, 0 to `MAX_MINUTES`; else None."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        minutes = float(value)
    except OverflowError:
        return None
    return minutes if math.isfinite(minutes) and 0 <= minutes <= MAX_MINUTES else None


def _utc(moment: datetime) -> str:
    return index.utc_at(moment.isoformat())
