"""The named queries of the store (#393 decisions 8 and 9): `ticket_cost`, `session_cost`,
`limit_rate`, and the share of the weekly subscription limit a cost is; and `test_places`
(#394 decision 9).

Each returns a `QueryResult` (columns and rows) for the CLI to print. The SQL of `ticket_cost`
is the event list's section 7, kept as a file beside this module and run as written.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from importlib import resources

from orch_it import settings
from orch_it.events import index, schema
from orch_it.events.store import QueryResult, Store

WEEKLY_LIMIT_SETTING = "weekly_limit_usd_per_pct"
CLAUDE_PREFIX = "claude-"


def load_sql(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.sql").read_text("utf-8")


# --- the share of the weekly limit -----------------------------------------------------------

def pct_week(usd: float | None, model: str | None, rate: float | None) -> str:
    """A cost as a share of the weekly limit: `n/a` for a model that is not Claude, `not set`
    while the setting is unset, `price_missing` for a turn without a price, else `usd / rate`
    (dollars per 1 %) as `0.18 %`. Never a guessed number."""

    if not (model or "").startswith(CLAUDE_PREFIX):
        return "n/a"
    if rate is None:
        return "not set"
    if usd is None:
        return "price_missing"
    return f"{usd / rate:.2f} %"


class QueryError(Exception):
    """A named query that has no answer for what was asked. `code`: `ticket_not_ready` (no
    edit added `ready-for-agent`), `ticket_not_closed` (no `ticket.closed`)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


RATE_BOUNDS = (0.01, 1_000_000.0)  # dollars per 1 %: anything outside is no reading of a card


def checked_rate(usd_per_pct: object, plan: object, measured: object) -> dict:
    """The setting's value, or `SettingsError("bad_setting")` naming what is wrong: a rate
    that is no finite number within `RATE_BOUNDS`, a blank plan, a `measured` that is no ISO
    8601 date or time."""

    if isinstance(usd_per_pct, bool) or not isinstance(usd_per_pct, (int, float)):
        raise settings.SettingsError("bad_setting", f"{WEEKLY_LIMIT_SETTING}.usd_per_pct")
    try:
        rate = float(usd_per_pct)
    except OverflowError:
        raise settings.SettingsError("bad_setting", f"{WEEKLY_LIMIT_SETTING}.usd_per_pct") from None
    if not (math.isfinite(rate) and RATE_BOUNDS[0] <= rate <= RATE_BOUNDS[1]):
        raise settings.SettingsError("bad_setting", f"{WEEKLY_LIMIT_SETTING}.usd_per_pct")
    if not isinstance(plan, str) or not plan.strip():
        raise settings.SettingsError("bad_setting", f"{WEEKLY_LIMIT_SETTING}.plan")
    try:
        datetime.fromisoformat(measured)
    except (TypeError, ValueError):
        raise settings.SettingsError("bad_setting", f"{WEEKLY_LIMIT_SETTING}.measured") from None
    return {"usd_per_pct": rate, "plan": plan, "measured": measured}


def weekly_rate() -> dict | None:
    """The setting `weekly_limit_usd_per_pct` as `{"usd_per_pct", "plan", "measured"}`, or None
    when unset. A setting of another shape raises `SettingsError("bad_setting")`."""

    value = settings.load().get(WEEKLY_LIMIT_SETTING)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise settings.SettingsError("bad_setting", WEEKLY_LIMIT_SETTING)
    return checked_rate(value.get("usd_per_pct"), value.get("plan"), value.get("measured"))


def _usd_per_pct() -> float | None:
    rate = weekly_rate()
    return rate["usd_per_pct"] if rate else None


# --- ticket-cost -----------------------------------------------------------------------------

_SPEND_FILTER = "kind = 'turn.used' AND repo = :repo"
_CLAUDE_SPEND_FILTER = "kind = 'turn.used' AND model LIKE 'claude-%' AND repo = :repo"


def ticket_cost(store: Store, repo: str, n: int) -> QueryResult:
    """Section 7: the cost of ticket `n` from its ready edit to its close, per role and `how`
    (`direct`, or `shared` by the tickets a session held). `% week` is computed from the Claude
    part of each row: the same query restricted to `model LIKE 'claude-%'`."""

    sql = load_sql("ticket_cost")
    params = {"repo": repo, "n": n}
    bounds_sql = sql[:sql.index("-- a turn spent")].rstrip().rstrip(",") + "\nSELECT t_ready, t_close FROM bounds"
    t_ready, t_close = store.query(bounds_sql, params).rows[0]
    if t_ready is None:
        raise QueryError("ticket_not_ready", f"{repo}#{n} has no edit adding ready-for-agent")
    if t_close is None:
        raise QueryError("ticket_not_closed", f"{repo}#{n} has no ticket.closed")
    claude_sql = sql.replace(_SPEND_FILTER, _CLAUDE_SPEND_FILTER)
    if claude_sql == sql:
        raise RuntimeError("ticket_cost.sql no longer holds the turn.used filter")
    rate = _usd_per_pct()
    claude = {(role, how): usd for role, how, usd in store.query(claude_sql, params).rows}
    rows = []
    for role, how, usd in store.query(sql, params).rows:
        if (role, how) in claude:
            share = pct_week(claude[(role, how)], CLAUDE_PREFIX, rate)
        else:
            share = "n/a"  # no Claude turn in this row
        rows.append((role, how, usd, share))
    return QueryResult(["role", "how", "usd", "% week"], rows)


# --- session-cost ------------------------------------------------------------------------------

SESSION_COLUMNS = ["session", "machine", "model", "turns", "input", "output", "cache_read",
                   "cache_write", "usd", "% week", "minutes"]
_AT_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_SESSION_STEPS = "('session', 'subagent')"
_NO_TURNS = ("", 0, 0, 0, 0, 0, 0.0, 0)


class _Sums:
    """What a row or a total line adds up. `usd` leaves out the turns without a price."""

    FIELDS = ("turns", "input", "output", "cache_read", "cache_write", "usd", "missing",
              "claude_turns", "claude_missing", "claude_usd", "minutes")

    def __init__(self):
        for name in self.FIELDS:
            setattr(self, name, 0)

    def add_turns(self, model, turns, input_, output, cache_read, cache_write, usd, missing):
        self.turns += turns
        self.input += input_
        self.output += output
        self.cache_read += cache_read
        self.cache_write += cache_write
        self.usd += usd
        self.missing += missing
        if model.startswith(CLAUDE_PREFIX):
            self.claude_turns += turns
            self.claude_missing += missing
            self.claude_usd += usd

    def add_all(self, other: _Sums):
        for name in self.FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def usd_cell(self, *, bare_when_all_missing: bool) -> str:
        if bare_when_all_missing and self.missing and self.missing == self.turns:
            return "price_missing"
        text = f"{self.usd:.2f}"
        return f"{text} ({self.missing} price_missing left out)" if self.missing else text

    def pct_cell(self, rate) -> str:
        if not self.claude_turns:
            return "n/a"
        priced = self.claude_turns > self.claude_missing
        return pct_week(self.claude_usd if priced else None, CLAUDE_PREFIX, rate)

    def row(self, label, machine, model, rate, minutes, *, bare_when_all_missing) -> tuple:
        return (label, machine, model, self.turns, self.input, self.output, self.cache_read,
                self.cache_write, self.usd_cell(bare_when_all_missing=bare_when_all_missing),
                self.pct_cell(rate), minutes)


def _utc(at: str, *, end_of_day: bool = False) -> str:
    """`at` (ISO 8601 with a zone, or a bare UTC date) as the index writes it. A bare date as
    an upper bound is the last moment of that day, so the bound keeps the whole day."""

    if len(at) != 10:
        return index.utc_at(at)
    start = index.utc_at(f"{at}T00:00:00Z")
    if not end_of_day:
        return start
    day_after = datetime.strptime(start, _AT_FORMAT) + timedelta(days=1, microseconds=-1)
    return day_after.strftime(_AT_FORMAT)


def _minutes(first, last, started, ended) -> float:
    """From the session's start to its end; a missing or inverted pair falls back to its
    first and last events."""

    start, end = (started, ended or last) if started else (first, last)
    if end < start:
        start, end = first, last
    spent = datetime.strptime(end, _AT_FORMAT) - datetime.strptime(start, _AT_FORMAT)
    return max(spent.total_seconds() / 60, 0.0)


def session_cost(store: Store, *, session: str | None = None, repo: str | None = None,
                 since: str | None = None, until: str | None = None) -> QueryResult:
    """Dollars, `% week`, tokens and minutes per session and model, a subagent listed under its
    parent, a total line per parent (its subagents included) and a grand total.

    `session` keeps that session and its subagents; `repo`, `since` and `until` (inclusive)
    keep the events they match, so sums and minutes are of those events. A turn without a
    price shows `price_missing` and is left out of the sums. Minutes run from the session's
    `step.started` to its `step.ended`; with no end, to its last event; with no start, from its
    first event. A total line adds the minutes of the sessions it holds."""

    rate = _usd_per_pct()
    where, params = ["session IS NOT NULL"], {}
    for clause, name, value in (("repo = :repo", "repo", repo),
                                ("at >= :since", "since", since and _utc(since)),
                                ("at <= :until", "until", until and _utc(until, end_of_day=True))):
        if value is not None:
            where.append(clause)
            params[name] = value
    filters = " AND ".join(where)

    info = {}  # session -> (parent, machine, first at, last at, started at, ended at)
    for sid, *rest in store.query(
            "SELECT session, max(parent_session), min(machine), min(at), max(at),"
            f" min(CASE WHEN kind = 'step.started' AND step IN {_SESSION_STEPS} THEN at END),"
            f" max(CASE WHEN kind = 'step.ended' AND step IN {_SESSION_STEPS} THEN at END)"
            f" FROM events WHERE {filters} GROUP BY session", params).rows:
        info[sid] = rest

    def ancestors(sid):
        found, parent = [], info[sid][0]
        while parent is not None and parent != sid and parent not in found:
            found.append(parent)
            parent = info[parent][0] if parent in info else None
        return found

    models = {}  # session -> [(model, turns, input, output, cache_read, cache_write, usd, missing)]
    for sid, *rest in store.query(
            "SELECT session, coalesce(model, ''), count(*), coalesce(sum(input_tokens), 0),"
            " coalesce(sum(output_tokens), 0), coalesce(sum(cache_read_tokens), 0),"
            " coalesce(sum(cache_write_tokens), 0), coalesce(sum(usd), 0), sum(usd IS NULL)"
            f" FROM events WHERE kind = 'turn.used' AND {filters}"
            " GROUP BY session, model ORDER BY model", params).rows:
        models.setdefault(sid, []).append(tuple(rest))

    groups = {}  # the top-level session -> the sessions under it, itself first
    for sid in sorted(info, key=lambda s: (info[s][2], s)):
        chain = ancestors(sid)
        if session is None or session == sid or session in chain:
            groups.setdefault((chain or [sid])[-1], []).append(sid)

    rows, grand = [], _Sums()
    for root in sorted(groups, key=lambda r: (min(info[s][2] for s in groups[r]), r)):
        total = _Sums()
        for sid in sorted(groups[root], key=lambda s: (s != root, info[s][2], s)):
            _, machine, first, last, started, ended = info[sid]
            minutes = _minutes(first, last, started, ended)
            total.minutes += minutes
            label = sid if sid == root else f"  {sid}"
            for n, (model, *numbers) in enumerate(models.get(sid) or [_NO_TURNS]):
                sums = _Sums()
                sums.add_turns(model, *numbers)
                total.add_turns(model, *numbers)
                rows.append(sums.row(label, machine, model, rate, f"{minutes:.1f}" if n == 0 else "",
                                     bare_when_all_missing=True))
        rows.append(total.row(f"total {root}", "", "", rate, f"{total.minutes:.1f}",
                              bare_when_all_missing=False))
        grand.add_all(total)
    if groups:
        rows.append(grand.row("total all", "", "", rate, f"{grand.minutes:.1f}",
                              bare_when_all_missing=False))
    return QueryResult(list(SESSION_COLUMNS), rows)


# --- limit-rate --------------------------------------------------------------------------------

def limit_rate(store: Store) -> QueryResult:
    """Dollars at API prices per 1 % of the weekly limit, from each two consecutive readings
    (`value.measured`, `name` `weekly_limit_pct`): the `usd` of Claude `turn.used` events with
    `at` in (from, to] over the rise. A pair whose percent fell (the weekly reset) shows
    `reset`, one that stayed `no_change`, one with no Claude turn between `no_turns`, one with a
    Claude turn without a price `price_missing`."""

    readings = store.query(
        "SELECT at, value FROM events WHERE kind = 'value.measured' AND name = 'weekly_limit_pct'"
        " AND typeof(value) IN ('integer', 'real') ORDER BY at, id").rows
    rows = []
    for (start, before), (end, after) in zip(readings, readings[1:]):
        turns, usd, missing = store.query(
            "SELECT count(*), coalesce(sum(usd), 0), coalesce(sum(usd IS NULL), 0) FROM events"
            " WHERE kind = 'turn.used' AND model LIKE 'claude-%' AND at > ? AND at <= ?",
            (start, end)).rows[0]
        if after < before:
            rate = "reset"
        elif after == before:
            rate = "no_change"
        elif not turns:
            rate = "no_turns"
        elif missing:
            rate = "price_missing"
        else:
            rate = f"{usd / (after - before):.2f}"
        rows.append((start, end, before, after, round(usd, 2), rate))
    return QueryResult(["from", "to", "pct_before", "pct_after", "usd", "usd_per_pct"], rows)


# --- test-places -------------------------------------------------------------------------------

TEST_PLACES_COLUMNS = ["place", "os", "runs", "outcomes", "median_run_minutes", "median_queue_minutes"]
_MEASURED_OUTCOMES = ("passed", "failed")  # a run that is neither measures nothing
_NOT_MEASURED = "n/a"


def _place_order(place: str) -> tuple:
    """Own first, then the machines by id, then github, then any other place."""

    if place == "own":
        return (0, "")
    if place.startswith("machine:"):
        return (1, place)
    return (2, "") if place == "github" else (3, place)


def _median_cell(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else _NOT_MEASURED


def test_places(store: Store, repo: str | None = None) -> QueryResult:
    """Per place and OS (the `place` and `run_os` of the `test_run` start): `runs` (ended test
    runs), `outcomes` (each outcome that happened, in the closed set's order), the median
    `run_minutes` of the ended passed and failed runs, and, for `github`, the median minutes of
    the `github_queue` waits whose `on` is a run of that place and OS. A median is `n/a` when
    nothing was measured, the queue median for any other place always. An ended run whose start
    is not in the store has no place and is left out. `repo` keeps that repo's events."""

    where, params = "", {}
    if repo is not None:
        where, params = " AND repo = :repo", {"repo": repo}
    placed = {}  # (repo, run) -> (place, os)
    for run_repo, run, place, run_os in store.query(
            "SELECT repo, run, place, run_os FROM events WHERE kind = 'step.started'"
            f" AND step = 'test_run' AND run IS NOT NULL AND place IS NOT NULL{where} ORDER BY at, id",
            params).rows:
        placed.setdefault((run_repo, run), (place, run_os))

    groups = {}  # (place, os) -> {"outcomes": [...], "minutes": [...], "queue": [...]}
    seen = set()
    for run_repo, run, outcome, minutes in store.query(
            "SELECT repo, run, outcome, run_minutes FROM events WHERE kind = 'step.ended'"
            f" AND step = 'test_run' AND run IS NOT NULL{where} ORDER BY at, id", params).rows:
        key = (run_repo, run)
        if key not in placed or key in seen:
            continue
        seen.add(key)
        group = groups.setdefault(placed[key], {"outcomes": [], "minutes": [], "queue": []})
        group["outcomes"].append(outcome)
        if outcome in _MEASURED_OUTCOMES and isinstance(minutes, (int, float)):
            group["minutes"].append(float(minutes))

    ends = {}  # (repo, wait run) -> the times its wait ended
    for run_repo, run, at in store.query(
            "SELECT repo, run, at FROM events WHERE kind = 'step.ended' AND step = 'wait'"
            f" AND run IS NOT NULL{where} ORDER BY at, id", params).rows:
        ends.setdefault((run_repo, run), []).append(at)
    for run_repo, run, on, at in store.query(
            "SELECT repo, run, \"on\", at FROM events WHERE kind = 'step.started' AND step = 'wait'"
            f" AND reason = 'github_queue' AND run IS NOT NULL{where} ORDER BY at, id", params).rows:
        target = placed.get((run_repo, on))
        end = next((e for e in ends.get((run_repo, run), []) if e >= at), None)
        if target is None or target[0] != "github" or end is None or target not in groups:
            continue
        waited = datetime.strptime(end, _AT_FORMAT) - datetime.strptime(at, _AT_FORMAT)
        groups[target]["queue"].append(waited.total_seconds() / 60)

    order = schema.outcomes()
    rows = []
    for (place, run_os), group in sorted(groups.items(), key=lambda g: (
            _place_order(g[0][0]), g[0][1] is None, g[0][1] or "")):
        counts = {name: group["outcomes"].count(name) for name in order}
        rows.append((place, run_os or _NOT_MEASURED, len(group["outcomes"]),
                     " ".join(f"{name}={n}" for name, n in counts.items() if n),
                     _median_cell(group["minutes"]),
                     _median_cell(group["queue"]) if place == "github" else _NOT_MEASURED))
    return QueryResult(list(TEST_PLACES_COLUMNS), rows)


# A query the ticket names, not a test: pytest must not collect it where a module imports it.
test_places.__test__ = False
