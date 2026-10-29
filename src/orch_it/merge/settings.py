"""The settings of the merge flow (#400 decision 5), the section `merge` of the installation's
settings file; a section `repos.<owner/name>.merge` (any case) overrides it key by key for that repo.

Keys, each optional:

- `pr_max_hours` (24): a pull request open longer than this, counted from its ready mark, goes to
  the brain;
- `line_minutes` (60): the bound of one run of the merge line, waits included; a run reaching it
  starts no new head and ends;
- `check_minutes` (30): the longest the line waits for the required checks of its head's sha;
- `poll_seconds` (30): how often the line reads its head's state while it waits;
- `max_updates` (3): how many times one run brings its head up to date before it leaves it for the
  next run (main moved under it each time), and how many merges of it GitHub may refuse in a run;
- `lock_wait_minutes` (20): how long a second line waits for the repo lock before it gives up;
- `lock_stale_minutes` (120): a lock older than this is reported stale and never taken; it must be
  longer than `line_minutes`, or a running line's lock would read stale;
- `check_stuck_minutes` (60): a required check pending longer than this is cancelled, so its run
  ends with no result and is run once more;
- `red_in_a_row` (3): the red returns of one ticket in a row that send it to the brain instead of
  back to ready;
- `red_main_park_days` (7): how long a ticket red only on tests red on main waits parked on the
  red-on-main ticket before it goes back to its queue late;
- `merge_method` (`merge`): how the line merges and how `pr open` turns auto-merge on: `merge`,
  `squash` or `rebase`.

A value of the wrong shape, or a key not named here, raises `SettingsError("bad_setting")` naming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from orch_it.settings import SettingsError

SECTION = "merge"
METHODS = ("merge", "squash", "rebase")
WEEK_MINUTES = 7 * 24 * 60  # a longer bound is no bound
# key -> (least, most) in the key's own unit; a poll under a second would hammer GitHub
_RANGES = {"pr_max_hours": (0, WEEK_MINUTES / 60), "poll_seconds": (1, 3600)}
COUNT_MOST = 1000  # `max_updates`, `red_in_a_row`, `red_main_park_days`: a count past this is no bound


@dataclass(frozen=True)
class MergeSettings:
    pr_max_hours: float = 24
    line_minutes: float = 60
    check_minutes: float = 30
    poll_seconds: float = 30
    max_updates: int = 3
    lock_wait_minutes: float = 20
    lock_stale_minutes: float = 120
    check_stuck_minutes: float = 60
    red_in_a_row: int = 3
    red_main_park_days: int = 7
    merge_method: str = "merge"

    @classmethod
    def from_settings(cls, raw: dict, repo: str) -> MergeSettings:
        merged = dict(_section(raw.get(SECTION, {}), SECTION))
        repos = raw.get("repos", {})
        if not isinstance(repos, dict):
            raise SettingsError("bad_setting", "repos")
        # GitHub's owner/name is the same repo whatever its case: two spellings of it are no answer
        spellings = [key for key in repos if str(key).lower() == repo.lower()]
        if len(spellings) > 1:
            raise SettingsError("bad_setting", f"repos: {repo} written {len(spellings)} ways")
        own = repos[spellings[0]] if spellings else {}
        if not isinstance(own, dict):
            raise SettingsError("bad_setting", f"repos.{repo}")
        merged |= _section(own.get(SECTION, {}), f"repos.{repo}.{SECTION}")
        known = {f.name: f for f in fields(cls)}
        values = {}
        for key, value in merged.items():
            if key not in known:
                raise SettingsError("bad_setting", f"{SECTION}.{key}: no such setting")
            if key == "merge_method":
                if value not in METHODS:
                    raise SettingsError("bad_setting", f"{SECTION}.{key}: one of {', '.join(METHODS)}")
            elif key in ("max_updates", "red_in_a_row", "red_main_park_days"):
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= COUNT_MOST:
                    raise SettingsError("bad_setting", f"{SECTION}.{key}: a whole number from 1 to {COUNT_MOST}")
            else:
                value = _bound(key, value, *_RANGES.get(key, (0, WEEK_MINUTES)))
            values[key] = value
        found = cls(**values)
        if found.lock_stale_minutes <= found.line_minutes:
            raise SettingsError("bad_setting", f"{SECTION}.lock_stale_minutes: longer than line_minutes")
        return found


def _section(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise SettingsError("bad_setting", where)
    return value


def _bound(key: str, value: object, least: float, most: float) -> float:
    """A number more than `least` (or `least` itself when it is not 0) and at most `most`."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError("bad_setting", f"{SECTION}.{key}")
    try:
        number = float(value)
    except OverflowError:
        raise SettingsError("bad_setting", f"{SECTION}.{key}") from None
    if not (math.isfinite(number) and (number > 0 if least == 0 else number >= least) and number <= most):
        raise SettingsError("bad_setting", f"{SECTION}.{key}: from {least:g} to {most:g}")
    return value
