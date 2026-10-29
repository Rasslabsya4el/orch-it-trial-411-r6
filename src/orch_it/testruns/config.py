"""The settings of test runs (#394 decisions 1-5), read from the installation's settings file.

Keys (each optional): `test_command` (argv list; `{python}`, `{report}`, `{ids_file}` and
`{work_tree}` are filled per run; default pytest writing `--junitxml`), `test_run_bound_minutes`
(default 15), `test_env` (name -> value), `github_tests` (default off), `test_machines` (one
entry per machine place: `id`, `host`, `os`, `checkout`, optional `capacity` (1), `python`,
`on` (true), `origin` (default: the caller work tree's `origin`), `ssh_options` (list of extra
`ssh` arguments such as `["-p", "2222"]`)), `own_capacity` (1) and `github_capacity` (20, GitHub's
concurrent job cap on a free account). A key under `repos.<owner/name>` (any case) overrides the same key for that repo.
A value of the wrong shape (a NUL in a command or `test_env` value, a bound over a week
included) raises `SettingsError("bad_setting")` naming the key.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from orch_it.settings import SettingsError

DEFAULT_COMMAND = ("{python}", "-m", "pytest", "-q", "-p", "orch_it_ids", "--junitxml={report}")
OSES = ("linux", "windows", "macos")
MAX_BOUND_MINUTES = 7 * 24 * 60  # a week: a longer bound is no bound
_MACHINE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Machine:
    id: str
    host: str
    os: str
    checkout: str
    capacity: int = 1
    python: str | None = None  # default: `python` on Windows, `python3` elsewhere
    on: bool = True
    origin: str | None = None  # the url the machine clones and fetches from; default: the caller's origin
    ssh_options: tuple[str, ...] = ()  # extra `ssh` arguments before the host (-p, -i, -o ...)

    @property
    def place(self) -> str:
        return f"machine:{self.id}"


@dataclass(frozen=True)
class TestSettings:
    __test__ = False  # not a pytest class, whatever its name starts with

    command: tuple[str, ...] = DEFAULT_COMMAND
    bound_minutes: float = 15.0
    test_env: dict[str, str] = field(default_factory=dict)
    github_tests: bool = False
    machines: tuple[Machine, ...] = ()
    own_capacity: int = 1
    github_capacity: int = 20

    @classmethod
    def from_settings(cls, raw: dict, repo: str) -> TestSettings:
        merged = dict(raw)
        repos = raw.get("repos", {})
        if not isinstance(repos, dict):
            raise SettingsError("bad_setting", "repos")
        # GitHub's owner/name is the same repo whatever its case
        own = next((value for key, value in repos.items() if str(key).lower() == repo.lower()), {})
        if not isinstance(own, dict):
            raise SettingsError("bad_setting", f"repos.{repo}")
        merged |= own
        values = {}
        if "test_command" in merged:
            command = merged["test_command"]
            if not (isinstance(command, list) and command
                    and all(isinstance(a, str) and "\x00" not in a for a in command)):
                raise SettingsError("bad_setting", "test_command")
            values["command"] = tuple(command)
        if "test_run_bound_minutes" in merged:
            values["bound_minutes"] = _positive("test_run_bound_minutes", merged["test_run_bound_minutes"])
        if "test_env" in merged:
            env = merged["test_env"]
            if not (isinstance(env, dict) and all(isinstance(k, str) and _ENV_NAME.fullmatch(k)
                                                  and isinstance(v, str) and "\x00" not in v
                                                  for k, v in env.items())):
                raise SettingsError("bad_setting", "test_env")
            values["test_env"] = dict(env)
        if "github_tests" in merged:
            if not isinstance(merged["github_tests"], bool):
                raise SettingsError("bad_setting", "github_tests")
            values["github_tests"] = merged["github_tests"]
        for key, name in (("own_capacity", "own_capacity"), ("github_capacity", "github_capacity")):
            if key in merged:
                values[name] = _count(key, merged[key])
        if "test_machines" in merged:
            values["machines"] = _machines(merged["test_machines"])
        return cls(**values)


def _positive(key: str, value: object) -> float:
    """A bound in minutes: more than 0, at most `MAX_BOUND_MINUTES`."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError("bad_setting", key)
    try:
        minutes = float(value)
    except OverflowError:
        raise SettingsError("bad_setting", key) from None
    if not (math.isfinite(minutes) and 0 < minutes <= MAX_BOUND_MINUTES):
        raise SettingsError("bad_setting", key)
    return minutes


def _count(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SettingsError("bad_setting", key)
    return value


def _machines(entries: object) -> tuple[Machine, ...]:
    if not isinstance(entries, list):
        raise SettingsError("bad_setting", "test_machines")
    machines, seen = [], set()
    for n, entry in enumerate(entries):
        where = f"test_machines[{n}]"
        if not isinstance(entry, dict):
            raise SettingsError("bad_setting", where)
        for key in ("id", "host", "os", "checkout"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise SettingsError("bad_setting", f"{where}.{key}")
        if entry["host"].startswith("-"):  # would read as an ssh option
            raise SettingsError("bad_setting", f"{where}.host")
        if not _MACHINE_ID.fullmatch(entry["id"]) or entry["id"] in seen:
            raise SettingsError("bad_setting", f"{where}.id")
        if entry["os"] not in OSES:
            raise SettingsError("bad_setting", f"{where}.os")
        python = entry.get("python")
        if python is not None and (not isinstance(python, str) or not python.strip()):
            raise SettingsError("bad_setting", f"{where}.python")
        origin = entry.get("origin")
        if origin is not None and (not isinstance(origin, str) or not origin.strip()):
            raise SettingsError("bad_setting", f"{where}.origin")
        options = entry.get("ssh_options", [])
        if not (isinstance(options, list) and all(isinstance(o, str) and o and "\x00" not in o
                                                  for o in options)):
            raise SettingsError("bad_setting", f"{where}.ssh_options")
        on = entry.get("on", True)
        if not isinstance(on, bool):
            raise SettingsError("bad_setting", f"{where}.on")
        seen.add(entry["id"])
        machines.append(Machine(entry["id"], entry["host"], entry["os"], entry["checkout"],
                                _count(f"{where}.capacity", entry.get("capacity", 1)), python, on,
                                origin, tuple(options)))
    return tuple(machines)
