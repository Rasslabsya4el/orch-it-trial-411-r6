"""The clean environment of a test process (#394 decision 3, TEST-15).

A test process gets only the OS's base variables, one fixed list per OS (the path, home, temp
and system-root names), plus the setting `test_env`; nothing else of the caller's environment
reaches it. `settings_hash` covers what decides a run's result besides the code: the command,
the bound and `test_env`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping

BASE_NAMES = {
    "windows": ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
                "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA"),
    "linux": ("PATH", "HOME", "TMPDIR"),
    "macos": ("PATH", "HOME", "TMPDIR"),
}


def this_os() -> str:
    """This machine's OS as a test place names it: `linux`, `windows` or `macos`."""

    if sys.platform == "win32":
        return "windows"
    return "macos" if sys.platform == "darwin" else "linux"


def clean_environment(test_env: Mapping[str, str], *, source: Mapping[str, str] | None = None,
                      os_name: str | None = None) -> dict[str, str]:
    """The base variables of `os_name` (default: this OS) found in `source` (default: this
    process's environment), then `test_env` over them. On Windows names match whatever their
    case."""

    source = os.environ if source is None else source
    os_name = os_name or this_os()
    wanted = BASE_NAMES[os_name]
    if os_name == "windows":
        upper = {name.upper(): name for name in source}
        env = {upper[name]: source[upper[name]] for name in wanted if name in upper}
    else:
        env = {name: source[name] for name in wanted if name in source}
    return env | dict(test_env)


def settings_hash(command: list[str] | tuple[str, ...], bound_minutes: float,
                  test_env: Mapping[str, str]) -> str:
    text = json.dumps({"command": list(command), "bound_minutes": bound_minutes,
                       "test_env": dict(test_env)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
