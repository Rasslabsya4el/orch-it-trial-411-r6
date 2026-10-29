"""The settings of one installation: a JSON file in the per-user data folder of the OS.

`ORCH_IT_SETTINGS` names another settings file; `ORCH_IT_STORE` overrides the `store` setting.
A missing file is no settings at all, never an error.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path


class SettingsError(ValueError):
    """`code`: `settings_unreadable` (the settings file is no JSON object of settings),
    `bad_setting` (a setting of the wrong shape; the field names which part), `settings_busy`
    (another writer held the settings lock too long, or the file could not be replaced)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def data_folder() -> Path:
    """The per-user data folder of orch-it on this OS (#393 decision 3)."""

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "orch-it"


def settings_path() -> Path:
    return Path(os.environ.get("ORCH_IT_SETTINGS") or data_folder() / "settings.json")


def load() -> dict:
    path = settings_path()
    for attempt in range(40):  # Windows refuses a read while a writer replaces the file
        try:
            text = path.read_text("utf-8-sig")
            break
        except FileNotFoundError:
            return {}
        except PermissionError as e:
            if attempt == 39:
                raise SettingsError("settings_unreadable", f"{path}: {e}") from None
            time.sleep(0.05)
        except (OSError, UnicodeDecodeError) as e:
            raise SettingsError("settings_unreadable", f"{path}: {e}") from None
    try:
        settings = json.loads(text)
    except ValueError as e:
        raise SettingsError("settings_unreadable", f"{path}: {e}") from None
    if not isinstance(settings, dict):
        raise SettingsError("settings_unreadable", f"{path}: not an object")
    return settings


def save(key: str, value: object) -> Path:
    """Set one setting, keeping the others; the file is replaced whole, never half written, under
    the settings lock, so two writers at once both land."""

    with _locked():
        settings = load()
        settings[key] = value
        return _write(settings)


LOCK_WAIT_SECONDS = 10  # a writer waits this long for another one's lock
LOCK_STALE_SECONDS = 30  # a lock this old was left by a writer that died: it is taken away


@contextlib.contextmanager
def _locked():
    """The settings lock: a folder beside the file, made by one writer at a time."""

    lock = settings_path().with_name(settings_path().name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            os.mkdir(lock)
            break
        except (FileExistsError, PermissionError):  # Windows: a create refused while another writer removes it
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_SECONDS:
                    os.rmdir(lock)
                    continue
            except OSError:
                pass  # taken away by another writer meanwhile: try again
            if time.monotonic() > deadline:
                raise SettingsError("settings_busy", f"{lock}: held longer than {LOCK_WAIT_SECONDS} s") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass


def _write(settings: dict) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", "utf-8")
    for attempt in range(40):  # Windows refuses a replace while a reader holds the file open
        try:
            os.replace(temporary, path)
            return path
        except PermissionError:
            if attempt == 39:
                temporary.unlink(missing_ok=True)
                raise SettingsError("settings_busy", f"{path}: could not be replaced") from None
            time.sleep(0.05)
    return path


def repo_section(raw: dict, repo: str) -> dict:
    """The section `repos.<owner/name>` of the settings (the repo's name in any case), or `{}`.
    Two spellings of one repo, or a section that is no object, raise `bad_setting`."""

    repos = raw.get("repos", {})
    if not isinstance(repos, dict):
        raise SettingsError("bad_setting", "repos")
    spellings = [key for key in repos if str(key).lower() == repo.lower()]
    if len(spellings) > 1:
        raise SettingsError("bad_setting", f"repos: {repo} written {len(spellings)} ways")
    own = repos[spellings[0]] if spellings else {}
    if not isinstance(own, dict):
        raise SettingsError("bad_setting", f"repos.{repo}")
    return own


def save_in_repo(repo: str, key: str, value: object) -> Path:
    """Set one setting of the section `repos.<owner/name>`, keeping every other one; the repo
    keeps the spelling its section already has."""

    with _locked():
        raw = load()
        repo_section(raw, repo)  # a bad section is refused before anything is written
        repos = raw.setdefault("repos", {})
        spelling = next((name for name in repos if str(name).lower() == repo.lower()), repo)
        repos[spelling] = {**repos.get(spelling, {}), key: value}
        return _write(raw)


def store_path() -> Path:
    """Where this machine's store lives: `ORCH_IT_STORE`, else the `store` setting, else
    `<data folder>/events`."""

    chosen = os.environ.get("ORCH_IT_STORE") or load().get("store")
    if chosen is not None and not isinstance(chosen, str):
        raise SettingsError("settings_unreadable", f"store: {chosen!r} is no path")
    return Path(chosen) if chosen else data_folder() / "events"
