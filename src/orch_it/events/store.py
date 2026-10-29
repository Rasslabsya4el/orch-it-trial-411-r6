"""The store: where events are written and read.

Writing (#393 decision 1): each writer appends to its own file
`<store>/<YYYY-MM-DD>/<writer id>.jsonl`, one event per line. No lock, no server; a crashed
writer loses at most its last line; two machines' stores merge by copying the day folders,
since the index unites events by `id`.

Reading (decision 2): `query` runs read-only SQL over the SQLite index `<store>/index.sqlite`,
a cache of the files brought up to date before every query.

Where (decision 3): `Store.open()` takes the `store` setting, default the per-user data
folder; a folder inside a git work tree is refused before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import orch_it
from orch_it import settings
from orch_it.events import index, schema
from orch_it.events.ids import live_id

STORE_ERROR_CODES = (
    "store_in_git_tree",
    "store_path_empty",
    "store_not_a_folder",
    "store_unwritable",
    "query_failed",
    "index_busy",
    "writer_closed",
)


class StoreError(RuntimeError):
    """A store that cannot be used as asked. `code` is one of `STORE_ERROR_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in STORE_ERROR_CODES:
            raise ValueError(f"not a store error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]


def machine_id() -> str:
    """The event list's `machine`: a hash of the host name, the name itself never stored."""

    return hashlib.sha256(socket.gethostname().lower().encode("utf-8")).hexdigest()[:16]


def os_name() -> str:
    return f"{platform.system()} {platform.release()}".strip()


def _git_work_tree_around(path: Path) -> Path | None:
    for folder in (path, *path.parents):
        if (folder / ".git").exists():
            return folder
    return None


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "index.sqlite"

    @classmethod
    def open(cls, root: str | Path | None = None) -> Store:
        """The store at `root` (default: the `store` setting), created when missing. Refuses
        with `store_in_git_tree` a folder inside a git work tree, by its path as given or as
        links resolve it, so events never reach a commit."""

        if root is not None and not str(root).strip():
            raise StoreError("store_path_empty")
        root = Path(root) if root is not None else settings.store_path()
        root = root.expanduser().absolute()
        for candidate in (root, root.resolve()):
            tree = _git_work_tree_around(candidate)
            if tree is not None:
                raise StoreError("store_in_git_tree", f"{root} is inside {tree}")
        if root.exists() and not root.is_dir():
            raise StoreError("store_not_a_folder", str(root))
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StoreError("store_unwritable", f"{root}: {e}") from None
        return cls(root)

    def writer(self) -> Writer:
        """A new writer with its own id (machine hash + a time-ordered random id), so no two
        writers, on this machine or another, ever share a file."""

        return Writer(self.root, f"{machine_id()[:8]}-{live_id()}")

    def refresh(self) -> index.RefreshReport:
        """Bring the index up to the files: new complete lines in, a half-written last line
        left for the next refresh."""

        try:
            return index.refresh(self.root, self.index_path)
        except index.IndexError_ as e:
            raise StoreError(e.code, e.detail) from None

    def query(self, sql: str, params: dict | tuple = ()) -> QueryResult:
        """Run read-only `sql` over the index's table `events` after a refresh: the envelope as
        columns (`at` in UTC, `YYYY-MM-DDTHH:MM:SS.ffffffZ`; `at_local` as written), `detail`
        as JSON, and each detail field the event list names as a column of its own."""

        self.refresh()
        try:
            columns, rows = index.query(self.index_path, sql, params)
        except index.IndexError_ as e:
            raise StoreError(e.code, e.detail) from None
        return QueryResult(columns, rows)

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Which of `ids` the store already holds."""

        self.refresh()
        try:
            return index.existing_ids(self.index_path, ids)
        except index.IndexError_ as e:
            raise StoreError(e.code, e.detail) from None


class Writer:
    """One writer's own file per day. `write` fills the writer's fields (`id`, `at`,
    `machine`, `os`, `layer_version`, an empty `detail`) when absent, validates, then appends
    the event as one line, handed to the OS whole before `write` returns. After a write that
    failed part-way the next event starts on a fresh line, so only the failed one is lost.
    Safe across threads of one process."""

    def __init__(self, root: Path, writer_id: str):
        self.root = root
        self.writer_id = writer_id
        self._file = None
        self._day = None
        self._fresh_line_owed = False
        self._lock = threading.Lock()
        self._closed = False

    def write(self, event: dict) -> dict:
        now = datetime.now().astimezone()
        full = {
            "id": live_id(), "at": now.isoformat(timespec="microseconds"),
            "machine": machine_id(), "os": os_name(), "layer_version": orch_it.__version__,
        } | event
        full.setdefault("detail", {})
        ordered = {name: full[name] for name in schema.envelope_fields() if name in full}
        ordered |= full  # envelope first in the line, then detail and anything validate refuses
        schema.validate(ordered)
        try:
            line = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (ValueError, RecursionError):
            raise schema.EventError("bad_type", "detail") from None
        data = line.encode("utf-8") + b"\n"
        with self._lock:
            if self._closed:
                raise StoreError("writer_closed", self.writer_id)
            file = self._file_for(now.date().isoformat())
            if self._fresh_line_owed:
                data = b"\n" + data
            self._fresh_line_owed = True
            view = memoryview(data)
            while view:
                view = view[file.write(view):]
            self._fresh_line_owed = False
        return ordered

    def _file_for(self, day: str):
        if day != self._day:
            if self._file is not None:
                self._file.close()
            folder = self.root / day
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self.writer_id}.jsonl"
            self._file = open(path, "ab", buffering=0)
            self._day = day
            self._fresh_line_owed = self._ends_mid_line(path)
        return self._file

    @staticmethod
    def _ends_mid_line(path: Path) -> bool:
        with path.open("rb") as f:
            f.seek(0, 2)
            if f.tell() == 0:
                return False
            f.seek(-1, 2)
            return f.read(1) != b"\n"

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
