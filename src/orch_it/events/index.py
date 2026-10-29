"""The SQLite index of the store: a cache of the writers' files (#393 decision 2).

Table `events` holds the envelope as columns (`at` in UTC at a fixed width so text order is
time order, `at_local` as written), `detail` as JSON and, as generated columns, each detail
field the event list names, so the list's queries run as written. Table `files` holds how far
each file is read and a hash of its head; a refresh reads only what follows, up to the last
complete line, and unites events by `id`: of two lines with one id, the one at the lowest
(file, offset) is the row, so deleting the index and rebuilding gives the same rows. A file
that shrank, changed its head or vanished makes the refresh rebuild from every file. A
complete line that is no valid event lands in `bad_lines` with its code. A different
closed-set file, or an index SQLite cannot read, rebuilds the cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from orch_it.events import schema

BUSY_TIMEOUT_S = 60
HEAD_BYTES = 1024
BOM = b"\xef\xbb\xbf"
_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                 getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


class IndexError_(Exception):
    """`code`: `query_failed` (the SQL failed or tried more than reading), `index_busy`
    (another process held the index past the bound)."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class RefreshReport:
    files_read: int
    added: int
    duplicates: int  # lines whose id an earlier line already holds
    bad_lines: int  # complete lines in the files that are no valid event, all files
    pending_bytes: int  # half-written last lines, left for the next refresh
    unreadable_files: int  # files the OS would not open this time, tried again next refresh
    rebuilt: bool


class _BadLine(Exception):
    def __init__(self, code: str):
        self.code = code


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _json_path(name: str) -> str:
    return "'$.\"" + name.replace("'", "''").replace('"', '\\"') + "\"'"


def _stored_columns() -> list[tuple[str, str]]:
    """(column, SQL declaration) of every stored column of `events`, in row order."""

    columns = []
    for name in schema.envelope_fields():
        if name == "id":
            columns.append(("id", "TEXT PRIMARY KEY"))
        elif name == "ticket":
            columns.append(("ticket", "INTEGER"))
        elif name == "at":
            columns += [("at", "TEXT NOT NULL"), ("at_local", "TEXT NOT NULL")]
        else:
            columns.append((name, "TEXT"))
    return columns + [("detail", "TEXT NOT NULL"), ("src", "TEXT NOT NULL"),
                      ("src_offset", "INTEGER NOT NULL")]


def _row(event: dict, src: str, src_offset: int) -> tuple:
    derived = {
        "at": utc_at(event["at"]), "at_local": event["at"], "src": src, "src_offset": src_offset,
        "detail": json.dumps(event["detail"], ensure_ascii=False, separators=(",", ":")),
    }
    return tuple(derived[name] if name in derived else event[name]
                 for name, _ in _stored_columns())


def _schema_sql() -> list[str]:
    stored = [f"{_quote(name)} {declaration}" for name, declaration in _stored_columns()]
    taken = {name for name, _ in _stored_columns()}
    generated = [
        f"{_quote(name)} GENERATED ALWAYS AS (json_extract(detail, {_json_path(name)})) VIRTUAL"
        for name in schema.detail_field_names() if name not in taken
    ]
    columns = ",\n  ".join(stored + generated)
    return [
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        "CREATE TABLE files (path TEXT PRIMARY KEY, read_to INTEGER NOT NULL, head TEXT NOT NULL)",
        "CREATE TABLE bad_lines (path TEXT NOT NULL, line_offset INTEGER NOT NULL,"
        " code TEXT NOT NULL, PRIMARY KEY (path, line_offset))",
        f"CREATE TABLE events (\n  {columns}\n)",
        "CREATE INDEX events_kind_at ON events (kind, at)",
        "CREATE INDEX events_session ON events (session)",
        "CREATE INDEX events_repo_ticket ON events (repo, ticket)",
    ]


def _upsert_sql() -> str:
    names = [name for name, _ in _stored_columns()]
    updates = ", ".join(f"{_quote(n)} = excluded.{_quote(n)}" for n in names if n != "id")
    return (f"INSERT INTO events ({', '.join(map(_quote, names))})"
            f" VALUES ({', '.join('?' * len(names))})"
            f" ON CONFLICT (id) DO UPDATE SET {updates}"
            " WHERE (excluded.src, excluded.src_offset) < (events.src, events.src_offset)")


def _schema_key() -> str:
    """The table definitions and the closed sets together: a change of either rebuilds the
    index, so every line is checked again against the sets in force."""

    closed_sets = json.dumps(schema._sets(), sort_keys=True)
    return hashlib.sha256("\n".join([*_schema_sql(), closed_sets]).encode("utf-8")).hexdigest()


def utc_at(at: str) -> str:
    return schema.parse_at(at).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _connect(index_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        target, uri = f"{index_path.absolute().as_uri()}?mode=ro", True
    else:
        target, uri = str(index_path), False
    connection = sqlite3.connect(target, timeout=BUSY_TIMEOUT_S, isolation_level=None, uri=uri)
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_S * 1000}")
    return connection


def _busy(error: sqlite3.OperationalError) -> bool:
    return "locked" in str(error) or "busy" in str(error)


def _reset(connection: sqlite3.Connection) -> None:
    for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
        connection.execute(f"DROP TABLE IF EXISTS {_quote(name)}")
    for statement in _schema_sql():
        connection.execute(statement)
    connection.execute("INSERT INTO meta VALUES ('schema', ?)", (_schema_key(),))


def _schema_is_current(connection: sqlite3.Connection) -> bool:
    if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone():
        return False
    row = connection.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
    return bool(row) and row[0] == _schema_key()


def refresh(root: Path, index_path: Path) -> RefreshReport:
    """Bring the index up to the files. An index SQLite cannot read is deleted and rebuilt."""

    try:
        return _refresh(root, index_path)
    except sqlite3.DatabaseError as e:
        if isinstance(e, sqlite3.OperationalError) and _busy(e):
            raise IndexError_("index_busy", str(e)) from None
        index_path.unlink(missing_ok=True)
        return _refresh(root, index_path)


def _refresh(root: Path, index_path: Path) -> RefreshReport:
    connection = _connect(index_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            report = _bring_up(root, connection)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    return report


def _head(path: Path, size: int) -> str:
    with path.open("rb") as f:
        return hashlib.sha256(f.read(min(size, HEAD_BYTES))).hexdigest()


def _bring_up(root: Path, connection: sqlite3.Connection) -> RefreshReport:
    rebuilt = not _schema_is_current(connection)
    if rebuilt:
        _reset(connection)
    known = {path: (read_to, head) for path, read_to, head in
             connection.execute("SELECT path, read_to, head FROM files").fetchall()}
    present: dict[str, tuple[Path, int, str]] = {}
    unreadable = 0
    for path in sorted(root.glob("*/*.jsonl")):
        name = path.relative_to(root).as_posix()
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
            present[name] = (path, size, _head(path, min(size, known.get(name, (size,))[0])))
        except OSError:
            unreadable += 1
            if name in known:
                present[name] = (path, known[name][0], known[name][1])
    changed = any(
        name not in present or present[name][1] < read_to or present[name][2] != head
        for name, (read_to, head) in known.items())
    if changed:
        _reset(connection)
        known, rebuilt = {}, True

    upsert = _upsert_sql()
    files_read = added = duplicates = pending = 0
    for name, (path, size, _) in present.items():
        start = known.get(name, (0, ""))[0]
        if size == start:
            continue
        try:
            read = _read_file(path, name, start, connection, upsert)
        except OSError:
            unreadable += 1
            continue
        files_read += 1
        offset, file_added, file_duplicates, file_pending = read
        added, duplicates, pending = added + file_added, duplicates + file_duplicates, pending + file_pending
        head = _head(path, offset)
        connection.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (name, offset, head))
    bad_lines = connection.execute("SELECT count(*) FROM bad_lines").fetchone()[0]
    return RefreshReport(files_read, added, duplicates, bad_lines, pending, unreadable, rebuilt)


def _read_file(path: Path, name: str, start: int, connection: sqlite3.Connection,
               upsert: str) -> tuple[int, int, int, int]:
    """Index the complete lines of `path` from byte `start`: (read to, added, duplicates,
    pending bytes)."""

    added = duplicates = pending = 0
    offset = start
    with path.open("rb") as f:
        f.seek(start)
        while True:
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                pending = len(raw)
                break
            line_offset, offset = offset, offset + len(raw)
            body = raw.rstrip(b"\r\n")
            if line_offset == 0 and body.startswith(BOM):
                body, line_offset = body[len(BOM):], len(BOM)
            if not body.strip():
                continue
            for event_offset, event in _events_of(name, line_offset, body, connection):
                exists = connection.execute(
                    "SELECT 1 FROM events WHERE id = ?", (event["id"],)).fetchone()
                connection.execute(upsert, _row(event, name, event_offset))
                if exists:
                    duplicates += 1
                else:
                    added += 1
    return offset, added, duplicates, pending


def _events_of(name: str, line_offset: int, body: bytes, connection: sqlite3.Connection):
    """The event of one complete line. A line that is none is recorded in `bad_lines`; when it
    is a torn line with a whole event glued after it (a writer died mid-line, another write
    followed), that event is still read: the first `{"` from which the rest parses."""

    try:
        yield line_offset, _parse(body)
        return
    except _BadLine as bad:
        connection.execute("INSERT OR REPLACE INTO bad_lines VALUES (?, ?, ?)",
                           (name, line_offset, bad.code))
    start = body.find(b'{"', 1)
    while start > 0:
        try:
            yield line_offset + start, _parse(body[start:])
            return
        except _BadLine:
            start = body.find(b'{"', start + 1)


def _parse(body: bytes) -> dict:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise _BadLine("bad_utf8") from None
    try:
        event = json.loads(text)
    except (ValueError, RecursionError):
        raise _BadLine("bad_json") from None
    try:
        schema.validate(event)
    except schema.EventError as e:
        raise _BadLine(e.code) from None
    return event


def _only_reads(action: int, *_args) -> int:
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def query(index_path: Path, sql: str, params: dict | tuple = ()) -> tuple[list[str], list[tuple]]:
    """Run `sql` on a read-only connection that allows reading and nothing else (no ATTACH,
    no PRAGMA, no VACUUM INTO)."""

    try:
        connection = _connect(index_path, read_only=True)
    except sqlite3.Error as e:
        raise IndexError_("query_failed", str(e)) from None
    try:
        connection.set_authorizer(_only_reads)
        cursor = connection.execute(sql, params)
        if cursor.description is None:
            raise IndexError_("query_failed", "the SQL holds no query")
        rows = cursor.fetchall()
        return [d[0] for d in cursor.description], rows
    except sqlite3.OperationalError as e:
        if _busy(e):
            raise IndexError_("index_busy", str(e)) from None
        raise IndexError_("query_failed", str(e)) from None
    except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError) as e:
        raise IndexError_("query_failed", str(e)) from None
    finally:
        connection.close()


def existing_ids(index_path: Path, ids: list[str]) -> set[str]:
    found: set[str] = set()
    unique = list(dict.fromkeys(ids))
    for at in range(0, len(unique), 500):
        chunk = unique[at:at + 500]
        sql = f"SELECT id FROM events WHERE id IN ({', '.join('?' * len(chunk))})"
        found.update(row[0] for row in query(index_path, sql, tuple(chunk))[1])
    return found
