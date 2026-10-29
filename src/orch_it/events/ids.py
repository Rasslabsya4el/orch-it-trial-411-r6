"""Event ids (#393 decision 5): time-ordered random for a live writer, a hash of the source
record for an imported event, so importing the same record again yields the same id."""

from __future__ import annotations

import hashlib
import os
import time
import uuid


def live_id() -> str:
    """A UUID version 7: 48 bits of Unix milliseconds, then random bits."""

    ms = time.time_ns() // 1_000_000
    value = (ms & ((1 << 48) - 1)) << 80 | int.from_bytes(os.urandom(10), "big")
    value = (value & ~(0xF << 76)) | (0x7 << 76)
    value = (value & ~(0x3 << 62)) | (0x2 << 62)
    return str(uuid.UUID(int=value))


def import_id(source: str, record_id: str, kind: str) -> str:
    """The id of the event of `kind` read from record `record_id` of `source`."""

    digest = hashlib.sha256("\x1f".join((source, record_id, kind)).encode("utf-8"))
    return "h-" + digest.hexdigest()[:32]
