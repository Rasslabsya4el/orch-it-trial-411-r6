"""Which text of a run is its report (#396 decision 2).

The old bridges took the last assistant message, and lost a finished report when opencode
compacted the session after it and the model answered the injected "Continue..." turn in one
line (#380). `choose` reads every assistant message of the run instead. The candidates are:

- each assistant message's text (its text parts, one per line), except opencode's compaction
  summary (`summary: true`), which is opencode's digest of the run, never the model's report;
- each chain of texts joined across a compaction that cut a message: the newest assistant
  message before the compaction ended without finishing (`finish` neither `stop` nor
  `tool-calls`), and the next text comes after the injected continue turn (a synthetic user part
  with `metadata.compaction_continue`). A message that finished said all it had and is never
  joined to what follows, whatever the compaction says: joined to it, the #380 reply ("the report
  has been completed in the prior message") would make the longest candidate.

The chosen one is the longest candidate that passes `check`, the newer one on a tie;
`report_from` says where it came from: `last` (the last assistant message, the one the old
bridges took), `earlier` (another single message) or `joined`. With no candidate passing, the
reason is `model_error` when a message carries opencode's `info.error`, else the first failing
test of the longest candidate, else why there is no candidate: `no_reply` (no assistant
message), `empty` (no assistant text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

MIN_CHARS = 3000
MIN_LINKS = 3
CHECK_REASONS = ("too_short", "few_links", "no_reddit")  # in the order the tests run
RUN_REASONS = ("model_error", "no_reply", "empty")
FINISHED = ("stop", "tool-calls")  # a message that ended on its own, not cut
_URL = re.compile(r"https?://[^\s<>\[\]{}\"'`|\\]+", re.IGNORECASE)
_URL_TAIL = ".,;:!?*_~"


@dataclass(frozen=True)
class Report:
    text: str | None  # None: no candidate passed the check
    report_from: str | None  # last, earlier, joined
    reason: str | None  # one of CHECK_REASONS or RUN_REASONS; None when a report was chosen
    chars: int
    links: int
    reddit_urls: int
    compactions: int
    last_chars: int  # the text of the last assistant message: what the old bridges took


def urls(text: str) -> set[str]:
    """The distinct URLs of `text`: trailing punctuation and an unmatched closing parenthesis
    cut (a Markdown link), host lower-cased, fragment and trailing slash dropped."""

    found = set()
    for url in _URL.findall(text):
        while True:
            url = url.rstrip(_URL_TAIL)
            if url.endswith(")") and url.count(")") > url.count("("):
                url = url[:-1]
            else:
                break
        try:
            parts = urlsplit(url)
        except ValueError:
            continue
        if parts.hostname:
            query = f"?{parts.query}" if parts.query else ""
            found.add(f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path.rstrip('/')}{query}")
    return found


def is_reddit(url: str) -> bool:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("reddit.com", "redd.it") or host.endswith(".reddit.com")


def check(text: str, *, require_reddit: bool) -> str | None:
    """None when `text` is a report, else the first test it fails: length (blanks around it not
    counted), then links, then the Reddit rule, so the reason names the real cause (#380 F2)."""

    if len(text.strip()) < MIN_CHARS:
        return "too_short"
    found = urls(text)
    if len(found) < MIN_LINKS:
        return "few_links"
    if require_reddit and not any(map(is_reddit, found)):
        return "no_reddit"
    return None


@dataclass
class _Scan:
    """What one pass over a run's messages found."""

    candidates: list[tuple[str, str]]  # (text, report_from), in the order they arose
    compactions: int = 0
    replies: int = 0  # assistant messages, the summaries not counted
    error: bool = False
    last_text: str = ""


def choose(messages, *, require_reddit: bool) -> Report:
    """The report of a run from its message list (`GET /session/{id}/message`). Anything odd
    (not a list, an item that is no message, a part without text) is read as nothing, never an
    error."""

    scan = _scan(messages if isinstance(messages, list) else [])
    ranked = sorted(enumerate(scan.candidates), key=lambda item: (len(item[1][0]), item[0]), reverse=True)
    for _, (text, where) in ranked:
        if check(text, require_reddit=require_reddit) is None:
            found = urls(text)
            return Report(text, where, None, len(text), len(found), sum(map(is_reddit, found)),
                          scan.compactions, len(scan.last_text))
    if scan.error:
        reason = "model_error"
    elif ranked:
        reason = check(ranked[0][1][0], require_reddit=require_reddit)
    else:
        reason = "empty" if scan.replies else "no_reply"
    return Report(None, None, reason, 0, 0, 0, scan.compactions, len(scan.last_text))


def _scan(messages: list) -> _Scan:
    scan = _Scan([])
    singles: list[tuple[str, int]] = []  # (text, its message's number among the replies)
    chain: list[str] = []  # texts that join across cut compactions, newest last
    finish = None  # of the newest reply
    cut = resumed = False  # a compaction cut the newest reply; the continue turn followed it
    for message in messages:
        info, parts = _info(message), _parts(message)
        if info.get("role") == "user":
            if any(part.get("type") == "compaction" for part in parts):
                scan.compactions += 1
                cut, resumed = finish not in FINISHED, False
            elif parts and all(map(_is_continue, parts)):
                resumed = True
            else:  # a prompt of its own: what follows answers it
                chain, cut, resumed = [], False, False
            continue
        if info.get("role") != "assistant" or info.get("summary") is True:
            continue
        scan.replies += 1
        scan.error = scan.error or bool(info.get("error"))
        finish = info.get("finish")
        text = "\n".join(part["text"] for part in parts
                         if part.get("type") == "text" and isinstance(part.get("text"), str))
        scan.last_text = text
        if not text.strip():
            continue
        singles.append((text, scan.replies))
        if chain and cut and resumed:
            chain.append(text)
            scan.candidates.append(("\n".join(chain), "joined"))
        else:
            chain = [text]
        cut = resumed = False
    scan.candidates[:0] = [(text, "last" if number == scan.replies else "earlier") for text, number in singles]
    return scan


def _info(message) -> dict:
    info = message.get("info") if isinstance(message, dict) else None
    return info if isinstance(info, dict) else {}


def _parts(message) -> list[dict]:
    parts = message.get("parts") if isinstance(message, dict) else None
    return [part for part in parts if isinstance(part, dict)] if isinstance(parts, list) else []


def _is_continue(part: dict) -> bool:
    metadata = part.get("metadata")
    return (part.get("type") == "text" and part.get("synthetic") is True
            and isinstance(metadata, dict) and metadata.get("compaction_continue") is True)
