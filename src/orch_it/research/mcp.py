"""The MCP front of the research bridge (#396 decisions 1 and 5): `python -m orch_it research-mcp`,
registered in orch-it's `.mcp.json` as `orch-it-research`, newline-delimited JSON-RPC on stdin and
stdout, standard library only.

Two tools. `research_start(brief, model?, no_reddit?, report_path?, ticket?, repo?)` refuses at
once what the settings cannot serve (the same refusals as the command line, the key's variable
named), then starts a job: the command-line front itself, `python -m orch_it research`, as a
background process with no console window and in its own process group, its brief, log and
answer in `<state>/jobs/<job>/`. A research run takes many minutes, longer than one MCP call may
last; the job outlives a restart of this server. `research_job(job, wait_s?)` answers
`running` (with the run's latest progress from the store), `ended` with the run's whole end
shape plus the report text and its path, or `gone` when the job's process ended with no answer
or the job outlived its bounds (`JOB_GONE_REASONS`). The provider key reaches this server only through `.mcp.json`'s
`env` expansion (`${VAR}`), never from the registry or a file.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from orch_it import settings as installation
from orch_it.events.ids import live_id
from orch_it.events.store import Store, StoreError
from orch_it.research import run
from orch_it.research.settings import ResearchRefused, ResearchSettings
from orch_it.settings import SettingsError

SERVER_NAME = "orch-it-research"
PROTOCOL = "2024-11-05"
MAX_WAIT_SECONDS = 50  # one call stays well under a client's request timeout
SILENT_SECONDS = 3 * 60  # a job of an earlier server whose run wrote nothing this long is lost
_JOB = re.compile(r"[0-9a-z][0-9a-z-]{7,80}")


MCP_ERROR_CODES = ("unknown_job", "bad_argument", "unknown_tool")
# why a job ended with no answer: its process ended without writing one, or it was started by an
# earlier server of this front and is past its bounds
JOB_GONE_REASONS = ("worker_died", "worker_lost")


class McpError(Exception):
    """A tool call that cannot be served. `code` is one of `MCP_ERROR_CODES`: `unknown_job` (no job
    of that id), `bad_argument` (an argument of the wrong type or empty), `unknown_tool`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in MCP_ERROR_CODES:
            raise ValueError(f"not an MCP error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def tools(research: ResearchSettings) -> list[dict]:
    model = {"type": "string", "description": "a model alias of the settings; default by brief size"}
    if research.models:
        model["enum"] = list(research.models)
    return [
        {"name": "research_start",
         "description": ("Start a web research by a research model (it can open Reddit and forums). "
                         "Returns a job at once; the run takes minutes: then call research_job."),
         "inputSchema": {"type": "object", "required": ["brief"], "properties": {
             "brief": {"type": "string", "description": "the whole brief; a `docs/research/<name>.md` path in "
                                                        "backticks names where the report goes"},
             "model": model,
             "no_reddit": {"type": "boolean", "description": "primary sources: no Reddit link required"},
             "report_path": {"type": "string", "description": "write the report here (relative to the work tree)"},
             "ticket": {"type": "integer"},
             "repo": {"type": "string", "description": "owner/name; default the work tree's origin"}}}},
        {"name": "research_job",
         "description": ("A job's state: running (with the run's latest progress), ended (the run's end with the "
                         "report text and its path) or gone (its process ended with no answer)."),
         "inputSchema": {"type": "object", "required": ["job"], "properties": {
             "job": {"type": "string"},
             "wait_s": {"type": "number", "description": f"wait up to this long for the end (max {MAX_WAIT_SECONDS})"}}}},
    ]


class Jobs:
    """The jobs of one work tree: started as background processes, answered from their folders."""

    def __init__(self, env: dict, work_tree: Path):
        self.env = env
        self.work_tree = work_tree
        self.started: dict[str, subprocess.Popen] = {}

    def settings(self) -> ResearchSettings:
        return ResearchSettings.from_settings(installation.load())

    def start(self, arguments: dict) -> dict:
        brief = _argument(arguments, "brief", str, required=True)
        try:
            brief.encode("utf-8")
        except UnicodeEncodeError:
            raise McpError("bad_argument", "brief: not text") from None
        model = _argument(arguments, "model", str)
        report_to = _argument(arguments, "report_path", str)
        repo = _argument(arguments, "repo", str)
        no_reddit = _argument(arguments, "no_reddit", bool) or False
        ticket = _argument(arguments, "ticket", int)
        if ticket is not None and ticket <= 0:
            raise McpError("bad_argument", "ticket")
        research = self.settings()
        ready = run.preflight(brief, settings=research, env=self.env, model=model, work_tree=self.work_tree,
                              report_to=report_to)
        repo = repo or run.repo_of(self.work_tree)

        job = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{live_id()[-8:].lower()}"
        folder = research.state / "jobs" / job
        folder.mkdir(parents=True)
        brief_file = folder / f"{ready.target.stem if ready.target else 'brief'}.md"
        brief_file.write_text(brief, "utf-8", newline="\n")
        argv = [sys.executable, "-m", "orch_it", "research", str(brief_file), "--work-tree", str(self.work_tree),
                "--repo", repo, "--model", ready.alias, "--done-file", str(folder / "done.json"), "--job", job]
        if no_reddit:
            argv.append("--no-reddit")
        if ready.target is not None:
            argv += ["--report", str(ready.target)]
        if ticket is not None:
            argv += ["--ticket", str(ticket)]
        (folder / "job.json").write_text(json.dumps({"started": time.time()}), "utf-8")
        with open(folder / "worker.log", "ab") as log:
            self.started[job] = subprocess.Popen(argv, cwd=self.work_tree, env=self.env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, **_background())
        return {"job": job, "state": "running", "model": ready.alias,
                "report_path": str(ready.target) if ready.target else None,
                "next": f"call research_job with this job (wait_s up to {MAX_WAIT_SECONDS}) until it is not running"}

    def status(self, arguments: dict) -> dict:
        job = arguments.get("job")
        if not isinstance(job, str) or not _JOB.fullmatch(job):
            raise McpError("unknown_job", str(job)[:80])
        folder = self.settings().state / "jobs" / job
        if not (folder / "job.json").is_file():
            raise McpError("unknown_job", job)
        wait = arguments.get("wait_s") or 0
        if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(wait):
            wait = 0.0
        deadline = time.monotonic() + min(max(float(wait), 0.0), MAX_WAIT_SECONDS)
        while True:
            done = _read_json(folder / "done.json")
            if done is not None:
                return self._done(job, done)
            gone = self._gone(job, folder)
            if gone:
                done = _read_json(folder / "done.json")  # it may have just answered
                return self._done(job, done) if done is not None else {
                    "job": job, "state": "gone", "reason": gone, "log_tail": _tail(folder / "worker.log")}
            if time.monotonic() >= deadline:
                return {"job": job, "state": "running", "progress": self._last_event(job, "step.progressed")}
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))

    def _done(self, job: str, done: dict) -> dict:
        """The run's end, with the text of its report (or of its partial report) and its path."""

        out = {"job": job, "state": "ended", **done}
        for key, text_key in (("report_path", "report"), ("partial_path", "partial")):
            if done.get(key):
                try:
                    out[text_key] = Path(done[key]).read_text("utf-8")
                except (OSError, UnicodeDecodeError) as e:
                    out[f"{text_key}_unreadable"] = type(e).__name__
        return out

    def _gone(self, job: str, folder: Path) -> str | None:
        """Why a job with no answer is over (one of `JOB_GONE_REASONS`), or None while it may
        still answer. A job this server started is over when its process is; one an earlier
        server started is over when its run has written nothing for `SILENT_SECONDS` (a live run
        writes a `step.progressed` every 15 s)."""

        process = self.started.get(job)
        if process is not None:
            return "worker_died" if process.poll() is not None else None
        started = (_read_json(folder / "job.json") or {}).get("started")
        last = self._last_event(job, None, at=True)
        newest = max(filter(None, [last, started if isinstance(started, (int, float)) else None]), default=None)
        return "worker_lost" if newest is None or time.time() - newest > SILENT_SECONDS else None

    def _last_event(self, job: str, kind: str | None, *, at: bool = False):
        """The latest event of the job's run (of `kind`, or any): its detail, or with `at` its
        time in epoch seconds; None when the store has none or cannot be read."""

        sql = ("SELECT e.detail, e.at FROM events s JOIN events e ON e.run = s.run"
               " WHERE s.kind = 'step.started' AND s.step = 'research_run' AND json_extract(s.detail, '$.job') = :job"
               + (" AND e.kind = :kind" if kind else "") + " ORDER BY e.at DESC LIMIT 1")
        try:
            rows = Store.open().query(sql, {"job": job, "kind": kind} if kind else {"job": job}).rows
        except (StoreError, SettingsError):
            return None
        if not rows:
            return None
        if at:
            return datetime.fromisoformat(rows[0][1].replace("Z", "+00:00")).timestamp()
        return json.loads(rows[0][0])


def handle(message, jobs: Jobs) -> dict | None:
    """The reply to one JSON-RPC message, or None for a notification or a line that is no request."""

    if not isinstance(message, dict):
        return None
    method, message_id = message.get("method"), message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    if method == "initialize":
        return _reply(message_id, {"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                                   "capabilities": {"tools": {}},
                                   "serverInfo": {"name": SERVER_NAME, "version": "1"}})
    if method == "ping":
        return _reply(message_id, {})
    if method == "tools/list":
        try:
            research = jobs.settings()
        except SettingsError:
            research = ResearchSettings()
        return _reply(message_id, {"tools": tools(research)})
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            if name == "research_start":
                out = jobs.start(arguments)
            elif name == "research_job":
                out = jobs.status(arguments)
            else:
                raise McpError("unknown_tool", str(name))
        except (McpError, ResearchRefused, SettingsError, OSError) as e:
            return _error(message_id, str(e))
        except Exception as e:  # one bad call never ends the server
            return _error(message_id, f"internal: {type(e).__name__}")
        return _reply(message_id, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=1)}]})
    if message_id is None:
        return None
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve(stdin=None, stdout=None, *, env=None, work_tree: Path | None = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for stream in (stdin, stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", newline="\n")
    jobs = Jobs(dict(os.environ if env is None else env), Path(work_tree or Path.cwd()))
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(message, jobs)
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


def _reply(message_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id, text: str) -> dict:
    return _reply(message_id, {"isError": True, "content": [{"type": "text", "text": f"error: {text}"}]})


def _argument(arguments: dict, name: str, kind: type, *, required: bool = False):
    """The tool argument `name` when it is of `kind` (a bool is no int here), None when absent;
    anything else is `bad_argument` naming it."""

    value = arguments.get(name)
    if value is None or (value == "" and not required):
        if required:
            raise McpError("bad_argument", name)
        return None
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)) or (kind is str and not value.strip()):
        raise McpError("bad_argument", name)
    return value


def _background() -> dict:
    """Popen options for a job's process: no console window, its own process group, so neither a
    console's Ctrl+C nor this server's end takes it down."""

    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _read_json(path: Path) -> dict | None:
    try:
        found = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return found if isinstance(found, dict) else None


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text("utf-8", "replace").splitlines()[-lines:])
    except OSError:
        return ""
