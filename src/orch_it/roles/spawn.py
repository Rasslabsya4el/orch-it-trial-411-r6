"""Spawn (#399 decision 3): the one way a role's subagent is started. `spawn` fills the role's inputs
(the `inputs` of the roles settings) from one place, refuses with the input's name when one is missing,
writes the `subagent` step, and returns the prompt the caller passes to the Agent tool as it is, with
the role's name as the subagent type.

The inputs:
- `issue`: the ticket `#N` (a positive number) of `repo`, fetched with `gh issue view` (body, labels and
  every comment) into `issue-<N>.md` of the spawn's folder; a ticket `gh` cannot read is
  `issue_unreadable`;
- `unit`: a unit id of ORCH's plan, a data row of a table headed `| unit | ...` whose first cell is the id, from the
  newest comment that has one, else the body (two rows of the id in one text are `unit_ambiguous`);
  the row goes into the prompt, and an id with no row is `unit_not_found`;
- `diff`: a diff file, or a git range of the worktree (`main...HEAD`) that spawn writes to `diff.patch`;
  one that cannot be read is `diff_unreadable`, an empty one `diff_empty` (a review of nothing is no
  review: #380 counted two "0 findings" reviews made with no diff);
- `worktree`: the git work tree the subagent works in, default the one around the folder spawn runs
  in; a folder that is no work tree is `worktree_not_git`;
- `scratchpad`: the folder of the spawn's files (`<scratchpad>/<run>/`), default `spawn/` of orch-it's
  data folder; one inside a git work tree is `scratchpad_in_git_tree`, since those files must never
  reach a commit, and one that cannot be made a folder is `scratchpad_unusable`. ORCH passes its
  session's scratchpad, a folder its subagents may read. Every path the prompt names is absolute.

An input the role needs and the request lacks is `input_missing`; an input the role does not take is
left out of its prompt. Each refusal of an input names it (`SpawnRefused.input`); `repo_unknown`
(no `--repo` and no GitHub origin) names none. Every spawn of a role that has inputs and a known repo
writes `step.started` of step `subagent` (run `spawn-<id>`, the role, the ticket, `prompt_has`: the
inputs filled); a refused one also writes its `step.ended` with outcome `refused`, `refused` (the
code) and `refused_input`. The prompt's line `Spawn: <run>` lets the stop hold find the step again.
The brain, with no inputs, is the owner's chat: `role_not_spawned`.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from orch_it import settings
from orch_it.events.ids import live_id
from orch_it.events.store import Store
from orch_it.roles import INPUTS, Role, RoleSet

REFUSAL_CODES = ("role_unknown", "role_not_spawned", "input_missing", "issue_unreadable", "unit_not_found",
                 "unit_ambiguous", "diff_unreadable", "diff_empty", "worktree_not_git", "scratchpad_in_git_tree",
                 "scratchpad_unusable", "repo_unknown")
GIT_DIFF_TIMEOUT_S = 60
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ISSUE_FIELDS = "number,title,url,state,body,labels,comments"
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_SEPARATOR_ROW = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
_MAX_ISSUE = 2**63 - 1
_RANGE = re.compile(r"[\w./@^~{}-]+(?:\.\.\.?[\w./@^~{}-]+)?")
_LABELS = {"issue": "Issue file", "unit": "Unit", "diff": "Diff file", "worktree": "Worktree",
           "scratchpad": "Scratchpad"}


class SpawnRefused(Exception):
    """A spawn that cannot start: `code` one of `REFUSAL_CODES`, `input` the input it is about
    (empty for the role's own refusals and `repo_unknown`)."""

    def __init__(self, code: str, input_: str = "", detail: str = ""):
        if code not in REFUSAL_CODES:
            raise ValueError(f"not a spawn refusal code: {code}")
        self.code = code
        self.input = input_
        said = ": ".join(part for part in (code, input_, detail) if part)
        super().__init__(said)


@dataclass(frozen=True)
class SpawnRequest:
    issue: int | None = None
    repo: str | None = None
    unit: str | None = None
    diff: str | None = None
    worktree: Path | None = None
    scratchpad: Path | None = None
    cwd: Path | None = None  # where spawn runs; the default worktree is the work tree around it

    def but(self, **changes) -> SpawnRequest:
        """The same request with `changes`."""

        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class Spawned:
    run: str
    role: str
    prompt: str
    files: dict[str, Path]  # the files written for the subagent, by input


def default_gh() -> Callable[[list[str]], str]:
    """The real `gh` runner (bounded by a timeout, no console window); the tests replace it."""

    from orch_it.testruns.github import run_gh

    return run_gh


def _work_tree_around(path: Path) -> Path | None:
    path = path.resolve()
    for folder in (path, *path.parents):
        if (folder / ".git").exists():
            return folder
    return None


def _issue_text(raw: dict) -> str:
    labels = ", ".join(str(label.get("name", "")) for label in raw.get("labels") or [] if isinstance(label, dict))
    parts = [f"# #{raw.get('number')} {raw.get('title', '')}", "",
             f"{raw.get('url', '')} ({raw.get('state', '')}; labels: {labels or 'none'})", "", str(raw.get("body") or "")]
    for comment in raw.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        author = (comment.get("author") or {}).get("login", "?")
        parts += ["", f"## Comment by {author}, {comment.get('createdAt', '')}", "", str(comment.get("body") or "")]
    return "\n".join(parts).rstrip() + "\n"


def _first_cell(row: str) -> str:
    return row.strip().strip("|").split("|")[0].strip().strip("`*").strip()


def _data_rows(text: str) -> list[str]:
    """The data rows of the plan tables of `text`: tables whose header's first cell is `unit`, not
    a separator or a fenced line."""

    rows, header, fence = [], None, None
    for line in text.replace("\r\n", "\n").split("\n"):
        opened = re.match(r"^\s*(```+|~~~+)", line)
        if opened:
            fence = None if fence and opened.group(1)[0] == fence[0] else (fence or opened.group(1))
            header = None
            continue
        if fence or not _TABLE_ROW.match(line):
            header = None
            continue
        if header is None:
            header = _first_cell(line).lower()  # the first line of a table is its header
        elif header == "unit" and not _SEPARATOR_ROW.match(line):
            rows.append(line.strip())
    return rows


def _unit_row(raw: dict, unit: str) -> str | None:
    """The plan's row of `unit`: a data row whose first cell is the id (backticks and bold aside),
    from the newest comment that has one, else from the body. Two rows of the id in one text are
    `unit_ambiguous` (raised as `LookupError`)."""

    comments = [str(c.get("body") or "") for c in raw.get("comments") or [] if isinstance(c, dict)]
    for source in [*reversed(comments), str(raw.get("body") or "")]:
        found = [row for row in _data_rows(source) if _first_cell(row) == unit]
        if len(found) > 1:
            raise LookupError(f"{len(found)} rows of `{unit}` in one text of the ticket")
        if found:
            return found[0]
    return None


@dataclass
class _Spawn:
    """One spawn on its way: what is filled so far, and the step it writes either way."""

    role: Role
    request: SpawnRequest
    store: Store
    run: str = field(default_factory=lambda: f"spawn-{live_id()}")
    worktree: Path | None = None
    repo: str | None = None
    filled: dict[str, str] = field(default_factory=dict)  # input -> the value the prompt names
    files: dict[str, Path] = field(default_factory=dict)

    def refused(self, code: str, input_: str = "", detail: str = "") -> SpawnRefused:
        """The refusal, its `subagent` pair written."""

        self.record(None, refused=(code, input_))
        return SpawnRefused(code, input_, detail)

    def record(self, prompt: str | None, refused: tuple[str, str] | None = None) -> None:
        """The `subagent` step; with no repo there is no envelope, and nothing is written."""

        if self.repo is None:
            return
        try:
            self._write(prompt, refused)
        except Exception:  # noqa: BLE001 - a store that cannot be written never hides the spawn's own answer
            pass

    def _write(self, prompt: str | None, refused: tuple[str, str] | None) -> None:
        request = self.request
        envelope = {"repo": self.repo, "ticket": request.issue if request.issue and request.issue > 0 else None,
                    "role": self.role.name, "session": None, "parent_session": None, "turn": None,
                    "step": "subagent", "run": self.run, "note": None}
        started = {"model": self.role.model, "effort": self.role.effort,
                   "agent_file": f".claude/agents/{self.role.name}.md",
                   "title": f"{self.role.name} #{request.issue}" + (f" unit {request.unit}" if request.unit else ""),
                   "launched_by": "session", "unit": request.unit,
                   "worktree": str(self.worktree) if self.worktree else None,
                   "prompt_has": [name for name in INPUTS if name in self.filled], "prompt_chars": len(prompt or "")}
        with self.store.writer() as writer:
            writer.write(envelope | {"kind": "step.started", "detail": started})
            if refused:
                ended = {"outcome": "refused", "refused": refused[0]}
                if refused[1]:
                    ended["refused_input"] = refused[1]
                writer.write(envelope | {"kind": "step.ended", "detail": ended})


def _diff_bytes(diff: str, worktree: Path | None) -> bytes:
    """The diff `diff` names: a file's bytes, or `git diff <range>` of the worktree; raises the
    reason it cannot be read as `diff_unreadable` text."""

    path = Path(diff)
    if path.is_file():
        try:
            return path.read_bytes()
        except OSError as e:
            raise ValueError(str(e)) from None
    if path.exists():
        raise ValueError(f"{diff} is a folder")
    if worktree is None or diff.startswith("-") or not _RANGE.fullmatch(diff):
        raise ValueError(f"{diff!r} is no file and no git range")
    try:
        done = subprocess.run(["git", "-C", str(worktree), "diff", diff, "--"], capture_output=True,
                              timeout=GIT_DIFF_TIMEOUT_S, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(str(e)) from None
    if done.returncode != 0:
        raise ValueError(done.stderr.decode("utf-8", "replace").strip()[:200])
    return done.stdout


def spawn(role: str, request: SpawnRequest, *, roles: RoleSet, store: Store,
          gh: Callable[[list[str]], str] | None = None) -> Spawned:
    """Fill `role`'s inputs from `request`, write the `subagent` step, and return the prompt; or
    raise `SpawnRefused` naming the input that is missing or cannot be used (the step is then
    written ended `refused`). `gh` runs one `gh` call (default: the real CLI)."""

    from orch_it.testruns.github import GithubError

    gh = gh or default_gh()
    if role not in roles.roles:
        raise SpawnRefused("role_unknown", detail=role)
    if not roles.roles[role].inputs:
        raise SpawnRefused("role_not_spawned", detail=f"{role} is a session, not a subagent")
    from orch_it.events.schema import is_repo

    needs = roles.roles[role].inputs
    job = _Spawn(roles.roles[role], request, store)
    if request.repo is not None and (not is_repo(request.repo) or request.repo.startswith("-")):
        raise SpawnRefused("repo_unknown", detail=f"{request.repo!r} is no owner/name")
    job.repo = request.repo
    job.worktree = (request.worktree or _work_tree_around(request.cwd or Path.cwd()))
    if job.worktree is not None:
        job.worktree = job.worktree.absolute()
        if request.worktree is not None and (not str(request.worktree).strip()
                                             or _work_tree_around(request.worktree) != request.worktree.resolve()):
            job.worktree = None
            raise job.refused("worktree_not_git", "worktree", str(request.worktree))
        job.filled["worktree"] = str(job.worktree)
        if job.repo is None:
            from orch_it.tickets.hook import repo_of

            job.repo = repo_of(str(job.worktree))
    if request.scratchpad is not None and not str(request.scratchpad).strip():
        raise job.refused("scratchpad_unusable", "scratchpad", "an empty path")
    scratchpad = (request.scratchpad or settings.data_folder() / "spawn").absolute()
    if _work_tree_around(scratchpad) is not None:
        raise job.refused("scratchpad_in_git_tree", "scratchpad", str(scratchpad))
    folder = scratchpad / job.run
    job.filled["scratchpad"] = str(folder)
    unit = request.unit.strip() if isinstance(request.unit, str) and request.unit.strip() else None
    given = {"issue": request.issue, "unit": unit, "diff": request.diff, "worktree": job.worktree}
    for name in INPUTS:
        if name in needs and given.get(name, "") is None:
            raise job.refused("input_missing", name)
    if job.repo is None:
        raise SpawnRefused("repo_unknown", detail="give --repo")
    if "issue" in needs and not 0 < request.issue <= _MAX_ISSUE:
        raise job.refused("issue_unreadable", "issue", f"#{request.issue} is no ticket number")
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise job.refused("scratchpad_unusable", "scratchpad", str(e)) from None
    title, unit_row = "", None
    if "issue" in needs:
        try:
            raw = json.loads(gh(["issue", "view", str(request.issue), "--repo", job.repo, "--json", _ISSUE_FIELDS]))
            if not isinstance(raw, dict) or raw.get("number") != request.issue:
                raise ValueError("not the ticket asked for")
        except (GithubError, ValueError, TypeError) as e:
            raise job.refused("issue_unreadable", "issue", f"#{request.issue}: {e}"[:300]) from None
        text = _issue_text(raw)
        title = str(raw.get("title") or "")
        job.files["issue"] = folder / f"issue-{request.issue}.md"
        job.files["issue"].write_bytes(text.encode("utf-8", "replace"))
        job.filled["issue"] = str(job.files["issue"])
        if "unit" in needs:
            try:
                unit_row = _unit_row(raw, unit)
            except LookupError as e:
                raise job.refused("unit_ambiguous", "unit", str(e)) from None
            if unit_row is None:
                raise job.refused("unit_not_found", "unit", f"no row `{unit}` in the plan of #{request.issue}")
            job.filled["unit"] = f"{unit}, the plan's row: {unit_row}"
    if "diff" in needs:
        try:
            content = _diff_bytes(request.diff, job.worktree)
        except ValueError as e:
            raise job.refused("diff_unreadable", "diff", str(e)) from None
        if not content.strip():
            raise job.refused("diff_empty", "diff", request.diff)
        job.files["diff"] = folder / "diff.patch"
        job.files["diff"].write_bytes(content)
        job.filled["diff"] = str(job.files["diff"])
    lines = [f"You are the {role} of the orchestration layer: follow `.claude/agents/{role}.md`, step by step.",
             f"Spawn: {job.run}"]
    if "issue" in needs:
        lines.append(f"Ticket: #{request.issue} {title}".rstrip())
    lines += [f"{_LABELS[name]}: {job.filled[name]}" for name in needs]
    report = job.role.report  # the rules the stop hold reads, named here so the role file names no number
    if report and report.cap_lines:
        lines.append(f"Report cap: {report.cap_lines} lines")
    if report and report.owes:
        lines.append(f"Report lines owed: {', '.join(report.owes)}")
    if report and report.context_line:
        lines.append(f"Context line: {report.context_line} tokens, then hand back with `table: <path>`")
    lines.append("Every file above was written at spawn for you: read each one whole before your first step.")
    prompt = "\n".join(lines) + "\n"
    job.record(prompt)
    return Spawned(job.run, role, prompt, job.files)
