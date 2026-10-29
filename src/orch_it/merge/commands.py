"""The commands of the merge flow (#400):

- `pr open <ticket> --acceptance <file> [--notes <file>] [--branch B] [--base main] [--work-tree .] [--repo R]`: ORCH's
  acceptance step (decision 1, `orch_it.merge.pr_open`): the measured rows checked against the
  ticket's, the branch pushed, one pull request per ticket opened with `Refs #N` (or its own reused),
  the acceptance comment, the ready mark, auto-merge on, the ticket parked on the pull request; prints
  `pr #<n> opened|reused at <head>: <park line>`. `--branch` defaults to the work tree's branch;
- `follow [<pr>] [--event <file>] [--base main] [--repo R] [--role github]`: the ticket follows its
  pull request (decision 3, `orch_it.merge.follow`): the pull request named, or the one a GitHub event
  of the `follow` workflow is about; prints `#<pr> <outcome> ticket=#<n> <detail>` or `skipped: <why>`;
- `merge line [--base main] [--repo R]`: one run of the repo's merge line (decision 2,
  `orch_it.merge.line`); prints `waits: the line of <repo> is held by <holder>` at once when another
  line holds it, then one line per head, `#<pr> <result> waited <minutes> min updates <n> <why>`, and
  `line of <repo> into <base>: merged=<n> left=<n> empty|bound`; a line it could not take is
  `error: lock_busy` or `error: lock_stale`;
- `main check [--branch main] [--repo R] [--role github]`: the red-on-main tickets brought in line
  with the newest ended `tests` run on the branch (decision 4, `orch_it.merge.main_check`); prints
  `main <outcome> at <sha> closed=#<n>,... opened=#<n>,...`;
- `reconcile [--base main] [--repo R] [--role script]`: the scheduled sweep (`orch_it.merge.reconcile`);
  outside GitHub Actions it also brings the workflows' events into the store; prints whether Actions
  are on, then one line per repair (`<repair> #<n>`) and per pull request it could not follow;
- `merge unlock [--repo R]`: remove the repo lock of the merge line, only when it is stale (older
  than `lock_stale_minutes`); prints whose it was and writes `lock.removed`. A lock that is not
  stale is never removed (`error: lock_not_stale`, naming its holder); one it cannot remove whole
  is `error: lock_unwritable` and still names its holder;
- `merge closing-check [--base main] [--head HEAD] [--work-tree .]`: a closing word before a reference
  in the commits of `origin/<base>..<head>`, and in the title and body the environment carries as
  `PR_TITLE` and `PR_BODY` (`orch_it.merge.closing_check`, the home repo's ticket #422); exits 1 and names each hit (its sha
  or `title`/`body`, and the line) when there is one, 0 when there is none.

`--repo` defaults to the work tree's GitHub origin. A refusal prints `error: <code>` and exits 2.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from orch_it import settings
from orch_it.events.store import Store
from orch_it.merge import closing_check
from orch_it.merge import git as merge_git
from orch_it.merge import lock
from orch_it.merge.follow import ROLE as GITHUB, FollowError, follow, pr_of_event
from orch_it.merge.gh import GhPullHub
from orch_it.merge.git import GitError
from orch_it.merge.line import holder_of, run_line
from orch_it.merge.main_check import check_main
from orch_it.merge.reconcile import reconcile
from orch_it.merge.pr_open import ROLE as ORCH, PrError, open_pr
from orch_it.merge.settings import MergeSettings
from orch_it.testruns.github import DEFAULT_REQUIRED_JOBS, REQUIRED_JOBS_SETTING
from orch_it.tickets.commands import OWNER_SETTING
from orch_it.tickets.gh_check import read_event
from orch_it.tickets.park import ROLE as SCRIPT

GROUPS = ("merge", "pr", "follow", "main", "reconcile")
ERRORS = (lock.LockError, PrError, FollowError, closing_check.ClosingError)


def add_parsers(commands, common: argparse.ArgumentParser) -> None:
    repo = argparse.ArgumentParser(add_help=False, parents=[common])
    repo.add_argument("--repo", help="owner/name (default: the work tree's GitHub origin)")
    group = commands.add_parser("merge", help="the merge line").add_subparsers(dest="command", required=True)
    group.add_parser("unlock", parents=[repo], help="remove the repo lock of the merge line when it is stale")
    p = group.add_parser("closing-check", help="refuse a closing word before a reference in a pull request's text")
    p.add_argument("--base", default="main", help="the branch the pull request merges into (default: main)")
    p.add_argument("--head", default="HEAD", help="the pull request's head (default: HEAD)")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p = group.add_parser("line", parents=[repo], help="land the repo's green pull requests one by one")
    p.add_argument("--base", default="main", help="the branch the line merges into (default: main)")
    group = commands.add_parser("pr", help="the pull request of a ticket").add_subparsers(dest="command",
                                                                                          required=True)
    p = group.add_parser("open", parents=[repo], help="ORCH's acceptance step: one pull request per ticket")
    p.add_argument("ticket", type=int)
    p.add_argument("--acceptance", required=True, metavar="FILE",
                   help="the measured rows: a table with the columns row, result, evidence")
    p.add_argument("--notes", metavar="FILE", help="ORCH's report lines for the body of a new pull request")
    p.add_argument("--branch", help="the ticket's branch (default: the work tree's)")
    p.add_argument("--base", default="main", help="the branch it merges into (default: main)")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--role", default=ORCH)

    group = commands.add_parser("main", help="the main line's tests").add_subparsers(dest="command", required=True)
    p = group.add_parser("check", parents=[repo], help="open, keep or close the red-on-main tickets")
    p.add_argument("--branch", default="main", help="the branch whose tests runs are read (default: main)")
    p.add_argument("--role", default=GITHUB)

    p = commands.add_parser("reconcile", parents=[repo], help="the scheduled repair of every mismatch")
    p.add_argument("--base", default="main", help="the branch of the merge line (default: main)")
    p.add_argument("--role", default=SCRIPT)

    p = commands.add_parser("follow", parents=[repo], help="move a ticket as its pull request stands")
    p.add_argument("pr", nargs="?", type=int, help="the pull request (or give --event)")
    p.add_argument("--event", metavar="PATH", help="a GitHub event payload ($GITHUB_EVENT_PATH)")
    p.add_argument("--base", default="main", help="the branch of the merge line (default: main)")
    p.add_argument("--role", default=GITHUB)


def run(args, store_of: Callable[[], Store], repo_of: Callable[[str], str]) -> tuple[str, int]:
    now = datetime.now(UTC)
    if args.group == "pr":
        return _pr_open(args, store_of, repo_of, now), 0
    if args.group == "follow":
        return _follow(args, store_of, repo_of, now), 0
    if args.group == "main":
        return _main_check(args, store_of, repo_of, now), 0
    if args.group == "reconcile":
        return _reconcile(args, store_of, repo_of, now), 0
    if args.command == "closing-check":
        return _closing_check(args)
    if args.command == "line":
        return _line(args, store_of, repo_of), 0
    return _unlock(args, store_of, repo_of, now), 0


def _closing_check(args) -> tuple[str, int]:
    done = closing_check.check(merge_git.work_tree(Path(args.work_tree)), args.base, args.head, os.environ)
    return closing_check.render(done), (1 if done.hits else 0)


def _unlock(args, store_of, repo_of, now: datetime) -> str:
    repo = args.repo or repo_of(".")
    removed = lock.unlock_stale(repo, settings=MergeSettings.from_settings(settings.load(), repo), clock=lambda: now)
    held = removed.age_minutes(now)
    with store_of().writer() as writer:
        writer.write({"kind": "lock.removed", "repo": repo, "ticket": None, "role": SCRIPT, "session": None,
                      "parent_session": None, "turn": None, "step": None, "run": None, "note": None,
                      "detail": {"holder": removed.holder, "since": removed.since.isoformat(),
                                 "held_minutes": round(held, 1) if held != float("inf") else None}})
    return f"removed the stale lock of {repo}: {removed.said()}\n"


def _pr_open(args, store_of, repo_of, now: datetime) -> str:
    repo = args.repo or repo_of(args.work_tree)
    try:
        acceptance = Path(args.acceptance).read_text("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise PrError("acceptance_unreadable", f"{args.acceptance}: {e}") from None
    try:
        notes = Path(args.notes).read_text("utf-8-sig") if args.notes else ""
    except (OSError, UnicodeDecodeError) as e:
        raise PrError("acceptance_unreadable", f"{args.notes}: {e}") from None
    git = merge_git.work_tree(Path(args.work_tree))
    try:
        branch = args.branch or git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    except GitError as e:
        raise PrError("git_failed", str(e)) from None
    done = open_pr(GhPullHub(), store_of(), repo, args.ticket, acceptance=acceptance, branch=branch, base=args.base,
                   git=git, settings=MergeSettings.from_settings(settings.load(), repo), now=now, role=args.role,
                   notes=notes)
    return f"pr #{done.pr} {'reused' if done.reused else 'opened'} at {done.head[:12]}: {done.park}\n"


def _follow(args, store_of, repo_of, now: datetime) -> str:
    repo = args.repo or repo_of(".")
    hub = GhPullHub()
    if (args.pr is None) == (args.event is None):
        raise FollowError("bad_options", "give a pull request or --event, not both")
    number, why = (args.pr, "") if args.pr is not None else pr_of_event(read_event(args.event), hub, repo)
    if number is None:
        return f"skipped: {why}\n"
    raw = settings.load()
    done = follow(hub, store_of(), repo, number, base=args.base, settings=MergeSettings.from_settings(raw, repo),
                  now=now, role=args.role, required=required_jobs(raw), owner=owner_login(raw))
    ticket = f"#{done.ticket}" if done.ticket else "-"
    return f"#{done.pr} {done.outcome} ticket={ticket} {done.detail}".rstrip() + "\n"


def _main_check(args, store_of, repo_of, now: datetime) -> str:
    repo = args.repo or repo_of(".")
    done = check_main(GhPullHub(), store_of(), repo, args.branch, role=args.role)
    opened = ",".join(f"#{n}" for n in done.opened) or "-"
    closed = ",".join(f"#{n}" for n in done.closed) or "-"
    brain = f" brain=#{done.brain[0]}" if done.brain else ""
    return f"main {done.outcome} at {(done.sha or '-')[:12]} closed={closed} opened={opened}{brain}\n"


def _reconcile(args, store_of, repo_of, now: datetime) -> str:
    repo = args.repo or repo_of(".")
    raw = settings.load()
    done = reconcile(GhPullHub(), store_of(), repo, args.base, settings=MergeSettings.from_settings(raw, repo), now=now,
                     role=args.role, required=required_jobs(raw), owner=owner_login(raw),
                     bring_events=os.environ.get("GITHUB_ACTIONS") != "true")
    lines = ["actions=on" if done.actions else "actions=off: the sweep runs here"]
    lines += [f"{code} {what}" for code, what in done.repairs]
    lines += [f"not followed {why}" for why in done.failed]
    if done.imported:
        lines.append(f"brought in {done.imported} events files of the workflows")
    return "".join(f"{line}\n" for line in lines)


def _line(args, store_of, repo_of) -> str:
    repo = args.repo or repo_of(".")
    raw = settings.load()

    def say(holder) -> None:
        print(f"waits: the line of {repo} is held by {holder.said()}", flush=True)

    done = run_line(GhPullHub(), store_of(), repo, args.base, settings=MergeSettings.from_settings(raw, repo),
                    holder=f"{holder_of(repo, args.base)} (pid {os.getpid()})", on_wait=say,
                    required=required_jobs(raw), owner=owner_login(raw))
    lines = [f"#{s.pr} {s.result} waited {s.waited_minutes:g} min updates {s.updates} {s.detail}".rstrip()
             for s in done.served]
    lines.append(f"line of {repo} into {args.base}: merged={done.merged} left={done.left} {done.reason}")
    return "".join(f"{line}\n" for line in lines)


def parks_follower(store: Store, repo: str, now: datetime, role: str) -> Callable[[int], str]:
    """The follow of a pull request for the park reader of `tickets parks`: it moves the ticket
    of a `pr_ended` park whose pull request ended and says what it did."""

    hub, raw = GhPullHub(), settings.load()

    def followed(number: int) -> str:
        done = follow(hub, store, repo, number, settings=MergeSettings.from_settings(raw, repo), now=now, role=role,
                      required=required_jobs(raw), owner=owner_login(raw))
        return f"#{done.pr} {done.outcome} {done.detail}".rstrip()

    return followed


def required_jobs(raw: dict) -> tuple[str, ...]:
    """The required check names (the setting `github_required_jobs`, default `linux`, `windows`)."""

    jobs = raw.get(REQUIRED_JOBS_SETTING, DEFAULT_REQUIRED_JOBS)
    if not isinstance(jobs, list) or not jobs or not all(isinstance(job, str) and job for job in jobs):
        raise settings.SettingsError("bad_setting", REQUIRED_JOBS_SETTING)
    return tuple(jobs)


def owner_login(raw: dict) -> str | None:
    owner = raw.get(OWNER_SETTING)
    return owner if isinstance(owner, str) and owner else None

