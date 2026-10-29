"""The `tickets` commands of the command line (#398 decisions 3, 4 and 5):

- `tickets park <n> <kind> <ref> --until <YYYY-MM-DD> [--repo R] [--role script]`: the one way a
  ticket is parked: its `Parked:` line and the `parked` label in one write; prints the line;
- `tickets unpark <n> [--repo R]`: line and label off together, by hand;
- `tickets parks [--repo R]`: the reader: every parked ticket read, acted on and reported;
- `tickets queue ready|sweep [--repo R]`: the tickets of a queue in the order they are taken
  (`orch_it.tickets.queues`; the brain's hand-off issue is never in one);
- `tickets check-github --event <file>`: the shape check of an `issues` event, run by the
  workflow `ticket-shape`; prints `#<n> kept`, `#<n> removed: <codes>` or `skipped`.

`--repo` defaults to the work tree's GitHub origin. A refusal is a `ParkError`: `error: <code>`.
The session hook runs as its own module, `python -m orch_it.tickets.hook`, to start fast.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime

from orch_it import settings
from orch_it.events.store import QueryResult, Store
from orch_it.tickets.gh_check import check_event, read_event
from orch_it.tickets.hub import GhHub
from orch_it.tickets.park import KINDS, ROLE, park, unpark
from orch_it.tickets.queues import QUEUES, queue
from orch_it.tickets.reader import read_parks

OWNER_SETTING = "owner_login"  # whose comment answers an `owner_answer` park; default the repo's owner


def add_parsers(commands, common: argparse.ArgumentParser) -> None:
    group = commands.add_parser("tickets", help="the tickets of the layer").add_subparsers(
        dest="command", required=True)
    ticket = argparse.ArgumentParser(add_help=False, parents=[common])
    ticket.add_argument("--repo", help="owner/name (default: the work tree's GitHub origin)")
    ticket.add_argument("--role", default=ROLE)
    p = group.add_parser("park", parents=[ticket], help="park a ticket on one event until a date")
    p.add_argument("number", type=int)
    p.add_argument("kind", help=f"one of {', '.join(KINDS)}")
    p.add_argument("ref", help="the PR or ticket (#N, owner/repo#N), date, run id or version it waits on")
    p.add_argument("--until", required=True, metavar="YYYY-MM-DD", help="the last day it may wait (UTC)")
    p = group.add_parser("unpark", parents=[ticket], help="take a ticket's park off by hand")
    p.add_argument("number", type=int)
    group.add_parser("parks", parents=[ticket], help="read every parked ticket and act on it")
    p = group.add_parser("queue", parents=[ticket], help="the tickets of a queue, in the order they are taken")
    p.add_argument("name", choices=list(QUEUES), help="ready: what the dispatcher offers; sweep: the brain's sweep")
    p = group.add_parser("check-github", parents=[common], help="judge the ticket of a GitHub issues event")
    p.add_argument("--event", required=True, metavar="PATH", help="the event payload ($GITHUB_EVENT_PATH)")


def run(args, store_of: Callable[[], Store], repo_of: Callable[[], str], render,
        follow_of: Callable[[Store, str, datetime, str], Callable[[int], str]] | None = None) -> str:
    """The command's printed result; the store and the repo are opened only by the commands
    that need them. `follow_of(store, repo, now, role)` gives the reader the follow of a pull
    request (#400), for its `pr_ended` parks."""

    now = datetime.now(UTC)
    if args.command == "check-github":
        return check_event(read_event(args.event), GhHub()).line + "\n"
    if args.command == "park":
        made = park(GhHub(), store_of(), repo_of(), args.number, args.kind, args.ref, args.until, today=now.date(),
                    role=args.role)
        return made.line + "\n"
    if args.command == "unpark":
        taken = unpark(GhHub(), store_of(), repo_of(), args.number, now=now, role=args.role)
        return f"unparked: {taken.line if taken else 'a park that could not be read'}\n"
    if args.command == "queue":
        found = queue(GhHub(), repo_of(), args.name)
        return render(QueryResult(["ticket", "title", "labels"],
                                  [(issue.number, issue.title, ",".join(issue.labels)) for issue in found]))
    raw = settings.load()
    owner = raw.get(OWNER_SETTING)
    hub, store, repo = GhHub(), store_of(), repo_of()
    reports = read_parks(hub, store, repo, now=now, owner=owner if isinstance(owner, str) else None, role=args.role,
                         follow=follow_of(store, repo, now, args.role) if follow_of else None)
    return render(QueryResult(["ticket", "outcome", "park", "detail"],
                              [(r.number, r.outcome, r.line, r.detail) for r in reports]))
