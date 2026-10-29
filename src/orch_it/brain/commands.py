"""The `brain` commands of the command line (orchestration-layer#403 decision 3):

- `brain handoff ensure [--repo R]`: the repo's pinned hand-off issue made so (`handoff.ensure`);
  prints `brain hand-off #<n>: <what it did>` or `nothing to change`;
- `brain handoff write --file F [--ticket N] [--repo R]`: one hand-off, the file's text, posted on
  it (`handoff.write`); prints `hand-off written on #<n>[, pointer on #<ticket>]`.

`--repo` defaults to the work tree's GitHub origin. A refusal is a `HandoffError`: `error: <code>`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from orch_it.brain import handoff
from orch_it.events.store import Store
from orch_it.tickets.hub import GhHub


def add_parsers(commands, common: argparse.ArgumentParser) -> None:
    group = commands.add_parser("brain", help="the brain's hand-off issue").add_subparsers(
        dest="command", required=True)
    home = group.add_parser("handoff", help="the brain's pinned hand-off issue").add_subparsers(
        dest="action", required=True)
    repo = argparse.ArgumentParser(add_help=False, parents=[common])
    repo.add_argument("--repo", help="owner/name (default: the work tree's GitHub origin)")
    home.add_parser("ensure", parents=[repo], help="create, label and pin the hand-off issue when missing")
    p = home.add_parser("write", parents=[repo], help="post one hand-off on the hand-off issue")
    p.add_argument("--file", required=True, metavar="PATH", help="the hand-off's text (UTF-8)")
    p.add_argument("--ticket", type=int, help="the ticket in work: linked, and pointed back to the hand-off")


def run(args, store_of: Callable[[], Store], repo_of: Callable[[], str]) -> str:
    if args.action == "ensure":
        return handoff.ensure(GhHub(), repo_of()).line()
    try:
        text = Path(args.file).read_text("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise handoff.HandoffError("handoff_unreadable", f"{args.file}: {type(e).__name__}") from None
    return handoff.write(GhHub(), store_of(), repo_of(), text, ticket=args.ticket)
