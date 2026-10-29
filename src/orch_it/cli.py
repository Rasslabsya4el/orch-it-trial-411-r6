"""The command line: `python -m orch_it events <command>` (#393 decisions 8 and 9; #394 decisions
7 and 9 for `import-github` and `test-places`), `python -m orch_it tests run` (#394 decision 9);
`python -m orch_it labels sync` (#398 decision 6); `python -m orch_it roles render` (#399 decision
5); `python -m orch_it guards scan` (#395 decision 4: it prints one line per private value found and exits 1 when there is any), `guards install` and
`guards status` (decisions 5 and 6: the hooks path, and the six facts of the guards), `hook pre-push`
(the git hook that scans a push to a public repo); `python -m orch_it skills
shadowed` (#397 decision 3), a session-start hook that exits 0 whatever happens; `python -m orch_it
research <brief>` and `research-mcp` (#396 decision 1, the research bridge's two fronts); `python -m orch_it
tickets ...` (#398 decisions 4 and 5, in `orch_it.tickets.commands`); `python -m orch_it skills check`,
`skills report` and `skills update` (#397 decision 4); `python -m orch_it merge ...` and the
other commands of the merge flow (#400, in `orch_it.merge.commands`); `python -m orch_it brain handoff ...`
(orchestration-layer#403 decision 3, in `orch_it.brain.commands`).

Commands print a result as an aligned text table (header, then rows). Any coded error of the
store, the settings, the importer, a named query or a command prints `error: <code>` (plus the
field or detail) to stderr and the exit code is 2.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from orch_it import identity, labels, roles, settings
from orch_it.brain import commands as brain_commands
from orch_it.brain.handoff import HandoffError
from orch_it.events import queries, schema, transcripts
from orch_it.events.store import QueryResult, Store, StoreError
from orch_it.guards import hook, install, prepush, scanner
from orch_it.guards import status as guards_status
from orch_it.merge import commands as merge_commands
from orch_it.research import mcp as research_mcp
from orch_it.research import names as research_names
from orch_it.research import run as research_run
from orch_it.research.settings import ResearchRefused, ResearchSettings
from orch_it.roles.spawn import SpawnRefused
from orch_it.skills import shadow, update
from orch_it.skills.copies import SkillsError
from orch_it.skills.forge import GithubForge
from orch_it.testruns import github
from orch_it.testruns.config import TestSettings
from orch_it.testruns.runner import NEED_OSES, RunError, run_tests
from orch_it.tickets import commands as ticket_commands
from orch_it.tickets import hook as ticket_hook
from orch_it.tickets.gh_check import CheckError
from orch_it.tickets.park import ParkError

DEFAULT_REPO = "Rasslabsya4el/orch-it"
EXIT_ERROR = 2
EXIT_INTERRUPTED = 130
# `tests run` exits 0 on passed, 1 on failed, 3 on any other outcome (no result, refused, lost,
# stopped): a script tells "red" from "no answer" without reading the line
TESTS_EXIT = {"passed": 0, "failed": 1}
TESTS_EXIT_OTHER = 3
# `research` exits the same way: 0 passed, 1 failed (a reply that is no report), 3 no result
RESEARCH_EXIT = TESTS_EXIT


_LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
_INTEGER = re.compile(r"-?(0|[1-9][0-9]*)")
_DECIMAL = re.compile(r"-?(0|[1-9][0-9]*)\.[0-9]+")


class CommandError(Exception):
    """A command given a value it cannot use. `code`: `bad_pct` (a weekly percent outside
    0-100), `repo_unknown` (no `--repo` and no GitHub origin in the work tree), `bad_options`
    (`guards scan` options that do not go together)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _param(text: str) -> tuple[str, int | float | str]:
    """`name=value`; the value binds as a number only when written plainly as one (`7`,
    `-2.5`), so `007`, `1_000` or `nan` stay text."""

    name, equals, value = text.partition("=")
    if not equals or not name:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    if _INTEGER.fullmatch(value):
        return name, int(value)
    if _DECIMAL.fullmatch(value):
        return name, float(value)
    return name, value


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m orch_it")
    root.add_argument("--store", metavar="PATH", help="the store folder (default: the `store` setting)")
    commands = root.add_subparsers(dest="group", required=True)
    events = commands.add_parser("events", help="the event store").add_subparsers(
        dest="command", required=True)

    # the store option is also taken after the command; it never overrides one given before it
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", metavar="PATH", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    def command(name, help_text):
        return events.add_parser(name, parents=[common], help=help_text)

    p = command("query", "run read-only SQL over the table `events`")
    p.add_argument("sql")
    p.add_argument("--param", action="append", type=_param, default=[], metavar="NAME=VALUE",
                   help="bind :NAME in the SQL (a number when it reads as one)")

    p = command("ticket-cost", "cost of a ticket from ready to close, per role")
    p.add_argument("--repo", required=True)
    p.add_argument("--ticket", required=True, type=int)

    p = command("session-cost", "$, %% week, tokens and minutes per session and model")
    p.add_argument("--session")
    p.add_argument("--repo")
    p.add_argument("--since", metavar="ISO")
    p.add_argument("--until", metavar="ISO")

    command("limit-rate", "dollars per 1 %% of the weekly limit between readings")

    command("guard-fires", "per guard and week: denies, passes, reversals and failed-open errors")

    p = command("measure-limit", "record a reading of the usage card's weekly percent")
    p.add_argument("pct", type=float)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--role", default="owner")
    p.add_argument("--note")

    p = command("set-limit-rate", "save the setting weekly_limit_usd_per_pct")
    p.add_argument("usd_per_pct", type=float)
    p.add_argument("--plan", required=True)
    p.add_argument("--measured", required=True, metavar="DATE")

    p = command("import", "import a folder of Claude Code transcripts")
    p.add_argument("folder")
    p.add_argument("--repo", required=True)
    p.add_argument("--role", default="worker")
    p.add_argument("--subagent-role")
    p.add_argument("--ticket", type=int)
    p.add_argument("--session")
    p.add_argument("--quiet-minutes", type=float, default=30)

    p = command("import-github", "import the runs of the repo's `tests` workflow from GitHub")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--since", metavar="ISO")

    p = command("test-places", "runs, outcomes and median run and GitHub queue minutes per place and OS")
    p.add_argument("--repo")

    tests = commands.add_parser("tests", help="test runs").add_subparsers(dest="command", required=True)
    p = tests.add_parser("run", parents=[common], help="run the repo's tests at a test place")
    p.add_argument("--os", default="any", choices=NEED_OSES,
                   help="the OS the run needs (default: any)")
    p.add_argument("--ids-file", metavar="PATH", help="run only the tests named in this file")
    p.add_argument("--place", help="force a place (own, machine:<id>, github); default: the place rule")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--repo", help="owner/name (default: from the work tree's origin)")
    p.add_argument("--ticket", type=int)
    p.add_argument("--role", default="script")

    guards = commands.add_parser("guards", help="guards of the public repo").add_subparsers(
        dest="command", required=True)
    p = guards.add_parser("scan", help="look for private data in commits or in a tree")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--range", dest="rev_range", metavar="A..B", help="the commits of a range")
    what.add_argument("--tree", action="store_true", help="every tracked file at --rev")
    what.add_argument("--all", action="store_true", help="every commit reachable from HEAD")
    p.add_argument("--rev", metavar="REV", help="the revision --tree reads (default: HEAD)")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--machine-independent", action="store_true",
                   help="only what holds on any machine: no home, host name or word of this one")
    p.add_argument("--require-noreply-email", action=argparse.BooleanOptionalAction, default=None,
                   help="every commit's author and committer e-mail is a GitHub noreply address, with --range "
                        "(default: on when the repo's setting commit_identity is noreply; --all is never checked)")
    p.add_argument("--allow-email", action="append", default=[], metavar="ADDR",
                   help="an address that is no hit")
    p.add_argument("--allow-email-of", action="append", default=[], metavar="LOGIN",
                   help="allow the public email of this GitHub account")
    p.add_argument("--words-file", metavar="PATH",
                   help="the private words (default: the setting `private_words_file`)")
    p = guards.add_parser("reverse", parents=[common], help="mark a guard deny a false fire")
    p.add_argument("fired_event", metavar="EVENT-ID", help="the `guard.fired` deny to reverse")
    p.add_argument("--by", default="owner", help="who marks it (default: owner)")

    p = guards.add_parser("install", parents=[common],
                          help="point the core.hooksPath setting at the repo's tracked hooks folder")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--repo", help="owner/name (default: from the work tree's origin)")

    p = guards.add_parser("status", parents=[common],
                          help="read main's branch protection and the hooks: the six facts of decision 5")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--repo", help="owner/name (default: from the work tree's origin)")

    hook_commands = commands.add_parser("hook", help="the Claude Code and git hooks").add_subparsers(
        dest="command", required=True)
    hook_commands.add_parser("pre-tool-use", parents=[common],
                             help="the PreToolUse guard; reads the hook's JSON on stdin")
    hook_commands.add_parser("subagent-stop", parents=[common],
                             help="the SubagentStop hold of report lines; reads the hook's JSON on stdin")
    hook_commands.add_parser("brain", parents=[common],
                             help="the brain's SessionStart and UserPromptSubmit hook; reads the hook's JSON on stdin")
    p = hook_commands.add_parser("pre-push", parents=[common],
                                 help="the git pre-push guard; git passes the remote's name and url "
                                      "and writes the refs on stdin")
    p.add_argument("remote_name", nargs="?", default="")
    p.add_argument("remote_url", nargs="?", default="")

    label_commands = commands.add_parser("labels", help="the layer's labels").add_subparsers(
        dest="command", required=True)
    p = label_commands.add_parser("sync", parents=[common],
                                  help="make a repo's labels match the layer's set")
    p.add_argument("--repo", help="owner/name (default: the work tree's GitHub origin)")

    role_commands = commands.add_parser("roles", help="the layer's roles").add_subparsers(
        dest="command", required=True)
    p = role_commands.add_parser("render", help="write each agent file's model and effort from the "
                                                "roles settings")
    p.add_argument("--settings", metavar="PATH", help="a roles settings file (default: roles.json)")
    p.add_argument("--work-tree", default=".", metavar="PATH")

    skills = commands.add_parser("skills", help="the repo's skill copies").add_subparsers(
        dest="command", required=True)
    skills.add_parser("shadowed", parents=[common],
                      help="the session-start hook: a skill.shadowed event per personal skill of a repo skill's name")
    p = skills.add_parser("check", parents=[common],
                          help="fetch his upstream and give each skill its class against the manifest")
    p.add_argument("--ref", default="main", help="his branch (default: main)")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p = skills.add_parser("report", parents=[common],
                          help="the daily run: on a change, one issue per upstream commit with every skill's class")
    p.add_argument("--ref", default="main", help="his branch (default: main)")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--repo", help="owner/name (default: the work tree's GitHub origin)")
    p = skills.add_parser("update", parents=[common],
                          help="run by ORCH on the report's issue: one pull request with the new copies")
    p.add_argument("--ref", default="main", help="his branch (default: main)")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--repo", help="owner/name (default: the work tree's GitHub origin)")

    p = commands.add_parser("research", parents=[common], help="run a brief through the research bridge")
    p.add_argument("brief", help="the brief file (Markdown); the report goes to the path it names")
    p.add_argument("--model", help=f"a model alias of the setting research.models: {_aliases()} "
                                   "(default: by brief size)")
    p.add_argument("--no-reddit", action="store_true", help="primary sources: no Reddit link required")
    p.add_argument("--report", metavar="PATH", help="write the report here instead of the brief's path")
    p.add_argument("--work-tree", default=".", metavar="PATH")
    p.add_argument("--repo", help="owner/name (default: from the work tree's origin)")
    p.add_argument("--ticket", type=int)
    p.add_argument("--done-file", metavar="PATH", help=argparse.SUPPRESS)  # the MCP front's job answer
    p.add_argument("--job", help=argparse.SUPPRESS)  # the MCP front's job id, carried by the started event

    commands.add_parser("research-mcp", help="the research bridge's MCP server on stdin and stdout")
    p = commands.add_parser("research-check-names", help="the names check: the bridge's text against the names its "
                                                         "settings name and the known families")
    p.add_argument("--work-tree", default=".", metavar="PATH", help="the work tree whose bridge text is read")

    ticket_commands.add_parsers(commands, common)
    brain_commands.add_parsers(commands, common)
    merge_commands.add_parsers(commands, common)

    p = commands.add_parser("spawn", parents=[common],
                            help="build a role's subagent prompt from its inputs, or refuse naming the missing one")
    p.add_argument("role", choices=_spawned_roles(), help="a role of the roles settings that is spawned")
    p.add_argument("--issue", type=int, help="the ticket number")
    p.add_argument("--unit", help="the unit id, a row of the plan's unit table in the ticket")
    p.add_argument("--diff", help="a diff file, or a git range of the worktree (main...HEAD)")
    p.add_argument("--worktree", metavar="PATH", help="the git work tree (default: the one around here)")
    p.add_argument("--scratchpad", metavar="PATH", help="the folder of the spawn's files (default: orch-it's data folder)")
    p.add_argument("--repo", help="owner/name (default: the worktree's GitHub origin)")
    return root


def parser() -> argparse.ArgumentParser:
    """The command line's parser: the role-text check reads every `python -m orch_it` command a
    text names against it (#399 decision 4)."""

    return _parser()


def _aliases() -> str:
    """The aliases the settings name, for the help text; a settings file that cannot be read
    leaves the help plain (the command then says what is wrong)."""

    try:
        return ", ".join(ResearchSettings.from_settings(settings.load()).models) or "none set"
    except settings.SettingsError:
        return "see the settings"


def render(result: QueryResult) -> str:
    """The result as text: a header line, then one line per row, columns aligned."""

    table = [list(result.columns)]
    table += [["NULL" if value is None else str(value) for value in row] for row in result.rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(result.columns))]
    return "".join("  ".join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip() + "\n"
                   for line in table)


def _store(args) -> Store:
    return Store.open(args.store)


def _repo_of(work_tree: str) -> str:
    """`owner/name` of the work tree's `origin` on GitHub, or `CommandError("repo_unknown")`."""

    found = ticket_hook.repo_of(work_tree)
    if not found:
        raise CommandError("repo_unknown", f"{work_tree}: give --repo")
    return found


def _tests_run(args) -> tuple[str, int]:
    repo = args.repo or _repo_of(args.work_tree)
    test_settings = TestSettings.from_settings(settings.load(), repo)
    done = run_tests(_store(args), work_tree=Path(args.work_tree), repo=repo, settings=test_settings,
                     ids_file=Path(args.ids_file) if args.ids_file else None, place=args.place,
                     need_os=args.os, role=args.role, ticket=args.ticket)
    line = (f"run={done.run} place={done.place} outcome={done.outcome} reason={done.reason or '-'}"
            f" collected={done.collected} failed={done.failed}\n")
    return line, TESTS_EXIT.get(done.outcome, TESTS_EXIT_OTHER)


def _email_of(login: str) -> str | None:
    """The public email of a GitHub account, or None (with a warning) when it cannot be read or
    the account shows none."""

    email = None
    if _LOGIN.fullmatch(login):
        try:
            email = github.run_gh(["api", f"users/{login}", "--jq", ".email"]).strip()
        except github.GithubError:
            pass
    if not email or email == "null" or "@" not in email:
        print(f"warning: email_of_unreadable: {login}", file=sys.stderr)
        return None
    return email


def _guards(args) -> tuple[str, int]:
    if args.command == "scan":
        return _guards_scan(args)
    if args.command == "reverse":
        event = hook.reverse(_store(args), args.fired_event, by=args.by)
        return f"reversed={event['id']} guard={event['detail']['guard']}\n", 0
    if args.command == "install":
        return _guards_install(args)
    if args.command == "status":
        return _guards_status(args)
    raise AssertionError(args.command)


def _guards_status(args) -> tuple[str, int]:
    """`guards status`: the six facts as a table (fact, ok, detail); exit 0 when all are ok, 1 when
    any is not (a protection that cannot be read is a failed fact, not an error)."""

    work_tree = Path(args.work_tree)
    report = guards_status.status(_store(args), args.repo or _repo_or_none(work_tree) or DEFAULT_REPO,
                                  work_tree)
    rows = [[fact.name, "yes" if fact.ok else "no", fact.detail] for fact in report.facts]
    return render(QueryResult(["fact", "ok", "detail"], rows)), 0 if report.ok else 1


def _guards_install(args) -> tuple[str, int]:
    """`guards install`: a short summary and exit 0; another `core.hooksPath` is a coded error
    (`hooks_path_taken`, exit 2) that leaves it as it was."""

    work_tree = Path(args.work_tree)
    done = install.install(work_tree, _store(args),
                           repo=args.repo or _repo_or_none(work_tree) or DEFAULT_REPO)
    return (f"pre-push hook: {'installed' if done.changed else 'already installed'}"
            f" (core.hooksPath={done.hooks_path})\n"
            f"claude code hook: {'present' if done.claude_hook else 'missing'}"
            " (.claude/settings.json PreToolUse)\n"), 0


def _hook(args) -> tuple[str, int]:
    if args.command == "brain":
        try:  # a prompt or a start is never stopped by this hook, whatever goes wrong
            from orch_it.brain.hook import run as run_brain_hook

            raw = sys.stdin.buffer.read().decode("utf-8", "replace")
            return run_brain_hook(raw, open_store=lambda: _store(args))
        except Exception as e:  # noqa: BLE001
            print(f"brain hook: {type(e).__name__}: {e}", file=sys.stderr)
            return "", 0
    if args.command == "subagent-stop":
        # the hold never blocks a stop by an error: whatever goes wrong lets the stop through
        try:
            from orch_it.roles.hold import run_subagent_stop

            raw = sys.stdin.buffer.read().decode("utf-8", "replace")
            return run_subagent_stop(raw, open_store=lambda: _store(args))  # a bad store never stops a block
        except Exception:
            return "", 0
    if args.command == "pre-tool-use":
        # the hook must never block a tool call: a store or settings error (which the CLI would
        # otherwise print and exit 2 on, Claude Code's blocking code) lets the call through
        try:
            raw = sys.stdin.read()
            return hook.run_pre_tool_use(raw, _store(args))
        except Exception:
            return "", 0
    if args.command == "pre-push":
        # a pre-push that always blocks would wedge every push: whatever goes wrong before the scan
        # (a store or settings error, a bad stdin) lets the push through; a scan that finds private
        # data exits 1 (the guard working) and says so on stderr
        try:
            raw = sys.stdin.buffer.read().decode("utf-8", "replace")
            return prepush.run_pre_push(raw, _store(args), remote_url=args.remote_url,
                                        remote_name=args.remote_name)
        except Exception:
            return "", 0
    raise AssertionError(args.command)


def _require_noreply(args, work_tree: Path) -> bool:
    """The check of `guards scan --range`: the option when given, else on when the repo's setting
    `commit_identity` is `noreply`. A work tree with no GitHub origin has no repo section to read,
    so the default mode applies."""

    if args.require_noreply_email is not None:
        return args.require_noreply_email
    repo = _repo_or_none(work_tree)
    mode = identity.DEFAULT_MODE if repo is None else identity.mode_of(repo)
    return mode == identity.DEFAULT_MODE


def _guards_scan(args) -> tuple[str, int]:
    """`guards scan`: one line per hit, then `hits=<n> commits=<n>`; exit 0 with no hit, 1 with hits."""

    if args.rev and not args.tree:
        raise CommandError("bad_options", "--rev goes with --tree")
    if args.words_file and args.machine_independent:
        raise CommandError("bad_options", "--words-file goes with the machine's own patterns")
    allowed = [*args.allow_email, *filter(None, map(_email_of, args.allow_email_of))]
    if args.machine_independent:
        patterns = scanner.Patterns.machine_independent(allowed_emails=allowed)
    else:
        patterns = scanner.Patterns.of_machine(
            allowed_emails=allowed, words_file=Path(args.words_file) if args.words_file else None)
    work_tree = Path(args.work_tree)
    unnamed: list[tuple[str, str]] = []  # (sha, author | committer) of a non-noreply e-mail
    if args.tree:
        hits, commits = scanner.scan_tree(work_tree, patterns, args.rev or "HEAD"), 1
    else:
        shas = scanner.commits_of_range(work_tree, "HEAD" if args.all else args.rev_range)
        hits, commits = scanner.scan_commits(work_tree, shas, patterns), len(shas)
        # the whole history (`--all`) is never rewritten, so its e-mails are not checked
        if args.rev_range is not None and _require_noreply(args, work_tree):
            unnamed = identity.noreply_failures(work_tree, shas)
    lines = [f"{hit.commit[:12]} {hit.where} {_one_line(hit.file)}:"
             f"{'-' if hit.line is None else hit.line} {hit.kind}\n" for hit in hits]
    lines += [f"{sha[:12]} {role} -:- not_noreply\n" for sha, role in unnamed]
    text = "".join(lines) + f"hits={len(hits) + len(unnamed)} commits={commits}\n"
    # a console that cannot show a file name's letters shows them escaped, not an error
    text = text.encode(sys.stdout.encoding or "utf-8", "backslashreplace").decode(
        sys.stdout.encoding or "utf-8")
    return text, 1 if hits or unnamed else 0


def _one_line(path: str | None) -> str:
    return "-" if path is None else path.replace("\n", "\\n").replace("\r", "\\r")


def _labels_sync(args) -> str:
    done = labels.sync(args.repo or _repo_of("."))
    return render(QueryResult(["label", "action"], done))


def _roles_render(args) -> str:
    done = roles.render(Path(args.work_tree), roles.RoleSet.load(args.settings))
    return render(QueryResult(["file", "action"], done))


def _spawned_roles() -> list[str] | None:
    """The roles `spawn` takes (those with inputs); a roles file that cannot be read leaves the
    choice open, and the command then says what is wrong."""

    try:
        return sorted(name for name, role in roles.RoleSet.load().roles.items() if role.inputs)
    except roles.RolesError:
        return None


def _spawn(args) -> str:
    from orch_it.roles import spawn

    if args.worktree is not None and not args.worktree.strip():  # Path("") would be the folder it runs in
        raise SpawnRefused("worktree_not_git", "worktree", "an empty path")
    if args.scratchpad is not None and not args.scratchpad.strip():
        raise SpawnRefused("scratchpad_unusable", "scratchpad", "an empty path")
    request = spawn.SpawnRequest(issue=args.issue, repo=args.repo, unit=args.unit, diff=args.diff,
                                 worktree=Path(args.worktree) if args.worktree is not None else None,
                                 scratchpad=Path(args.scratchpad) if args.scratchpad is not None else None)
    return spawn.spawn(args.role, request, roles=roles.RoleSet.load(), store=_store(args)).prompt


def _research(args) -> tuple[str, int]:
    try:
        brief = Path(args.brief).read_text("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise ResearchRefused("brief_unreadable", f"{args.brief}: {e}") from None
    repo = args.repo or research_run.repo_of(Path(args.work_tree))
    done = research_run.research(
        brief, settings=ResearchSettings.from_settings(settings.load()), store=_store(args), repo=repo,
        work_tree=Path(args.work_tree), brief_name=Path(args.brief).stem, model=args.model,
        no_reddit=args.no_reddit, report_to=args.report, ticket=args.ticket, job=args.job)
    if args.done_file:
        research_run.write_done(Path(args.done_file), done)
    return done.line(), RESEARCH_EXIT.get(done.outcome, TESTS_EXIT_OTHER)


def _check_names(args) -> tuple[str, int]:
    """The names check of the bridge's text, against the names the installation's settings and the work
    tree's own `settings.json` name: one line per hit, exit 1 on a hit, 0 clean."""

    work_tree = Path(args.work_tree)
    hits = research_names.find_names(research_names.bridge_texts(work_tree),
                                     research_names.refused_names_from_installation(work_tree=work_tree))
    if not hits:
        return "no refused name in the bridge's text\n", 0
    return "".join(f"{path}:{line}: {name}\n" for path, line, name in hits), 1


def _run(args) -> str:
    command = args.command
    if command == "query":
        return render(_store(args).query(args.sql, dict(args.param)))
    if command == "ticket-cost":
        return render(queries.ticket_cost(_store(args), args.repo, args.ticket))
    if command == "session-cost":
        return render(queries.session_cost(_store(args), session=args.session, repo=args.repo,
                                           since=args.since, until=args.until))
    if command == "limit-rate":
        return render(queries.limit_rate(_store(args)))
    if command == "guard-fires":
        return render(hook.guard_fires(_store(args)))
    if command == "measure-limit":
        if not 0 <= args.pct <= 100:
            raise CommandError("bad_pct", str(args.pct))
        with _store(args).writer() as writer:
            writer.write({
                "kind": "value.measured", "repo": args.repo, "ticket": None, "role": args.role,
                "session": None, "parent_session": None, "turn": None, "step": None, "run": None,
                "note": args.note,
                "detail": {"name": "weekly_limit_pct", "value": args.pct, "unit": "%",
                           "source": "usage_card"}})
        return ""
    if command == "set-limit-rate":
        settings.save(queries.WEEKLY_LIMIT_SETTING,
                      queries.checked_rate(args.usd_per_pct, args.plan, args.measured))
        return ""
    if command == "import":
        done = transcripts.import_transcripts(
            Path(args.folder), _store(args), repo=args.repo, role=args.role,
            subagent_role=args.subagent_role, ticket=args.ticket, session=args.session,
            quiet_minutes=args.quiet_minutes)
        return (f"written={done.written} already_stored={done.already_stored}"
                f" sessions={done.sessions} bad_lines={done.bad_lines}"
                f" pending_lines={done.pending_lines} refused_events={done.refused_events}"
                f" unreadable_files={done.unreadable_files}\n")
    if command == "import-github":
        done = github.import_github(_store(args), args.repo, since=args.since)
        failed = f" failure_codes={','.join(done.failure_codes)}" if done.failure_codes else ""
        return (f"runs_read={done.runs_read} jobs={done.jobs} written={done.written}"
                f" already_stored={done.already_stored} reports_missing={done.reports_missing}"
                f" runs_failed={done.runs_failed}{failed}\n")
    if command == "test-places":
        return render(queries.test_places(_store(args), args.repo))
    raise AssertionError(command)


def _repo_or_none(work_tree: Path) -> str | None:
    try:
        return _repo_of(str(work_tree))
    except CommandError:
        return None


def _skills_shadowed(args) -> int:
    """The SessionStart hook (#397 decision 3): prints nothing and exits 0 whatever happens, so it
    never blocks a session; an error goes to stderr."""

    try:
        shadow.run_hook(sys.stdin.buffer.read(), open_store=lambda: _store(args), repo_of=_repo_or_none)
    except Exception as e:  # fail open: a broken hook must never stop a session
        print(f"error: {e}", file=sys.stderr)
    return 0


def _skills(args) -> str:
    if args.command == "check":
        return update.render(update.check(Path(args.work_tree), ref=args.ref))
    repo = args.repo or _repo_of(args.work_tree)
    if args.command == "report":
        done = update.report(Path(args.work_tree), forge=GithubForge(repo), ref=args.ref)
        if done.status == "no_change":
            return "no change\n"
        said = {"opened": "issue", "already_open": "issue already open, body updated",
                "declined": "issue closed by the brain for this commit, not opened again"}[done.status]
        return f"{said}: {done.url} {','.join(done.labels)}".rstrip() + "\n"
    done = update.update(Path(args.work_tree), forge=GithubForge(repo), ref=args.ref,
                         identity=identity.for_repo(repo))
    if done.status == "no_change":
        return "no change\n"
    if done.status == "waits_for_brain":
        return "no pull request: this upstream commit waits for the brain (its issue is labelled needs-triage)\n"
    rule = f"labels={','.join(done.labels)}" if done.labels else "no label"
    if done.status == "already_open":
        return f"pull request already open: {done.url} {rule}\n"
    if done.status == "declined":
        return f"pull request closed unmerged for this commit, not opened again: {done.url}\n"
    return f"pull request: {done.url} {rule}\n"


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_ERROR
    if args.group == "skills" and args.command == "shadowed":
        return _skills_shadowed(args)
    try:
        if args.group == "tests":
            text, code = _tests_run(args)
        elif args.group == "guards":
            text, code = _guards(args)
        elif args.group == "hook":
            text, code = _hook(args)
        elif args.group == "labels":
            text, code = _labels_sync(args), 0
        elif args.group == "roles":
            text, code = _roles_render(args), 0
        elif args.group == "spawn":
            text, code = _spawn(args), 0
        elif args.group == "skills":
            text, code = _skills(args), 0
        elif args.group == "research":
            text, code = _research(args)
        elif args.group == "research-mcp":
            return research_mcp.serve()
        elif args.group == "research-check-names":
            text, code = _check_names(args)
        elif args.group == "tickets":
            text, code = ticket_commands.run(args, lambda: _store(args), lambda: args.repo or _repo_of("."),
                                             render, follow_of=merge_commands.parks_follower), 0
        elif args.group == "brain":
            text, code = brain_commands.run(args, lambda: _store(args), lambda: args.repo or _repo_of(".")), 0
        elif args.group in merge_commands.GROUPS:
            text, code = merge_commands.run(args, lambda: _store(args), _repo_of)
        else:
            text, code = _run(args), 0
    except (schema.EventError, StoreError, settings.SettingsError, transcripts.ImportError_,
            queries.QueryError, github.GithubError, labels.LabelError, roles.RolesError, ParkError, CheckError,
            SkillsError, SpawnRefused, CommandError, RunError, scanner.ScanError, install.InstallError, hook.ReverseError,
            ResearchRefused, HandoffError, identity.IdentityError, *merge_commands.ERRORS) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # a run stopped by a person has already written its end
        print("error: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    sys.stdout.write(text)
    return code
