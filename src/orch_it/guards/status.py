"""`guards status` (#395 decision 5, row 9): read `main`'s branch protection and the two hooks, report
six facts, and write them as three `check` steps (`branch_protection`, `hooks_path`,
`guard_install`). Push to main without green tests is held by GitHub's own branch protection
(part 0), so this only reads it, never changes it.

The six facts: the required checks are exactly `linux` and `windows`; the branch is strict;
admins are enforced; force push and deletion are off; the pre-push hook is installed
(`core.hooksPath` is our hooks folder and `hooks/pre-push` is there); the Claude Code hook is
installed. A protection that cannot be read fails its four facts and says why; nothing raises.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from orch_it.guards import install
from orch_it.testruns import github

REQUIRED_CHECKS = ("linux", "windows")
BRANCH = "main"
FACT_NAMES = ("required_checks", "strict", "enforce_admins", "force_push_and_deletion_off",
              "pre_push_hook", "claude_hook")


@dataclass(frozen=True)
class Fact:
    """One of the six facts: `ok` and a line of `detail` for the table."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Report:
    facts: tuple[Fact, ...]

    @property
    def ok(self) -> bool:
        return all(fact.ok for fact in self.facts)


def read_protection(repo: str) -> dict:
    """`main`'s branch protection from the GitHub API, as the API answers it."""

    return json.loads(github.run_gh(["api", f"repos/{repo}/branches/{BRANCH}/protection"]))


def _unprotected(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, github.GithubError) and ("not protected" in text or "404" in text)


def _enabled(protection: dict, key: str) -> bool | None:
    flag = protection.get(key)
    return flag.get("enabled") if isinstance(flag, dict) and isinstance(flag.get("enabled"), bool) else None


def _checks_of(protection: dict) -> tuple[list[str], bool | None]:
    """The names of the required checks (the older `contexts` and the newer `checks`) and `strict`."""

    section = protection.get("required_status_checks")
    if not isinstance(section, dict):
        return [], None
    names = {name for name in section.get("contexts") or [] if isinstance(name, str)}
    names |= {c["context"] for c in section.get("checks") or []
              if isinstance(c, dict) and isinstance(c.get("context"), str)}
    strict = section.get("strict")
    return sorted(names), strict if isinstance(strict, bool) else None


def _branch_facts(protection_reader, repo: str):
    """(the four protection facts, the `check` findings, the `check` facts)."""

    try:
        protection = protection_reader(repo)
        if not isinstance(protection, dict):
            raise ValueError("not a mapping")
    except Exception as error:
        if _unprotected(error):
            finding, detail = "branch_unprotected", f"unprotected: {BRANCH} has no protection"
        else:
            finding, detail = "protection_unreadable", "protection could not be read"
        facts = [Fact(name, False, detail) for name in FACT_NAMES[:4]]
        return facts, [finding], {"protection": finding.split("_", 1)[1]}
    names, strict = _checks_of(protection)
    admins = _enabled(protection, "enforce_admins")
    force, deletion = _enabled(protection, "allow_force_pushes"), _enabled(protection, "allow_deletions")
    found = {
        "required_checks": (names == sorted(REQUIRED_CHECKS), "required_checks_wrong"),
        "strict": (strict is True, "not_strict"),
        "enforce_admins": (admins is True, "admins_not_enforced"),
    }
    findings = [finding for ok, finding in found.values() if not ok]
    if force is not False:
        findings.append("force_push_allowed")
    if deletion is not False:
        findings.append("deletion_allowed")
    details = {"required_checks": ", ".join(names) or "none", "strict": str(strict).lower(),
               "enforce_admins": str(admins).lower(),
               "force_push_and_deletion_off": f"allow_force_pushes={str(force).lower()}"
                                              f" allow_deletions={str(deletion).lower()}"}
    facts = [Fact(name, ok, details[name]) for name, (ok, _) in found.items()]
    facts.append(Fact("force_push_and_deletion_off", force is False and deletion is False,
                      details["force_push_and_deletion_off"]))
    return facts, findings, {"required_checks": names, "strict": strict, "enforce_admins": admins,
                             "allow_force_pushes": force, "allow_deletions": deletion}


def status(store, repo: str, work_tree: Path | str = ".", *,
           protection: Callable[[str], dict] | None = None) -> Report:
    """Read `repo`'s protection of `main` (`protection(repo)`, default the GitHub API) and the hooks
    of `work_tree`, write the three checks and return the six facts. Never raises on what it
    reads: a protection or a folder that cannot be read is a failed fact."""

    branch, branch_findings, branch_facts = _branch_facts(protection or read_protection, repo)

    hooks_path = None
    pre_push_file = claude = False
    ours = False
    hook_findings: list[str] = []
    try:
        root = install.top_level(Path(work_tree))
        hooks_path = install.read_hooks_path(root)
        ours = install.is_ours(hooks_path, root)
        pre_push_file = install.pre_push_present(root)
        claude = install.claude_hook_present(root)
    except install.InstallError as error:
        hook_findings.append(error.code)
    if hooks_path is None and "not_a_repo" not in hook_findings:
        hook_findings.append("hooks_path_unset")
    elif hooks_path is not None and not ours:
        hook_findings.append("hooks_path_other")
    if not pre_push_file:
        hook_findings.append("pre_push_file_missing")
    pre_push_hook = ours and pre_push_file
    install_findings = ([] if pre_push_hook else ["pre_push_hook_missing"]) + (
        [] if claude else ["claude_hook_missing"])

    facts = [*branch, Fact("pre_push_hook", pre_push_hook,
                           f"core.hooksPath={hooks_path or 'unset'}"
                           f" {install.HOOKS_FOLDER}/{install.PRE_PUSH_FILE} "
                           f"{'present' if pre_push_file else 'missing'}"),
             Fact("claude_hook", claude, f"{'.'.join(install.CLAUDE_SETTINGS)} PreToolUse "
                                         f"{'present' if claude else 'missing'}")]
    for check, findings, detail in (
            ("branch_protection", branch_findings, branch_facts),
            ("hooks_path", hook_findings,
             {"hooks_path": hooks_path, "ours": ours, "pre_push_file": pre_push_file}),
            ("guard_install", install_findings, {"pre_push_hook": pre_push_hook, "claude_hook": claude})):
        install.write_check(store, repo, check, repo, "failed" if findings else "passed", findings, detail)
    return Report(tuple(facts))
