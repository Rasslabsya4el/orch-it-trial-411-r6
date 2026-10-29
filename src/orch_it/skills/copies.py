"""Skill copies (#397 decision 1): his skills, byte for byte from his upstream at one commit, in
`.claude/skills/<name>/`, and the manifest `.claude/skills/pocock.json` that says where each came
from.

A tree is the skills of one upstream commit: `{upstream path: {file path in the folder: bytes}}`,
read from the commit's blobs, never from a work tree (whose line endings git may have changed).
A skill is a folder under `skills/` holding a `SKILL.md`, outside `skills/deprecated/`; its name is
the folder's name. The manifest records the upstream repo and commit, and per skill its upstream
path and the git blob hash of every file: the copies (`skills`) and his skills left out
(`not_copied`, each with why), so a later commit's new skill is told from one that was always
there; and `gone`, each name his upstream dropped with what became of it, written by the update
(#397 decision 4) so a text still naming it is caught (decision 5). A folder of `.claude/skills/` the manifest does not name is ours: never written, never
removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

GIT_TIMEOUT_SECONDS = 120
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SKILLS_FOLDER = PurePosixPath(".claude/skills")
MANIFEST_NAME = "pocock.json"
PENDING_NAME = "pocock.pending.json"
SKILLS_ROOT = "skills/"
LEFT_OUT_GROUP = "skills/deprecated/"
SKILL_FILE = "SKILL.md"
ABOUT = ("His skills copied byte for byte from `upstream` at `commit` (#397 decision 1): per copy its upstream path"
         " and the git blob hash of every file; `not_copied` his skills left out and why; `gone` the names his"
         " upstream dropped and what became of them. Written with the"
         " copies by `orch_it.skills.copies.write`; a skill folder this file does not name is ours.")

SKILLS_ERROR_CODES = (
    "git_failed",          # git missing, timed out or refused
    "commit_unknown",      # the commit is not in the git folder
    "no_skills",           # a tree with no skill: never read as every skill removed
    "name_clash",          # two upstream skill folders of one name
    "unsupported_entry",   # a link or submodule inside a skill folder
    "manifest_unreadable", # the manifest is no JSON
    "bad_manifest",        # the manifest is JSON of the wrong shape
    "copy_over_ours",      # a copy would overwrite a skill folder of ours
    "fetch_failed",        # his upstream could not be fetched
    "manifest_mismatch",   # the manifest is not his tree at its own commit
    "copies_edited",       # a copy differs from the manifest: an update would overwrite it
    "work_tree_dirty",     # the work tree an update branches from has changes
    "forge_failed",        # gh refused or timed out
)

_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_BLOB = re.compile(r"[0-9a-f]{40}")

Tree = dict[str, dict[str, bytes]]


class SkillsError(RuntimeError):
    """`code` is one of `SKILLS_ERROR_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in SKILLS_ERROR_CODES:
            raise ValueError(f"not a skills error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def blob_hash(data: bytes) -> str:
    """The hash git gives these bytes as a blob, so the manifest compares with any upstream tree."""

    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def skill_name(path: str) -> str:
    return PurePosixPath(path).name


def _git(git_dir: Path, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=git_dir, input=stdin, capture_output=True,
                              timeout=GIT_TIMEOUT_SECONDS, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SkillsError("git_failed", f"git {args[0]}: {e}") from None


def read_tree(git_dir: str | Path, commit: str) -> Tree:
    """The skills of `commit` in the git folder `git_dir`, read from its blobs."""

    git_dir = Path(git_dir)
    if not _git(git_dir, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0:
        raise SkillsError("commit_unknown", commit)
    listed = _git(git_dir, "ls-tree", "-r", "-z", commit, "--", SKILLS_ROOT)
    if listed.returncode != 0:
        raise SkillsError("git_failed", listed.stderr.decode("utf-8", "replace").strip())
    entries = _entries(listed.stdout)
    folders = _skill_folders(entries)
    if not folders:
        raise SkillsError("no_skills", commit)
    wanted: list[tuple[str, str, str]] = []  # (folder, file path in folder, sha)
    for entry in entries:
        folder = next((f for f in folders if entry.path.startswith(f + "/")), None)
        if folder is None:
            continue
        if entry.kind != "blob" or entry.mode == "120000":
            raise SkillsError("unsupported_entry", f"{entry.path}: a link or submodule")
        wanted.append((folder, entry.path[len(folder) + 1:], entry.sha))
    for folder in folders:
        _check_portable(skill_name(folder), [rel for f, rel, _ in wanted if f == folder])
    blobs = _cat_blobs(git_dir, sorted({sha for _, _, sha in wanted}))
    tree: Tree = {folder: {} for folder in folders}
    for folder, rel, sha in wanted:
        tree[folder][rel] = blobs[sha]
    return tree


@dataclass(frozen=True)
class _Entry:
    mode: str
    kind: str
    sha: str
    path: str


def _entries(listing: bytes) -> list[_Entry]:
    entries = []
    for item in listing.split(b"\0"):
        if item:
            meta, _, path = item.partition(b"\t")
            try:
                text = path.decode("utf-8")
            except UnicodeDecodeError:
                raise SkillsError("unsupported_entry", f"{path!r}: not UTF-8") from None
            entries.append(_Entry(*meta.decode().split(), text))
    return entries


def _skill_folders(entries: list[_Entry]) -> list[str]:
    """The skill folders, one per name; a folder inside another skill's folder is part of it."""

    folders = sorted(str(PurePosixPath(e.path).parent) for e in entries
                     if PurePosixPath(e.path).name == SKILL_FILE and not e.path.startswith(LEFT_OUT_GROUP)
                     and e.path != SKILLS_ROOT + SKILL_FILE)
    folders = [f for f in folders if not any(f.startswith(other + "/") for other in folders)]
    names: dict[str, str] = {}
    for folder in folders:
        if skill_name(folder) in names:
            raise SkillsError("name_clash", f"{names[skill_name(folder)]} and {folder}")
        names[skill_name(folder)] = folder
    return folders


# names Windows cannot hold as written: a copy must be the same bytes on Linux and Windows
_WINDOWS_RESERVED = re.compile(r"(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?", re.IGNORECASE)
_WINDOWS_FORBIDDEN = re.compile(r'[<>:"|?*\\\x00-\x1f]')


def _check_portable(name: str, rels: list[str]) -> None:
    """Refuse, before anything is written, a skill whose name or file paths a manifest cannot hold
    or a Windows folder cannot hold as written."""

    if not _NAME.fullmatch(name) or _WINDOWS_RESERVED.fullmatch(name):
        raise SkillsError("unsupported_entry", f"{name}: not a skill name")
    seen: dict[str, str] = {}
    for rel in rels:
        parts = rel.split("/")
        if not _safe_rel(rel) or any(_WINDOWS_FORBIDDEN.search(p) or _WINDOWS_RESERVED.fullmatch(p)
                                     or p[-1] in ". " for p in parts):
            raise SkillsError("unsupported_entry", f"{name}/{rel}: not a portable file name")
        folded = rel.casefold()
        if folded in seen:
            raise SkillsError("unsupported_entry", f"{name}/{rel} and {seen[folded]}: differ only in case")
        seen[folded] = rel


def _cat_blobs(git_dir: Path, shas: list[str]) -> dict[str, bytes]:
    done = _git(git_dir, "cat-file", "--batch", stdin="".join(f"{sha}\n" for sha in shas).encode())
    if done.returncode != 0:
        raise SkillsError("git_failed", done.stderr.decode("utf-8", "replace").strip())
    out, at, blobs = done.stdout, 0, {}
    for sha in shas:
        end = out.index(b"\n", at)
        header = out[at:end].decode().split()
        if len(header) != 3 or header[0] != sha or header[1] != "blob":
            raise SkillsError("git_failed", f"cat-file: {out[at:end]!r}")
        size = int(header[2])
        blobs[sha] = out[end + 1:end + 1 + size]
        at = end + 1 + size + 1
    return blobs


def _files(skill: dict[str, bytes]) -> dict[str, str]:
    return {rel: blob_hash(data) for rel, data in sorted(skill.items())}


@dataclass
class Manifest:
    upstream: str
    commit: str
    skills: dict[str, dict]       # name -> {"path": upstream path, "files": {file: blob hash}}
    not_copied: dict[str, dict]   # name -> {"path", "why", "files"}
    gone: dict[str, str] = field(default_factory=dict)  # a name his upstream no longer has -> what became of it

    @classmethod
    def build(cls, tree: Tree, *, upstream: str, commit: str, not_copied: dict[str, str]) -> Manifest:
        """Every skill of `tree` is copied but the ones `not_copied` names (name -> why); a name
        there that the tree no longer has is dropped."""

        skills, left_out = {}, {}
        for path in sorted(tree, key=skill_name):
            name = skill_name(path)
            entry = {"path": path, "files": _files(tree[path])}
            if name in not_copied:
                left_out[name] = {"path": path, "why": not_copied[name], "files": entry["files"]}
            else:
                skills[name] = entry
        return cls(upstream, commit, skills, left_out)

    @staticmethod
    def path(repo: str | Path) -> Path:
        return Path(repo) / SKILLS_FOLDER / MANIFEST_NAME

    @classmethod
    def load(cls, repo: str | Path) -> Manifest | None:
        """The repo's manifest, or None when it has none."""

        path = cls.path(repo)
        try:
            data = json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, RecursionError) as e:
            raise SkillsError("manifest_unreadable", f"{MANIFEST_NAME}: {e}") from None
        problem = _shape_problem(data)
        if problem:
            raise SkillsError("bad_manifest", problem)
        return cls(data["upstream"], data["commit"], data["skills"], data["not_copied"], data.get("gone", {}))

    def text(self) -> str:
        data = {"_about": ABOUT, "upstream": self.upstream, "commit": self.commit,
                "skills": self.skills, "not_copied": self.not_copied}
        if self.gone:
            data["gone"] = self.gone
        return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def _shape_problem(data) -> str | None:
    if not isinstance(data, dict):
        return "not an object"
    for key in ("upstream", "commit"):
        if not isinstance(data.get(key), str) or not data[key]:
            return f"{key}: no text"
    for group, extra in (("skills", ()), ("not_copied", ("why",))):
        entries = data.get(group)
        if not isinstance(entries, dict):
            return f"{group}: not an object"
        for name, entry in entries.items():
            if not _NAME.fullmatch(name):
                return f"{group}.{name}: not a skill name"
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                return f"{group}.{name}: no path"
            if any(not isinstance(entry.get(key), str) for key in extra):
                return f"{group}.{name}: no why"
            files = entry.get("files")
            if not isinstance(files, dict):
                return f"{group}.{name}: no files"
            for rel, sha in files.items():
                if not _safe_rel(rel) or not isinstance(sha, str) or not _BLOB.fullmatch(sha):
                    return f"{group}.{name}.files.{rel}: bad file"
    gone = data.get("gone", {})
    if not isinstance(gone, dict) or any(not _NAME.fullmatch(n) or not isinstance(w, str) for n, w in gone.items()):
        return "gone: not an object of skill name -> text"
    if set(data["skills"]) & set(data["not_copied"]):
        return "a skill both copied and not copied"
    return None


def _safe_rel(rel) -> bool:
    if not isinstance(rel, str) or not rel or "\\" in rel or rel.startswith("/"):
        return False
    parts = rel.split("/")
    return all(part and part not in (".", "..") and ":" not in part for part in parts)


def _replace_bytes(path: Path, data: bytes) -> None:
    """Replace the file whole, never half written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _pending_path(repo: Path) -> Path:
    return repo / SKILLS_FOLDER / PENDING_NAME


def _pending(repo: Path) -> set[str]:
    try:
        names = json.loads(_pending_path(repo).read_text("utf-8"))
    except FileNotFoundError:
        return set()
    except (OSError, ValueError, RecursionError) as e:
        raise SkillsError("manifest_unreadable", f"{PENDING_NAME}: {e}") from None
    if not isinstance(names, list) or not all(isinstance(n, str) and _NAME.fullmatch(n) for n in names):
        raise SkillsError("bad_manifest", f"{PENDING_NAME}: not a list of skill names")
    return set(names)


def write(repo: str | Path, tree: Tree, manifest: Manifest) -> None:
    """Make `.claude/skills/` hold exactly the copies `manifest` names, from `tree`: files the copy
    no longer has removed first, then each file written whole, the folders of copies the old
    manifest named and this one drops removed; folders of ours untouched. The names being written
    are recorded in `pocock.pending.json` before the first file and the manifest is written last,
    so a write cut part-way leaves the old manifest, `verify` names the cut, and the same write
    run again finishes it. One writer at a time."""

    repo = Path(repo)
    root = repo / SKILLS_FOLDER
    old = Manifest.load(repo)
    ours_not = (set(old.skills) if old else set()) | _pending(repo)
    by_name = {skill_name(path): path for path in tree}
    for name in manifest.skills:
        if name not in ours_not and (root / name).exists():
            raise SkillsError("copy_over_ours", name)
    _replace_bytes(_pending_path(repo), json.dumps(sorted(ours_not | set(manifest.skills))).encode())
    for name in manifest.skills:
        folder, files = root / name, tree[by_name[name]]
        if folder.exists() and not folder.is_dir():
            folder.unlink()
        for path in sorted(folder.rglob("*"), reverse=True):  # deepest first, so a folder empties
            rel = path.relative_to(folder).as_posix()
            if path.is_file() and rel not in files:
                path.unlink()
            elif path.is_dir() and (rel in files or not any(path.iterdir())):
                shutil.rmtree(path)
        for rel, data in files.items():
            target = folder / rel
            if not target.is_file() or target.read_bytes() != data:
                _replace_bytes(target, data)
    for name in sorted(ours_not - set(manifest.skills)):  # copies of the old manifest or of a cut write
        shutil.rmtree(root / name, ignore_errors=True)
    _replace_bytes(Manifest.path(repo), manifest.text().encode("utf-8"))
    _pending_path(repo).unlink()


def verify(repo: str | Path) -> list[str]:
    """Each difference between the copies on disk and the manifest, one line each, sorted: a file
    missing, a hash that differs, a file the manifest does not name, a copy folder missing, a
    skill left out whose folder is present. Empty when every copy is as recorded."""

    repo = Path(repo)
    root = repo / SKILLS_FOLDER
    manifest = Manifest.load(repo)
    if manifest is None:
        return [f"{MANIFEST_NAME}: missing"]
    problems = [f"{PENDING_NAME}: a write was cut part-way"] if _pending_path(repo).exists() else []
    for name, entry in manifest.skills.items():
        folder = root / name
        if not folder.is_dir():
            problems.append(f"{name}: copy missing")
            continue
        for rel, sha in entry["files"].items():
            path = folder / rel
            if not path.is_file():
                problems.append(f"{name}/{rel}: missing")
            elif blob_hash(path.read_bytes()) != sha:
                problems.append(f"{name}/{rel}: hash differs")
        for path in folder.rglob("*"):
            rel = path.relative_to(folder).as_posix()
            if path.is_file() and rel not in entry["files"]:
                problems.append(f"{name}/{rel}: not in the manifest")
    for name in manifest.not_copied:
        if (root / name).exists():
            problems.append(f"{name}: not copied, yet present")
    return sorted(problems)
