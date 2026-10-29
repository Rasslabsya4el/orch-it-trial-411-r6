"""One command line split into the simple commands it runs, each with its words expanded and the
directory it runs in tracked (#395 decision 2: the resolving the guard needs before it reads the
object). The dialect is POSIX (a `tool` of "Bash", Git Bash included) or Windows (PowerShell and
cmd); wrappers switch between them. Nothing here decides anything about safety.

A word that cannot be read in full — it holds a command substitution `$(...)` or a backtick, an
unset or computed variable, or a glob the guard will not expand — is `None`, so a caller never
mistakes "could not read this" for "a safe literal".

A command nested past `MAX_DEPTH` is not dropped: it stands as one `Simple` whose head and one
argument are both `None` (`UNREADABLE`), so the trigger reads it as an opaque command, never as
nothing (#395 unit 2b).
"""

from __future__ import annotations

import ntpath
import posixpath
import re
from dataclasses import dataclass, field


@dataclass
class Simple:
    """One simple command: `argv` its words (a word that could not be read in full is None),
    `cwd` the directory it runs in (None when a `cd` went somewhere unreadable), `dialect`
    "posix" or "windows". `subs` holds the simple commands of any command substitution in its
    words, which also run."""

    argv: list[str | None]
    cwd: str | None
    dialect: str
    subs: list["Simple"] = field(default_factory=list)
    stdin: str | None = None        # the heredoc body fed to it, if any
    stdin_to_shell: bool = False    # its heredoc line pipes into a shell, which runs the body
    text: str = ""                  # the segment as written (heredoc markers removed)
    redirects: list = field(default_factory=list)  # `>` targets that overwrite (None: unreadable)
    too_deep: bool = False          # a command nested past MAX_DEPTH: unread, refused as opaque


# a command nested deeper than this (substitutions or `-c` wrappers inside one another) is read
# no further: the inner content is treated as unreadable, never followed into a stack overflow
MAX_DEPTH = 40

# a Windows-absolute path a `cd` target may take in Git Bash: a drive (`C:/x`, `C:\x`) or a UNC
# share (`\\srv\share`); such a target replaces the cwd rather than joining onto it
_WIN_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def _too_deep(cwd: str | None, dialect: str) -> Simple:
    """What stands for a command nested past `MAX_DEPTH`: an unreadable head with an unreadable
    argument."""

    return Simple([None, None], cwd, dialect, too_deep=True)


def simple_commands(command, *, tool: str, cwd: str | None, env, windows: bool,
                    depth: int = 0) -> list[Simple]:
    """Every simple command `command` runs, in order, each with its words expanded and its cwd
    tracked. `tool` "PowerShell" or a `windows` Git Bash both reach the Windows wrappers. Never
    raises: a non-string command, or nesting past `MAX_DEPTH`, yields what could be read."""

    if not isinstance(command, str):
        return []
    dialect = "windows" if tool == "PowerShell" else "posix"
    if depth > MAX_DEPTH:
        return [_too_deep(cwd, dialect)] if command.strip() else []
    walker = _Walker(dict(env or {}), windows, depth)
    walker.run(command, dialect, cwd)
    return walker.out


class _Walker:
    def __init__(self, env: dict, windows: bool, depth: int = 0):
        self.env = env
        self.windows = windows
        self.depth = depth
        self.out: list[Simple] = []

    def run(self, command: str, dialect: str, cwd: str | None) -> None:
        if dialect == "windows":
            _WindowsSplit(self).walk(command, cwd)
        else:
            _PosixSplit(self).walk(command, cwd)


# --- POSIX -----------------------------------------------------------------------------------

_POSIX_OPERATORS = (";", "&&", "||", "|&", "|", "&", "\n")


_FUNCTION_DEF = re.compile(
    r"^(?:function\s+[^\s(){}]+\s*(?:\(\s*\))?|[^\s(){}=]+\s*\(\s*\))\s*"
    r"(?P<body>\{.*\}|\(.*\))$", re.S)


class _PosixSplit:
    """Splits a POSIX command list into simple commands, tracking `cd` and assignments at the
    list level, and recursing into `( )`, `{ }` and command substitutions."""

    def __init__(self, walker: _Walker):
        self.walker = walker
        self.env = dict(walker.env)
        self.cwd: str | None = None
        self.heredocs: list[Heredoc] | None = None  # shared by a subshell and a substitution

    def walk(self, command: str, cwd: str | None) -> None:
        self.cwd = cwd
        if self.heredocs is None:
            command, self.heredocs = _strip_heredocs(command)
        for segment in _posix_segments(command):
            self._simple(segment)

    def _simple(self, segment: str) -> None:
        extra_subs: list[Simple] = []
        if segment[:1] == "(" and segment[-1:] == ")":
            if self.walker.depth >= MAX_DEPTH:
                self.walker.out.append(_too_deep(self.cwd, "posix"))  # nested too deep to follow
                return
            child = _PosixSplit(_Walker(self.env, self.walker.windows, self.walker.depth + 1))
            child.walker.out = self.walker.out  # a subshell's cd/assignments do not leak out
            child.env = dict(self.env)
            child.heredocs = self.heredocs
            child.walk(segment[1:-1], self.cwd)
            return
        if segment[:1] == "{" and segment[-1:] == "}":
            for inner in _posix_segments(segment[1:-1]):  # a brace group runs in this shell
                self._simple(inner)
            return
        defined = _FUNCTION_DEF.match(segment)
        if defined and self.walker.depth < MAX_DEPTH:
            # `function f { ...; }` / `f() { ...; }`: the body is read as if it ran here, so a
            # delete in a function defined and then called cannot hide
            self._simple(defined.group("body"))
            return
        raw_redirects: list[str] = []
        tokens = _posix_tokens(segment, raw_redirects)
        written: list[str | None] = []
        for raw in raw_redirects:
            text, redirect_subs = self._expand_word(raw)
            written.append(text)
            extra_subs += redirect_subs
        if not tokens:
            if written or extra_subs:  # `> file` alone empties the file
                self.walker.out.append(Simple([], self.cwd, "posix", extra_subs, None, False,
                                              _HD_FIND.sub("", segment), written))
            return
        held = [self.heredocs[int(k)] for k in _HD_FIND.findall(segment)
                if int(k) < len(self.heredocs)]
        assignments, rest = _leading_assignments(tokens)
        for name, raw in assignments:
            value, assigned_subs = self._expand_word(raw)
            self.env[name] = value or ""
            self.walker.out.extend(assigned_subs)  # `x=$(rm -rf y)` runs the substitution
        if not rest:
            return
        words, subs = self._expand_all(rest)
        subs += extra_subs
        if words and words[0] is not None:
            head = words[0]
            name = posixpath.basename(head)
            if name in ("cd", "pushd"):
                self._change_dir(words)
                return
            if name == "popd":
                return
        for heredoc in held:
            if not heredoc.quoted:  # an unquoted heredoc expands `$( )` and backticks: they run
                for kind, text in _dquote_pieces(heredoc.body):
                    if kind == "sub":
                        subs += self._parse_sub(text, fresh=True)
        self.walker.out.append(Simple(
            words, self.cwd, "posix", subs,
            "\n".join(h.body for h in held) if held else None,
            any(h.to_shell for h in held), _HD_FIND.sub("", segment), written))

    def _change_dir(self, words: list[str | None]) -> None:
        args = words[1:]
        has_arg = any(w is None or not w.startswith("-") or w == "-" for w in args)
        target = next((w for w in args if w is None or not w.startswith("-") or w == "-"), None)
        if not has_arg:  # `cd` with no argument goes HOME
            self.cwd = self.env.get("HOME") or self.env.get("USERPROFILE")
        elif target is None or target in ("", "-") or self.cwd is None:
            # an unreadable target, or `cd -` (the previous dir, unknown): the cwd is now unknown
            self.cwd = None
        elif self.walker.windows and _WIN_ABSOLUTE.match(target):
            # Git Bash on Windows: a drive-letter (`C:/x`, `C:\x`) or UNC target is absolute and
            # replaces the cwd — joining it onto the old one (posixpath glues it, since it is not
            # POSIX-absolute) would lose the real directory and refuse a later relative delete
            self.cwd = ntpath.normpath(target)
        else:
            self.cwd = posixpath.normpath(posixpath.join(self.cwd, target))

    def _expand_all(self, tokens: list[str]) -> tuple[list[str | None], list[Simple]]:
        words: list[str | None] = []
        subs: list[Simple] = []
        for token in tokens:
            text, token_subs = self._expand_word(token)
            words.append(text)
            subs += token_subs
        return words, subs

    def _expand_word(self, token: str) -> tuple[str | None, list[Simple]]:
        """One token expanded to a string, or None when part of it cannot be read. Command
        substitutions inside it are parsed for their own commands and make the word None."""

        subs: list[Simple] = []
        out: list[str] = []
        readable = True
        for piece in _posix_pieces(token):
            kind, body = piece
            if kind == "lit":
                out.append(body)
            elif kind == "squote":
                out.append(body)
            elif kind == "dquote":
                text, inner = self._expand_dquote(body)
                subs += inner
                if text is None:
                    readable = False
                else:
                    out.append(text)
            elif kind == "var":
                value = self._var(body)
                if value is None:
                    readable = False
                else:
                    out.append(value)
            elif kind == "tilde":
                home = self.env.get("HOME") or self.env.get("USERPROFILE")
                if home is None:
                    readable = False
                else:
                    out.append(home + body)
            elif kind == "sub":
                subs += self._parse_sub(body)
                readable = False
        return ("".join(out) if readable else None), subs

    def _expand_dquote(self, body: str) -> tuple[str | None, list[Simple]]:
        subs: list[Simple] = []
        out: list[str] = []
        readable = True
        for kind, text in _dquote_pieces(body):
            if kind == "lit":
                out.append(text)
            elif kind == "var":
                value = self._var(text)
                if value is None:
                    readable = False
                else:
                    out.append(value)
            elif kind == "sub":
                subs += self._parse_sub(text)
                readable = False
        return ("".join(out) if readable else None), subs

    def _var(self, spec: str) -> str | None:
        name = spec
        default = None
        found = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?::?-(.*))?$", spec, re.S)
        if found:
            name, default = found.group(1), found.group(2)
        elif not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec):
            return None  # ${#x}, ${x[...]}, positional: not read
        if name in self.env:
            return self.env[name]
        return default

    def _parse_sub(self, body: str, fresh: bool = False) -> list[Simple]:
        if self.walker.depth >= MAX_DEPTH:
            return [_too_deep(self.cwd, "posix")]  # nested too deep: left unread, as opaque
        sink = _Walker(self.env, self.walker.windows, self.walker.depth + 1)
        inner = _PosixSplit(sink)
        inner.env = dict(self.env)
        # a `$( )` body that came from already-stripped text carries markers into the shared
        # list; but a heredoc the outer strip missed inside a quote (`"$(cat <<'EOF' ... EOF)"`,
        # the `git commit -m "$(...)"` pattern) is still raw here, so its body must be stripped
        # fresh rather than read as commands
        if not fresh and not _has_raw_heredoc(body):
            inner.heredocs = self.heredocs
        inner.walk(body, self.cwd)
        return sink.out


# --- Windows (PowerShell and cmd) ------------------------------------------------------------

class _WindowsSplit:
    """Splits a PowerShell or cmd command line. The two share enough for the guard: `;`, `&&`,
    `||`, `|`, `&`, newlines and `(` `)` separate; quotes are `'` and `"`; `$name`, `${name}`
    and `%NAME%` expand; `cd`/`Set-Location`/`chdir` track the directory."""

    def __init__(self, walker: _Walker):
        self.walker = walker
        self.env = dict(walker.env)
        self.cwd: str | None = None

    def walk(self, command: str, cwd: str | None) -> None:
        self.cwd = cwd
        for segment in _windows_segments(command):
            self._simple(segment)

    def _simple(self, segment: str) -> None:
        if segment[:1] == "(" and segment[-1:] == ")":
            if self.walker.depth >= MAX_DEPTH:
                self.walker.out.append(_too_deep(self.cwd, "windows"))
                return
            self.walker.depth += 1
            for inner in _windows_segments(segment[1:-1]):
                self._simple(inner)
            self.walker.depth -= 1
            return
        tokens = _windows_tokens(segment)
        if not tokens:
            return
        words = [self._expand(token) for token in tokens]
        if words and words[0] is not None:
            name = _win_command_name(words[0])
            if name in ("cd", "chdir", "set-location", "sl", "pushd", "push-location"):
                self._change_dir(words)
                return
            if name in ("popd", "pop-location"):
                return
        self.walker.out.append(Simple(words, self.cwd, "windows", text=segment))

    def _change_dir(self, words: list[str | None]) -> None:
        target = next((w for w in words[1:] if not (w or "").startswith(("-", "/"))), None)
        if target is None or target == "" or self.cwd is None:
            if target is None:
                return  # `cd` with no argument does not move on Windows
            self.cwd = None
        else:
            self.cwd = ntpath.normpath(ntpath.join(self.cwd, target))

    def _expand(self, token: str) -> str | None:
        out: list[str] = []
        readable = True
        for index, (kind, body) in enumerate(_windows_pieces(token)):
            if kind in ("lit", "squote"):
                if index == 0 and body[:1] == "~" and (len(body) == 1 or body[1] in "/\\"):
                    home = self.env.get("USERPROFILE") or self.env.get("HOME")
                    if home is None:
                        readable = False
                    else:
                        out.append(home + body[1:])
                else:
                    out.append(body)
            elif kind == "var":
                name = body[4:] if body.lower().startswith("env:") else body  # $env:PATH
                value = self.env.get(name)
                if value is None:
                    readable = False
                else:
                    out.append(value)
            elif kind == "sub":
                readable = False
        return "".join(out) if readable else None


def _win_command_name(head: str) -> str:
    name = ntpath.basename(head).lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1", ".com"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


# --- tokenisers ------------------------------------------------------------------------------

@dataclass
class Heredoc:
    """One heredoc: `body` the lines it feeds a command, `quoted` whether its delimiter was quoted
    (a quoted one does not expand `$( )`), `to_shell` whether its line pipes into a shell."""

    body: str
    quoted: bool
    to_shell: bool


# a heredoc's place in the stripped text: `<<` then this marker, which the tokeniser drops as part
# of the redirection and `_simple` reads back to find the body
_HD_FIND = re.compile("\x00H(\\d+)\x00")
_PIPE_TO_SHELL = re.compile(
    r"\|\s*(?:(?:sudo|env)\s+(?:-\S+\s+)*)?(?:\S*/)?(?:bash|sh|zsh|dash|ksh)\b(?!\s+-\w*c)")


_RAW_HEREDOC = re.compile(r"<<(?!<)(?!\x00H)")


def _has_raw_heredoc(body: str) -> bool:
    """Whether `body` holds a heredoc opener that was not already turned into a marker (one the
    outer strip missed because it sat inside a quoted `$( )`)."""

    return _RAW_HEREDOC.search(body) is not None


def _strip_heredocs(command: str) -> tuple[str, list[Heredoc]]:
    """Take every heredoc body (`<<WORD ... WORD`) out of the command text and return the text
    with each `<<WORD` turned into `<<` plus a marker, with the bodies in order. A body is data
    fed to a command on stdin, not commands, so a delete word in it is no command (O35, P222);
    whether a consumer RUNS it is decided by the caller from the command it is fed to (#395 unit
    2b)."""

    out: list[str] = []
    heredocs: list[Heredoc] = []
    i = 0
    n = len(command)
    pending: list[tuple[int, str]] = []  # (index into heredocs, delimiter) opened on this line
    line_start = 0
    while i < n:
        ch = command[i]
        if ch in "'\"":
            end = _skip_quote(command, i, ch)
            out.append(command[i:end])
            i = end
            continue
        if ch == "<" and command[i:i + 2] == "<<" and command[i:i + 3] != "<<<":
            prefix = command[line_start:i]
            j = i + 2
            if j < n and command[j] == "-":
                j += 1
            while j < n and command[j] in " \t":
                j += 1
            delimiter, quoted, j = _heredoc_delimiter(command, j)
            if (delimiter is None or prefix.count("((") > prefix.count("))")
                    or (delimiter.isdigit() and not quoted)):
                out.append("<<")  # a shift (`$((1<<2))`, `1<<2`), not a heredoc
                i += 2
                continue
            eol = command.find("\n", j)
            heredocs.append(Heredoc("", quoted,
                                    bool(_PIPE_TO_SHELL.search(command[j:n if eol == -1 else eol]))))
            pending.append((len(heredocs) - 1, delimiter))
            out.append("<<\x00H%d\x00" % (len(heredocs) - 1))
            i = j
            continue
        if ch == "\n":
            out.append("\n")
            i += 1
            if pending:
                i = _read_heredoc_bodies(command, i, pending, heredocs)
                pending = []
            line_start = i
            continue
        out.append(ch)
        i += 1
    return "".join(out), heredocs


def _heredoc_delimiter(text: str, i: int) -> tuple[str | None, bool, int]:
    """The delimiter word at `i` with its quotes removed, whether any part of it was quoted, and
    where it ends."""

    n = len(text)
    j = i
    chars: list[str] = []
    quoted = False
    while j < n and text[j] not in " \t\n;|&<>()":
        c = text[j]
        if c in "'\"":
            end = _skip_quote(text, j, c)
            chars.append(text[j + 1:end - 1])
            quoted = True
            j = end
        elif c == "\\" and j + 1 < n:
            chars.append(text[j + 1])
            quoted = True
            j += 2
        else:
            chars.append(c)
            j += 1
    return ("".join(chars) or None), quoted, j


def _read_heredoc_bodies(text: str, i: int, pending, heredocs: list[Heredoc]) -> int:
    n = len(text)
    for index, delimiter in pending:
        lines: list[str] = []
        while i < n:
            end = text.find("\n", i)
            end = n if end == -1 else end
            line = text[i:end]
            i = end + 1 if end < n else n
            if line.strip() == delimiter:
                break
            lines.append(line)
        heredocs[index].body = "\n".join(lines)
    return i


def _posix_segments(command: str) -> list[str]:
    """The command list split into simple-command segments at top-level operators, with `( )`
    and `{ }` groups kept whole (their own cd does not leak out, so a group is one segment that
    the caller recurses into)."""

    return _split_top(command, _POSIX_OPERATORS, quotes="'\"", expansions=True, windows=False)


def _windows_segments(command: str) -> list[str]:
    return _split_top(command, (";", "&&", "||", "|", "&", "\n"), quotes="'\"",
                      expansions=False, windows=True)


def _split_top(command, operators, *, quotes, expansions, windows) -> list[str]:
    segments: list[str] = []
    buf: list[str] = []
    i = 0
    depth = 0
    n = len(command)
    operators = sorted(operators, key=len, reverse=True)
    while i < n:
        ch = command[i]
        if ch in quotes:
            end = _skip_quote(command, i, ch)
            buf.append(command[i:end])
            i = end
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(command[i:i + 2])
            i += 2
            continue
        if expansions and ch == "$" and command[i:i + 2] == "$(":
            end = _skip_paren(command, i + 1)
            buf.append(command[i:end])
            i = end
            continue
        if expansions and ch == "`":
            end = command.find("`", i + 1)
            end = n if end == -1 else end + 1
            buf.append(command[i:end])
            i = end
            continue
        if ch in "([{":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        if depth == 0:
            matched = next((op for op in operators if command.startswith(op, i)), None)
            if matched == "|" and expansions and buf and buf[-1][-1:] == ">":
                matched = None  # `>|` is one redirection operator, not a pipe
            if matched:
                segments.append("".join(buf))
                buf = []
                i += len(matched)
                continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return [s for s in (seg.strip() for seg in segments) if s]


def _skip_quote(text: str, i: int, quote: str) -> int:
    j = i + 1
    n = len(text)
    while j < n:
        if quote == '"' and text[j] == "\\" and j + 1 < n:
            j += 2
            continue
        if text[j] == quote:
            return j + 1
        j += 1
    return n


def _skip_paren(text: str, i: int) -> int:
    depth = 0
    j = i
    n = len(text)
    while j < n:
        if text[j] in "'\"":
            j = _skip_quote(text, j, text[j])
            continue
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return n


# `$IFS` / `${IFS}` is the field separator; unset it defaults to whitespace, so it splits words
_IFS_SPLIT = re.compile(r"\$\{IFS\}|\$IFS(?![A-Za-z0-9_])")


def _posix_tokens(segment: str, redirects: list | None = None) -> list[str]:
    """A simple command split into word tokens on unquoted whitespace, each token keeping its
    quotes and expansions for later reading. A trailing heredoc redirection is dropped (its body
    is not a word); `>file` and `2>&1` redirections are dropped too, but when `redirects` is
    given the raw target of each single `>` (one that overwrites) is added to it."""

    tokens: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(segment)
    while i < n:
        ch = segment[i]
        if ch in " \t":
            if buf:
                tokens.append("".join(buf))
                buf = []
            i += 1
            continue
        if ch in "'\"":
            end = _skip_quote(segment, i, ch)
            buf.append(segment[i:end])
            i = end
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(segment[i + 1])
            i += 2
            continue
        if ch == "$":
            ifs = _IFS_SPLIT.match(segment, i)
            if ifs:  # `$IFS` / `${IFS}` is word-splitting whitespace (`rm${IFS}-rf${IFS}x`)
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                i = ifs.end()
                continue
        if ch == "$" and segment[i:i + 2] == "$(":
            end = _skip_paren(segment, i + 1)
            buf.append(segment[i:end])
            i = end
            continue
        if ch == "`":
            end = segment.find("`", i + 1)
            end = n if end == -1 else end + 1
            buf.append(segment[i:end])
            i = end
            continue
        if ch in "<>":
            buf_token = "".join(buf)
            if buf_token and not buf_token.isdigit():  # a bare fd number (`2` of `2>&1`) is part
                tokens.append(buf_token)               # of the redirect, not a word
            buf = []
            if ch == ">" and redirects is not None:
                target = _overwrite_target(segment, i)
                if target:
                    redirects.append(target)
            i = _skip_redirect(segment, i)
            continue
        buf.append(ch)
        i += 1
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t]


def _overwrite_target(segment: str, i: int) -> str | None:
    """The raw file word of the `>` at `segment[i]` when it overwrites a file: not `>>` (appends),
    not `>&2` (a descriptor copy), not `>(cmd)` (a process substitution)."""

    n = len(segment)
    j = i + 1
    if segment[i:i + 2] == ">>" or (j < n and segment[j] in "&("):
        return None
    if j < n and segment[j] == "|":
        j += 1
    while j < n and segment[j] in " \t":
        j += 1
    start = j
    while j < n and segment[j] not in " \t|;&<>":
        if segment[j] in "'\"":
            j = _skip_quote(segment, j, segment[j])
        elif segment[j:j + 2] == "$(":
            j = _skip_paren(segment, j + 1)
        else:
            j += 1
    return segment[start:j] or None


def _skip_redirect(segment: str, i: int) -> int:
    n = len(segment)
    j = i
    while j < n and segment[j] in "<>&012":
        j += 1
    while j < n and segment[j] in " \t":
        j += 1
    while j < n and segment[j] not in " \t|;&":
        if segment[j] in "'\"":
            j = _skip_quote(segment, j, segment[j])
        else:
            j += 1
    return j


def _posix_pieces(token: str):
    """A token split into literal, single-quoted, double-quoted, variable, tilde and
    substitution pieces."""

    pieces = []
    i = 0
    n = len(token)
    if token[:1] == "~" and (n == 1 or token[1] in "/"):
        end = 1
        while end < n and token[end] not in "":
            break
        slash = token.find("/", 1)
        rest = token[1:] if slash == -1 else token[1:]
        pieces.append(("tilde", token[1:]))
        return pieces
    lit = []
    while i < n:
        ch = token[i]
        if ch == "'":
            end = _skip_quote(token, i, "'")
            lit.append(("squote", token[i + 1:end - 1]))
            i = end
        elif ch == '"':
            end = _skip_quote(token, i, '"')
            lit.append(("dquote", token[i + 1:end - 1]))
            i = end
        elif ch == "$" and token[i:i + 2] == "$(":
            end = _skip_paren(token, i + 1)
            lit.append(("sub", token[i + 2:end - 1]))
            i = end
        elif ch == "`":
            end = token.find("`", i + 1)
            end = n if end == -1 else end + 1
            lit.append(("sub", token[i + 1:end - 1]))
            i = end
        elif ch == "$":
            spec, i = _read_var(token, i)
            lit.append(("var", spec))
        elif ch == "\\" and i + 1 < n:
            lit.append(("lit", token[i + 1]))
            i += 2
        else:
            lit.append(("lit", ch))
            i += 1
    return lit


def _read_var(token: str, i: int) -> tuple[str, int]:
    n = len(token)
    if token[i:i + 2] == "${":
        end = token.find("}", i + 2)
        end = n if end == -1 else end
        return token[i + 2:end], (end + 1 if end < n else n)
    j = i + 1
    while j < n and (token[j].isalnum() or token[j] == "_"):
        j += 1
    return token[i + 1:j], j


def _dquote_pieces(body: str):
    pieces = []
    i = 0
    n = len(body)
    lit = []
    while i < n:
        ch = body[i]
        if ch == "\\" and i + 1 < n and body[i + 1] in '"\\$`':
            lit.append(body[i + 1])
            i += 2
        elif ch == "$" and body[i:i + 2] == "$(":
            if lit:
                pieces.append(("lit", "".join(lit)))
                lit = []
            end = _skip_paren(body, i + 1)
            pieces.append(("sub", body[i + 2:end - 1]))
            i = end
        elif ch == "`":
            if lit:
                pieces.append(("lit", "".join(lit)))
                lit = []
            end = body.find("`", i + 1)
            end = n if end == -1 else end + 1
            pieces.append(("sub", body[i + 1:end - 1]))
            i = end
        elif ch == "$":
            if lit:
                pieces.append(("lit", "".join(lit)))
                lit = []
            spec, i = _read_var(body, i)
            pieces.append(("var", spec))
        else:
            lit.append(ch)
            i += 1
    if lit:
        pieces.append(("lit", "".join(lit)))
    return pieces


def _leading_assignments(tokens: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    assignments = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "export":
            i += 1
            continue
        found = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", token, re.S)
        if found:
            assignments.append((found.group(1), found.group(2)))
            i += 1
        else:
            break
    return assignments, tokens[i:]


def _windows_tokens(segment: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(segment)
    while i < n:
        ch = segment[i]
        if ch in " \t":
            if buf:
                tokens.append("".join(buf))
                buf = []
            i += 1
        elif ch in "'\"":
            end = _skip_quote(segment, i, ch)
            buf.append(segment[i:end])
            i = end
        elif ch == "`" and i + 1 < n:
            buf.append(segment[i + 1])
            i += 2
        else:
            buf.append(ch)
            i += 1
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t]


def _windows_pieces(token: str):
    pieces = []
    i = 0
    n = len(token)
    lit = []

    def flush():
        if lit:
            pieces.append(("lit", "".join(lit)))
            lit.clear()

    while i < n:
        ch = token[i]
        if ch == "'":
            end = _skip_quote(token, i, "'")
            flush()
            pieces.append(("squote", token[i + 1:end - 1]))
            i = end
        elif ch == '"':
            end = _skip_quote(token, i, '"')
            flush()
            pieces.append(("lit", token[i + 1:end - 1]))
            i = end
        elif ch == "$" and token[i:i + 2] == "$(":
            flush()
            end = _skip_paren(token, i + 1)
            pieces.append(("sub", token[i + 2:end - 1]))
            i = end
        elif ch == "$":
            flush()
            spec, i = _read_win_var(token, i)
            pieces.append(("var", spec))
        elif ch == "%" and (found := re.match(r"%([A-Za-z_][A-Za-z0-9_()]*)%", token[i:])):
            flush()
            pieces.append(("var", found.group(1)))
            i += found.end()
        else:
            lit.append(ch)
            i += 1
    flush()
    return pieces


def _read_win_var(token: str, i: int) -> tuple[str, int]:
    n = len(token)
    if token[i:i + 2] == "${":
        end = token.find("}", i + 2)
        end = n if end == -1 else end
        return token[i + 2:end], (end + 1 if end < n else n)
    j = i + 1
    while j < n and (token[j].isalnum() or token[j] in "_:"):
        j += 1
    return token[i + 1:j], j
