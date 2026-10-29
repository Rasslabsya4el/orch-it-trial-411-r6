# Project instructions: no literal for a value that varies

A value that differs per repo, account, machine or OS is a setting with its default, never a literal in code, tests, rules or tickets.

Members of the rule. Each one of these is a value that differs, so each is a setting:

- an account, owner or repo name;
- a machine's host name, home folder, user name or projects folder;
- a git name or e-mail;
- a command or call that exists on one OS only, unless it sits behind an OS check with a path for the others;
- a bare `python` (the interpreter is found, not assumed);
- a GitHub check or job name written as a literal;
- state that lives on one machine (a lock, a queue, a file only one machine sees).

A literal of one of these kinds is allowed only in the settings file, in records (research, the brain's records, citations in ADRs) and in one exceptions list where each entry carries its reason. A path inside the repo stays allowed.

Test data uses invented names: another account, another repo, a folder with another name, never the owner's.

Every design answers the question "and on a second machine, under another account, on another OS?" before it is written.

This is the whole rule, and a question about it is answered from this text.
