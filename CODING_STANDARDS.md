# Coding standard: no literal for a value that varies

A value that differs per repo, account, machine or OS is a setting with its default, never a literal in code, tests, rules or tickets.

A finding cites this standard by its first sentence, word for word.

## What is allowed

A literal of one of the seven kinds below is allowed only in the settings file, in records (research, the brain's records, citations in ADRs) and in one exceptions list where each entry carries its reason. A path inside the repo stays allowed. A finding on a literal outside those places is a break of this standard; the reviewer cites the first sentence.

Test data uses invented names only: another account, another repo, a folder with another name, never the owner's. The examples below use the invented account `acme-dev`, the home folder of the invented user `jdoe` (written `<home>/jdoe`, because the private-data scan refuses a real-looking home path even in an example) and the host `build-box-7`.

Every design answers the question "and on a second machine, under another account, on another OS?" before it is written.

The seven kinds, each with a Wrong and a Right example; the one marked Wrong is wrong, the one marked Right is the form to write.

## 1. An account, owner or repo name

Wrong:

```python
REPO = "acme-dev/orch-tools"
```

Right:

```python
REPO = settings["repo"]  # the default is written in the settings file
```

## 2. A machine's host name, home folder, user name or projects folder

Wrong:

```python
ROOT = Path("<home>/jdoe/projects/orch-tools")  # an absolute path under one user's home folder
```

Right:

```python
ROOT = Path(settings["projects_folder"]) / settings["repo_folder"]  # both defaults are in the settings file
```

## 3. A git name or e-mail

Wrong:

```python
git("-c", "user.name=Pat Example", "-c", "user.email=pat@example.test", "commit", "-m", message)
```

Right:

```python
git("commit", "-m", message)  # the machine's own git config names the author, as for every commit
```

## 4. A command or call that exists on one OS only

Wrong (`taskkill` exists on Windows only):

```python
subprocess.run(["taskkill", "/F", "/PID", str(pid)])
```

Right (one call on every OS; an OS check with a path for each other OS is the other allowed form):

```python
psutil.Process(pid).kill()
```

## 5. A bare `python`

Wrong:

```python
subprocess.run(["python", "-m", "pytest", "-q"])
```

Right (the interpreter running the code is found, not assumed):

```python
subprocess.run([sys.executable, "-m", "pytest", "-q"])
```

## 6. A GitHub check or job name written as a literal

Wrong:

```python
if check["name"] == "test (ubuntu-latest, 3.12)":
```

Right (the names live in the settings file, with their default there):

```python
if check["name"] in settings["required_checks"]:
```

## 7. State that lives on one machine

Wrong (a lock in a folder only this machine sees, written in the code):

```python
LOCK = Path("/tmp/merge-line.lock")
```

Right (the folder comes from the settings file; the docstring says the lock is per machine):

```python
LOCK = Path(settings["lock_folder"]) / "merge-line.lock"  # per machine: each machine has its own lock folder
```
