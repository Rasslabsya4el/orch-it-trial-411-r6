"""pytest plugin `-p orch_it_ids`: runs only the tests named in the file `ORCH_IT_IDS_FILE`.

Standalone on purpose (stdlib and pytest only, no import of orch_it): a test place puts this
folder on the test process's `PYTHONPATH`, also on a machine where orch-it is not installed.
The id rule is the outcome module's (`orch_it.testruns.outcome`), so a test this plugin keeps is
a test the run's judge finds, and the reverse; a test of each checks the two agree.

Each line of the file is an id (blank lines and `#` lines skipped; `\\` reads as `/`, a leading
`./` is dropped): a pytest node id, or the start of one up to a `::` (a file, a class) or a `[`
(every parametrization of a test); a JUnit test id as `test.result` writes it
(`tests.test_x.TestC::test_y`); or a bare file, folder or dotted module standing for every test
under it. A node id splits on `::` only before its first `[`, as pytest's JUnit report does. A
collected test that no id names is deselected; an id that names no test is no error here: the
run's report then lacks it, and the layer reads that as `ids_not_collected` (#394 decision 2).
With the variable unset, nothing is filtered.
"""

from __future__ import annotations

import os

import pytest

ENV_IDS = "ORCH_IT_IDS_FILE"


def _ids() -> list[str] | None:
    path = os.environ.get(ENV_IDS)
    if not path:
        return None
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith("#")]


def normal(wanted: str) -> str:
    wanted = wanted.strip().replace("\\", "/")
    while wanted.startswith("./"):
        wanted = wanted[2:]
    return wanted


def module_of(path: str) -> str:
    path = path.strip("/")
    return (path[:-3] if path.endswith(".py") else path).replace("/", ".")


def junit_parts(node_id: str) -> tuple[str, str]:
    """A node id as pytest's JUnit report writes it: (classname, name)."""

    base, bracket, params = node_id.partition("[")
    names = base.split("::")
    names[-1] += bracket + params
    return ".".join([module_of(names[0]), *names[1:-1]]), names[-1]


def selects(wanted: str, node_id: str) -> bool:
    """Does the (normal) id `wanted` name the test `node_id`?"""

    if wanted == node_id or node_id.startswith((wanted + "::", wanted + "[")):
        return True
    classname, name = junit_parts(node_id)
    if wanted == f"{classname}::{name}" or wanted.startswith(f"{classname}::") and name.startswith(
            wanted[len(classname) + 2:] + "["):
        return True
    file = node_id.partition("::")[0]
    if "::" not in wanted:
        if file == wanted.rstrip("/") or file.startswith(wanted.rstrip("/") + "/"):
            return True
        dotted = module_of(wanted)
        return classname == dotted or classname.startswith(dotted + ".")
    return False


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    wanted = _ids()
    if wanted is None:
        return
    wanted = [normal(w) for w in wanted]
    keep, drop = [], []
    for item in items:
        (keep if any(selects(w, item.nodeid) for w in wanted) else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
