"""The trial's green tests: two trivial passing tests, so the workflow's pytest run is green."""


def test_green_addition():
    assert 1 + 1 == 2


def test_green_string():
    assert "orch" + "-it" == "orch-it"
