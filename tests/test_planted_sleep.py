"""Planted for #411 row 6 (throwaway repo): a test that sleeps past the job's bound; timeout(0) lifts the per-test bound."""
import time

import pytest


@pytest.mark.timeout(0)
def test_planted_sleep_past_the_job_bound():
    time.sleep(1200)
