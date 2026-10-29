"""The folder a test place puts on the test process's `PYTHONPATH`, so `-p orch_it_ids` (an ids
file's selection, #394 decision 2's `ids_not_collected` row) loads at any place."""

from pathlib import Path

FOLDER = Path(__file__).parent
