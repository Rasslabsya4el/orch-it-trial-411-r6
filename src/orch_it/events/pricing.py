"""The dollar price of one model turn from the price file `prices.json` (#393 decision 6)."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from importlib import resources


@cache
def _table() -> tuple[dict, str]:
    raw = resources.files("orch_it.events").joinpath("prices.json").read_bytes()
    return json.loads(raw)["models"], "prices-" + hashlib.sha256(raw).hexdigest()[:12]


TOKEN_KINDS = ("input", "output", "cache_read", "cache_write", "cache_write_1h")


def missing_rates(model: str) -> list[str]:
    """What the price file lacks to price any turn of `model`: `["model"]` when it has no row,
    else the token kinds with no rate (in every tier, for a model with tiers); empty when every
    turn of it can be priced (#399, the brain's answer on prices)."""

    models, _ = _table()
    entry = models.get(model)
    if entry is None:
        return ["model"]
    tiers = [entry["tiers"]["up_to"], entry["tiers"]["above"]] if "tiers" in entry else [entry]
    return [kind for kind in TOKEN_KINDS if any(kind not in rates for rates in tiers)]


def price_turn(model: str | None, usage: dict) -> tuple[float | None, str]:
    """(usd, price table id) of a turn of `model` with `usage` (`input_tokens`,
    `output_tokens`, `cache_read_tokens`, `cache_write_tokens` (5-minute and 1-hour together),
    `cache_write_1h_tokens`). usd is None when the file has no rate for the model or for a
    token kind the turn used."""

    models, table_id = _table()
    entry = models.get(model or "")
    if entry is None:
        return None, table_id
    write = usage.get("cache_write_tokens", 0)
    write_1h = min(usage.get("cache_write_1h_tokens", 0), write)
    tokens = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cache_read": usage.get("cache_read_tokens", 0),
        "cache_write": write - write_1h,
        "cache_write_1h": write_1h,
    }
    rates = entry
    if "tiers" in entry:
        tiers = entry["tiers"]
        prompt = tokens["input"] + tokens["cache_read"] + write
        rates = tiers["above"] if prompt > tiers["prompt_bound"] else tiers["up_to"]
    total = 0.0
    for kind, count in tokens.items():
        if not count:
            continue
        if kind not in rates:
            return None, table_id
        total += count * rates[kind]
    return total / 1_000_000, table_id
