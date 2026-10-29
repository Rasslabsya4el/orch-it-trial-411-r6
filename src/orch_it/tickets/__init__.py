"""Tickets of the new layer (#398): the one ticket model and its shape rules.

`parse` reads a body into a `Ticket`; `violations` returns every shape rule it breaks. Every
other part reads tickets only through these two."""

from orch_it.tickets.model import AcceptanceRow, HumanField, Ticket, parse
from orch_it.tickets.shape import RULES, Violation, violations

__all__ = ["AcceptanceRow", "HumanField", "RULES", "Ticket", "Violation", "parse", "violations"]
