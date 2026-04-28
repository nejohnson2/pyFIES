"""Canonical FIES item names and short descriptions."""

from __future__ import annotations

DEFAULT_FIES_ITEMS: tuple[str, ...] = (
    "WORRIED",
    "HEALTHY",
    "FEWFOOD",
    "SKIPPED",
    "ATELESS",
    "RUNOUT",
    "HUNGRY",
    "WHLDAY",
)
"""The eight FIES questions, in the order used by FAO's global standard.

Each item asks whether, during the past 12 months, the respondent (or their
household) experienced the condition described, because of lack of money or
other resources. Responses are dichotomized: ``Never`` -> 0, otherwise -> 1.
"""

ITEM_DESCRIPTIONS: dict[str, str] = {
    "WORRIED": "Worried about running out of food",
    "HEALTHY": "Unable to eat healthy and nutritious food",
    "FEWFOOD": "Ate only a few kinds of food",
    "SKIPPED": "Had to skip a meal",
    "ATELESS": "Ate less than they thought they should",
    "RUNOUT": "Household ran out of food",
    "HUNGRY": "Was hungry but did not eat",
    "WHLDAY": "Went without eating for a whole day",
}
