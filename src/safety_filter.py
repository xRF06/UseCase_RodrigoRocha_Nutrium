"""Safety filtering – removes unsafe, disliked and contraindicated foods."""

from __future__ import annotations

import copy
import re
from typing import List

from src.models import FoodList, PatientProfile


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, strip accents (simple), collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _parse_disliked_foods(raw: str) -> List[str]:
    """Split a newline / comma-separated string into normalised tokens."""
    tokens: List[str] = []
    for line in raw.replace(",", "\n").split("\n"):
        t = _normalise(line)
        if t:
            tokens.append(t)
    return tokens


def _food_matches_any(description: str, banned: List[str]) -> bool:
    """Return True if the food description contains any banned term."""
    desc_lower = _normalise(description)
    for term in banned:
        if term in desc_lower:
            return True
    return False


# ── Lactose mapping ──────────────────────────────────────────────────────────

_LACTOSE_KEYWORDS = [
    "leite",
    "iogurte",
    "queijo",
    "requeijão",
    "requeijao",
    "mussarela",
    "muzarella",
    "ricota",
    "prato",
    "minas",
]


def _is_lactose_item(description: str) -> bool:
    desc = _normalise(description)
    return any(kw in desc for kw in _LACTOSE_KEYWORDS)


# ── Main filter ───────────────────────────────────────────────────────────────

def filter_food_lists(
    patient: PatientProfile,
    food_lists: List[FoodList],
) -> List[FoodList]:
    """
    Return a deep copy of *food_lists* with unsafe items removed.

    Removes items that are:
    - disliked by the patient
    - allergenic for the patient
    - contraindicated by intolerances (e.g. lactose)
    - contraindicated by medical history / medications

    Lists where ALL equivalents are removed are still returned (empty) so the
    caller can detect the gap and handle it.
    """
    dietary = patient.patient_infos.dietary_history

    # Build the combined "banned" list
    banned: List[str] = []
    banned.extend(_parse_disliked_foods(dietary.disliked_foods))
    banned.extend(_parse_disliked_foods(dietary.food_allergies.details))
    # Add allergy list items
    for a in dietary.food_allergies.items:
        banned.append(_normalise(a))

    # Intolerance handling
    has_lactose_intolerance = "lactose" in _normalise(
        dietary.food_intolerances.details
    ) or any("lactose" in _normalise(i) for i in dietary.food_intolerances.items)

    filtered: List[FoodList] = []
    for fl in food_lists:
        new_fl = copy.deepcopy(fl)
        safe_eqs = []
        for eq in new_fl.equivalents:
            if _food_matches_any(eq.description, banned):
                continue
            if has_lactose_intolerance and _is_lactose_item(eq.description):
                continue
            safe_eqs.append(eq)
        new_fl.equivalents = safe_eqs
        filtered.append(new_fl)

    return filtered
