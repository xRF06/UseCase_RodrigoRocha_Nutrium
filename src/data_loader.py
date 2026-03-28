"""Loaders for patient profiles and food alternative lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.models import FoodEquivalent, FoodList, PatientProfile

_DATA_DIR = Path(__file__).resolve().parent.parent / "UseCase"


def load_patient_profiles(path: Path | None = None) -> List[PatientProfile]:
    """Load all patient profiles from the JSONL file."""
    path = path or _DATA_DIR / "input_nutri_approval.jsonl"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [PatientProfile.model_validate(p) for p in raw]


def load_food_lists(path: Path | None = None) -> List[FoodList]:
    """Load all food alternative lists from the JSONL file."""
    path = path or _DATA_DIR / "input_lists.jsonl"
    raw = json.loads(path.read_text(encoding="utf-8"))
    food_lists: List[FoodList] = []
    for item in raw:
        equivalents = [
            FoodEquivalent(food_id=eq[0], description=eq[1])
            for eq in item["equivalents"]
        ]
        food_lists.append(
            FoodList(
                name=item["name"],
                energy=item["energy"],
                macronutrients_in_grams=item["macronutrients_in_grams"],
                fiber_quantity_in_grams=item["fiber_quantity_in_grams"],
                equivalents=equivalents,
            )
        )
    return food_lists
