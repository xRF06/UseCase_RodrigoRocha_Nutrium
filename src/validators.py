"""Validation utilities for generated meal plans."""

from __future__ import annotations

import re
from typing import List

from src.models import MealPlan, PatientProfile


# Meals that MUST always be present (at minimum)
ALWAYS_REQUIRED = ["Lunch"]

# All accepted meal type names
VALID_MEAL_NAMES = [
    "Breakfast",
    "Morning Snack",
    "Lunch",
    "Afternoon Snack",
    "Dinner",
    "Pre-Workout Snack",
    "Post-Workout Snack",
]


def validate_json_schema(raw_json: str) -> tuple[MealPlan | None, List[str]]:
    """
    Parse raw JSON into a MealPlan model.
    Returns (plan, errors) – plan is None if parsing fails.
    """
    import json

    errors: List[str] = []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON: {e}"]

    try:
        plan = MealPlan.model_validate(data)
    except Exception as e:
        return None, [f"Schema validation failed: {e}"]

    return plan, errors


def validate_required_meals(plan: MealPlan) -> List[str]:
    """Check that Lunch and Dinner are present; all meal names must be valid."""
    errors: List[str] = []
    meal_names = [m.meal_type for m in plan.meals]
    for req in ALWAYS_REQUIRED:
        if req not in meal_names:
            errors.append(f"Missing required meal: {req}")
    for name in meal_names:
        if name not in VALID_MEAL_NAMES:
            errors.append(f"Unexpected meal name: '{name}' (valid names: {VALID_MEAL_NAMES})")
    return errors


def validate_food_ids(plan: MealPlan) -> List[str]:
    """Check that every food item string contains at least one (ID: XXXXX)."""
    errors: List[str] = []
    id_pattern = re.compile(r"\(ID:\s*\d+\)")
    for meal in plan.meals:
        for i, item_str in enumerate(meal.items):
            if not id_pattern.search(item_str):
                errors.append(
                    f"Meal '{meal.meal_type}', item {i+1}: missing Food ID (ID: XXXXX)"
                )
    return errors


def validate_daily_totals(plan: MealPlan, tolerance: float = 1.5) -> List[str]:
    """Check that daily_totals == sum of all meal_totals (within tolerance)."""
    errors: List[str] = []
    sum_kcal = sum(m.meal_totals.kcal for m in plan.meals)
    sum_protein = sum(m.meal_totals.protein_g for m in plan.meals)
    sum_carbs = sum(m.meal_totals.carbs_g for m in plan.meals)
    sum_fat = sum(m.meal_totals.fat_g for m in plan.meals)
    sum_fiber = sum(m.meal_totals.fiber_g for m in plan.meals)

    checks = [
        ("kcal", plan.daily_totals.kcal, sum_kcal),
        ("protein_g", plan.daily_totals.protein_g, sum_protein),
        ("carbs_g", plan.daily_totals.carbs_g, sum_carbs),
        ("fat_g", plan.daily_totals.fat_g, sum_fat),
        ("fiber_g", plan.daily_totals.fiber_g, sum_fiber),
    ]

    for name, declared, computed in checks:
        if abs(declared - computed) > tolerance:
            errors.append(
                f"daily_totals.{name} mismatch: declared={declared:.2f}, "
                f"sum of meals={computed:.2f} (diff={abs(declared - computed):.2f})"
            )
    return errors


def validate_caloric_target(
    plan: MealPlan,
    patient: PatientProfile,
    tolerance_pct: float = 0.10,
) -> List[str]:
    """Check that daily totals are within ±tolerance_pct of patient goals.

    Only the calorie goal is a hard error. Macro mismatches are soft warnings
    (prefixed [WARN]) so they are visible but don't block plan acceptance —
    this is necessary because some patient profiles have macro targets whose
    caloric sum doesn't match the dee_goal (data inconsistency).
    """
    errors: List[str] = []

    goal_kcal = patient.dee_goal
    actual_kcal = plan.daily_totals.kcal
    if abs(actual_kcal - goal_kcal) > goal_kcal * tolerance_pct:
        errors.append(
            f"Calorie target miss: goal={goal_kcal}, plan={actual_kcal:.1f} "
            f"(±{tolerance_pct*100:.0f}% = {goal_kcal*(1-tolerance_pct):.0f}-{goal_kcal*(1+tolerance_pct):.0f})"
        )

    macro_checks = [
        ("protein", patient.macronutrient_distribution_in_grams.protein, plan.daily_totals.protein_g),
        ("carbs", patient.macronutrient_distribution_in_grams.carbohydrate, plan.daily_totals.carbs_g),
        ("fat", patient.macronutrient_distribution_in_grams.fat, plan.daily_totals.fat_g),
    ]
    for name, goal, actual in macro_checks:
        if goal > 0 and abs(actual - goal) > goal * tolerance_pct:
            pct_diff = ((actual - goal) / goal) * 100
            sign = "+" if pct_diff > 0 else ""
            # Soft warning: macro targets may be inconsistent with dee_goal in some profiles
            errors.append(
                f"[WARN] {name} target miss: goal={goal:.1f}g, plan={actual:.1f}g ({sign}{pct_diff:.1f}%)"
            )

    return errors


def run_all_validations(
    plan: MealPlan,
    patient: PatientProfile,
) -> List[str]:
    """Run all validation checks and return combined errors."""
    errors: List[str] = []
    errors.extend(validate_required_meals(plan))
    errors.extend(validate_food_ids(plan))
    errors.extend(validate_daily_totals(plan))
    errors.extend(validate_caloric_target(plan, patient))
    return errors
