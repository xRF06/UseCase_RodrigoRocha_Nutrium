"""Prompt templates for generation and critique nodes."""

from __future__ import annotations

from src.models import FoodList, PatientProfile


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_food_lists(food_lists: list[FoodList]) -> str:
    """Pretty-print food lists for the LLM prompt."""
    sections: list[str] = []
    for fl in food_lists:
        if not fl.equivalents:
            continue  # skip fully-filtered lists
        items = "\n".join(
            f"  - {eq.description} (ID: {eq.food_id})"
            for eq in fl.equivalents
        )
        try:
            kcal_val = float(fl.energy.split()[0])
        except (ValueError, IndexError):
            kcal_val = 0.0
        section = (
            f"### {fl.name}\n"
            f"Energy per serving: {fl.energy}  ← add this many kcal per selection\n"
            f"Macros per serving: Protein {fl.macronutrients_in_grams.protein}g | "
            f"Carbs {fl.macronutrients_in_grams.carbohydrate}g | "
            f"Fat {fl.macronutrients_in_grams.fat}g\n"
            f"Fiber per serving: {fl.fiber_quantity_in_grams}g\n"
            f"Available items (choose ONE per group):\n{items}"
        )
        sections.append(section)
    return "\n\n".join(sections)


def _calorie_guidance(patient: PatientProfile, food_lists: list[FoodList]) -> str:
    """Compute how many food-group servings the LLM needs to reach the calorie goal."""
    if not food_lists:
        return ""
    total_kcal = 0.0
    count = 0
    for fl in food_lists:
        if not fl.equivalents:
            continue
        try:
            total_kcal += float(fl.energy.split()[0])
            count += 1
        except (ValueError, IndexError):
            pass
    if count == 0:
        return ""
    avg_kcal = total_kcal / count
    needed = round(patient.dee_goal / avg_kcal)
    return (
        f"\n### Calorie Planning Guide\n"
        f"- Target: {patient.dee_goal} kcal/day\n"
        f"- Average kcal per food-list serving: ~{avg_kcal:.0f} kcal\n"
        f"- You must select approximately **{needed} food-group servings** total "
        f"across all meals to reach the target.\n"
        f"- Distribute them across meals. A typical main meal (Lunch/Dinner) "
        f"needs 3–5 servings; a snack needs 1–3 servings.\n"
        f"- Crucial: If you want to increase or decrease the standard dose, use the `(Multiplier: X)` tag at the very end of the item string! (e.g., `(Multiplier: 1.5)` or `(Multiplier: 2.0)`).\n"
    )


def _format_patient_summary(patient: PatientProfile, food_lists: list[FoodList] | None = None) -> str:
    """Create a concise, structured patient summary for the LLM."""
    dh = patient.patient_infos.dietary_history
    mh = patient.patient_infos.medical_history

    diary_text = ""
    for entry in patient.patient_infos.food_diary_history_and_obs:
        meals_str = "\n".join(f"    - {m.meal_type}: {m.text}" for m in entry.meals)
        diary_text += f"  Date: {entry.date}\n{meals_str}\n"
        if entry.observations:
            diary_text += f"    Observations: {entry.observations}\n"

    behaviours_text = ""
    if patient.patient_infos.eating_behaviours:
        behaviours_text = "\n".join(
            f"  - [{eb.date}] {eb.text}"
            for eb in patient.patient_infos.eating_behaviours[:5]  # last 5
        )

    calorie_hint = _calorie_guidance(patient, food_lists) if food_lists else ""

    return f"""## Patient: {patient.patient_name}

### Nutritional Targets
- Daily Calorie Goal: {patient.dee_goal} {patient.dee_goal_unit}  ← PRIMARY target (must be within ±10%)
- Protein: {patient.macronutrient_distribution_in_grams.protein}g (secondary goal)
- Carbohydrates: {patient.macronutrient_distribution_in_grams.carbohydrate}g (secondary goal)
- Fat: {patient.macronutrient_distribution_in_grams.fat}g (secondary goal)
- Fiber: {patient.fiber_quantity_in_grams}g (secondary goal)
{calorie_hint}
### Schedule
- Wake-up: {dh.wake_up_time_24h or 'Not specified'}
- Bedtime: {dh.bedtime_24h or 'Not specified'}

### Medical History
- Diseases: {mh.diseases.details or 'None'}
- Medications: {mh.medications or 'None'}

### Dietary Preferences
- Diet type: {dh.diet_types.details or 'No specific diet'}
- Favorite foods: {dh.favorite_foods or 'Not specified'}
- Disliked foods: {dh.disliked_foods or 'None'}
- Allergies: {dh.food_allergies.details or 'None'}
- Intolerances: {dh.food_intolerances.details or 'None'}
- Deficiencies: {dh.nutritional_deficiencies.details or 'None'}

### Food Diary (eating habits)
{diary_text or '  Not available'}

### Recent Eating Behaviours
{behaviours_text or '  Not available'}
"""


# ── Generation Prompt ─────────────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = """\
You are a clinical nutrition AI assistant. Your task is to create a personalised \
1-day meal plan for a patient.

RULES (strictly follow):
1. Include only the meals that match the patient's actual daily habits. \
Allowed meal types (use EXACTLY these English names):
   - Breakfast
   - Morning Snack
   - Lunch
   - Afternoon Snack
   - Dinner
   - Pre-Workout Snack
   - Post-Workout Snack
   If the patient's diary shows they skip Breakfast, do NOT include it. \
Do NOT invent meals the patient does not have. At least Lunch must be present.
2. For each meal, select MULTIPLE food groups from the food alternative lists \
to reach the calorie target. A main meal (Lunch/Dinner) typically needs 3–5 \
food groups; a snack needs 1–3. To add multiple servings of the same food group, \
add it multiple times as separate entries in the `items` array.
3. Every food item MUST include its Food ID in the format (ID: XXXXX).
4. For each selected food group, output up to TWO alternatives from the group, separated by " OR ". \
If a group only has one safe alternative left, just output that one. \
You MUST specify a multiplier at the end to adjust the serving size. The multiplier MUST be a multiple of 0.5 (e.g., 0.5, 1.0, 1.5, 2.0, 2.5). Do NOT use other decimal values like 1.2 or 1.4. \
Format: "[Qty] [Unit] of [Food] ([Weight] g) (ID: [ID]) OR [Qty] [Unit] of [Food] ([Weight] g) (ID: [ID]) (Multiplier: [number])"
5. Meal totals: write your best estimate. They will be verified and corrected \
by the system, so focus on choosing the RIGHT AMOUNT of food items, not perfect maths.
6. daily_totals: write the sum of all meal_totals.
7. PRIMARY goal: daily total kcal must be within ±10% of the patient's calorie goal. \
Select enough food groups to reach it — check the Calorie Planning Guide.
8. Assign realistic meal times based on the patient's wake/sleep schedule.
9. Do NOT include any food items that are disliked, allergenic, or \
contraindicated. All unsafe items have already been filtered out of the lists.
10. Try to align food choices with the patient's food diary and preferences.

OUTPUT FORMAT: Respond ONLY with a valid JSON object matching this schema \
(no markdown, no explanation, no extra text):

{
  "daily_totals": {
    "kcal": <float>,
    "protein_g": <float>,
    "carbs_g": <float>,
    "fat_g": <float>,
    "fiber_g": <float>
  },
  "meals": [
    {
      "meal_type": "<string>",
      "time": "<HH:MM>",
      "items": [
        "<choice group string with alternatives separated by OR, each with (ID: XXXXX)>"
      ],
      "meal_totals": {
        "kcal": <float>,
        "protein_g": <float>,
        "carbs_g": <float>,
        "fat_g": <float>,
        "fiber_g": <float>
      }
    }
  ]
}
"""


def build_generation_user_prompt(
    patient: PatientProfile,
    food_lists: list[FoodList],
    prior_errors: list[str] | None = None,
    critique_feedback: str = "",
) -> str:
    """Build the user-facing generation prompt, optionally including prior errors for retry."""
    base = (
        f"{_format_patient_summary(patient, food_lists)}\n\n"
        f"## Available Food Alternative Lists (pre-filtered for safety)\n\n"
        f"{_format_food_lists(food_lists)}\n\n"
    )
    if prior_errors:
        import re
        # On retries, tell the LLM exactly what was wrong
        error_block = "\n".join(f"  - {e}" for e in prior_errors)
        
        # Calculate dynamic scaling tip based on how much it missed the calories
        ratio_tip = ""
        err_str = " ".join(prior_errors)
        match = re.search(r'goal=([\d.]+), plan=([\d.]+)', err_str)
        if match:
            goal_kcal = float(match.group(1))
            plan_kcal = float(match.group(2))
            if plan_kcal > 0:
                ratio = goal_kcal / plan_kcal
                
                if ratio > 1.1:
                    ratio_tip = (
                        f"You are significantly UNDER the calorie target. You must SCALE UP your CURRENT food quantities by ~**{ratio:.1f}x**!\n"
                        f"To reach {goal_kcal} kcal, increase your current multipliers on almost all items!\n"
                        f"For example: if you used `(Multiplier: 1.0)`, change it to `(Multiplier: 1.5)` or `2.0`. "
                        f"If you used `(Multiplier: 1.5)`, increase it to `2.0` or `2.5`.\n"
                        f"CRITICAL: Multipliers MUST be multiples of 0.5 (e.g. 1.0, 1.5, 2.0)."
                    )
                elif ratio < 0.9:
                    ratio_tip = (
                        f"You are significantly OVER the calorie target. You must SCALE DOWN your CURRENT food quantities by ~**{ratio:.1f}x**!\n"
                        f"To reach {goal_kcal} kcal, decrease your current multipliers on almost all items!\n"
                        f"For example: if you used `(Multiplier: 2.0)`, change it to `1.5` or `1.0`. "
                        f"If you used `(Multiplier: 1.5)`, decrease it to `1.0` or `0.5`.\n"
                        f"CRITICAL: Multipliers MUST remain multiples of 0.5 (e.g. 0.5, 1.0, 1.5)."
                    )

        base += (
            f"## ⚠️ Previous Attempt Errors (you MUST fix ALL of these)\n"
            f"{error_block}\n\n"
            f"---\n"
            f"💡 **CRITICAL TIP FOR FIXING CALORIES:**\n"
            f"{ratio_tip or 'You MUST change your `(Multiplier: X)` values to hit the calorie target exactly.'}\n"
            f"Do NOT ignore this! If you don't scale the multipliers, you will fail the patient.\n"
            f"---\n\n"
        )
    if critique_feedback:
        base += (
            f"## 🧐 Reviewer Feedback from Previous Attempt\n"
            f"{critique_feedback}\n\n"
        )
    base += "Generate the 1-day meal plan as a JSON object now."
    return base


# ── Critique Prompt ───────────────────────────────────────────────────────────

CRITIQUE_SYSTEM_PROMPT = """\
You are a clinical nutrition reviewer. Your role is ONLY to check objective, \
measurable criteria. Do NOT apply personal judgment or extra rules beyond those listed.

HARD CHECKS — a plan FAILS if any of these are violated:
1. STRUCTURE: Lunch must be present. All meal names must be one of:
   Breakfast, Morning Snack, Lunch, Afternoon Snack, Dinner, \
Pre-Workout Snack, Post-Workout Snack.
   Times must be in HH:MM format.
2. FOOD IDs: Every food item string must contain at least one (ID: XXXXX) tag.

SOFT CHECKS — add to suggestions only, do NOT use to reject:
- Macro balance (protein/carbs/fat percentages)
- Meal timing preferences
- Number of meals

IMPORTANT RULES:
- Assume the math and calorie targets are ALWAYS correct (they are validated by the system). DO NOT reject a plan for math or calorie reasons.
- Do NOT penalize for missing Breakfast if the patient's diary says they skip it.
- Set "approved": true if all HARD CHECKS pass, regardless of soft issues.

RESPOND with a JSON object:
{
  "approved": true/false,
  "issues": ["only list HARD CHECK failures here"],
  "suggestions": ["soft improvement suggestions here"]
}

Respond ONLY with valid JSON. No extra text.
"""


def build_critique_user_prompt(
    plan_json: str,
    patient: PatientProfile,
) -> str:
    dh = patient.patient_infos.dietary_history
    skips_breakfast = any(
        "não toma" in m.text.lower() or "sem café" in m.text.lower()
        for entry in patient.patient_infos.food_diary_history_and_obs
        for m in entry.meals
        if "café" in m.meal_type.lower() or "breakfast" in m.meal_type.lower()
    )
    skip_note = (
        "\nNOTE: Patient's diary explicitly states they skip breakfast — "
        "do NOT penalise for its absence.\n"
        if skips_breakfast else ""
    )
    return (
        f"## Patient Profile\n\n"
        f"- Name: {patient.patient_name}\n"
        f"- Calorie Goal (dee_goal): {patient.dee_goal} {patient.dee_goal_unit}\n"
        f"- Acceptable calorie range (±10%): "
        f"{patient.dee_goal * 0.90:.0f} – {patient.dee_goal * 1.10:.0f} kcal\n"
        f"- Disliked foods: {dh.disliked_foods or 'None'}\n"
        f"- Allergies: {dh.food_allergies.details or 'None'}\n"
        f"- Intolerances: {dh.food_intolerances.details or 'None'}\n"
        f"{skip_note}\n"
        f"## Generated Meal Plan\n\n"
        f"```json\n{plan_json}\n```\n\n"
        f"Apply ONLY the HARD CHECKS from your instructions. Evaluate this plan now."
    )
