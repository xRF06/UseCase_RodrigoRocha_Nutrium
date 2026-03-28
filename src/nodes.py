"""LangGraph node functions for the meal plan generation workflow."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from src.models import FoodList, GraphState
from src.safety_filter import filter_food_lists
from src.prompts import (
    GENERATION_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    build_generation_user_prompt,
    build_critique_user_prompt,
)
from src.validators import run_all_validations, validate_json_schema


# ── Deterministic total recalculator ─────────────────────────────────────────

def _format_qty(val: float) -> str:
    """Helper to convert decimals into neat Portuguese format (e.g. 1,5 or 0,75)."""
    if val == int(val):
        return str(int(val))
    return f"{val:g}".replace(".", ",")

def _apply_multiplier_to_string(item_str: str) -> str:
    """Visually updates the text quantities and weights using the LLM's multiplier, 
    removes the multiplier tag, and applies Portuguese grammatical rules/translations."""
    import re
    
    # Base translation / cleanup
    item_str = item_str.replace(" of ", " de ")
    item_str = item_str.replace(" grams ", " g ")
    
    mult_pattern = re.compile(r'\s*\(Multiplier:\s*([0-9.]+)\)', re.IGNORECASE)
    match_mult = mult_pattern.search(item_str)
    
    multiplier = 1.0
    if match_mult:
        multiplier = float(match_mult.group(1))
        clean_str = mult_pattern.sub('', item_str)
    else:
        clean_str = item_str
        
    parts = clean_str.split(" OR ")
    new_parts = []
    
    plurals = {
        "unidade": "unidades", "fatia": "fatias", "escumadeira": "escumadeiras",
        "colher": "colheres", "concha": "conchas", "porção": "porções",
        "filé": "filés", "pedaço": "pedaços", "copo": "copos", "bife": "bifes",
        "ovo": "ovos"
    }

    for part in parts:
        part = part.strip()
        # Multiply QTY at the start: "1/2", "1.5", "3", "100"
        qty_pattern = re.compile(r'^(\d+(?:\.\d+)?|\d+/\d+)(\s+)')
        qty_match = qty_pattern.search(part)
        
        # Multiply Grams inside parentheses: "(33 g)", "(33.5 g)"
        gram_pattern = re.compile(r'\(\s*(\d+(?:\.\d+)?)\s*g\s*\)')
        
        new_part = part
        target_qty = 1.0
        effective_mult = multiplier  # Default to normal multiplier if no QTY exists
        
        if qty_match:
            qty_str = qty_match.group(1)
            if '/' in qty_str:
                num, den = qty_str.split('/')
                qty_val = float(num) / float(den)
            else:
                qty_val = float(qty_str)
            
            # Raw mathematically correct target, e.g., 0.5 * 1.5 = 0.75
            raw_target = qty_val * multiplier
            # Round forcibly to the nearest 0.5 step 
            target_qty = round(raw_target * 2) / 2.0
            if target_qty <= 0:
                target_qty = 0.5  # Prevent showing 0 units
                
            # If we rounded it, we must adjust the grams multiplier proportionally 
            # so the text remains internally coherent
            effective_mult = target_qty / qty_val
            
            new_qty_str = _format_qty(target_qty)
            new_part = new_part[:qty_match.start(1)] + new_qty_str + new_part[qty_match.end(1):]
            
        gram_match = gram_pattern.search(new_part)
        if gram_match:
            gram_val = float(gram_match.group(1))
            new_gram = gram_val * effective_mult
            new_gram_str = f"{new_gram:g}".replace(".", ",")
            new_part = new_part[:gram_match.start(1)] + new_gram_str + new_part[gram_match.end(1):]
            
        # Optional pluralization if target_qty is strictly greater than 1
        if target_qty > 1.0:
            for sing, plur in plurals.items():
                new_part = re.sub(rf'\b{sing}\b', plur, new_part)
                
        new_parts.append(new_part)
        
    return " OR ".join(new_parts)

def _recalculate_plan_totals(
    plan_dict: dict,
    food_lists: List[FoodList],
) -> dict:
    """
    Recompute meal_totals and daily_totals from food-list metadata.

    The LLM reliably gets the arithmetic wrong (it copies the patient's
    calorie goal into daily_totals instead of summing the meals it chose).
    This function makes the numbers deterministic:
      1. For each item string in a meal, extract ALL (ID: XXXXX) tags.
      2. Look up any matched ID in the food lists to get its macros.
      3. Sum the macros per meal → overwrite meal_totals.
      4. Sum all meals → overwrite daily_totals.

    Items whose IDs cannot be found (unlikely) keep contribute 0 – a warning
    is printed so the issue surfaces in the log.
    """
    id_pattern = re.compile(r'\(ID:\s*(\d+)\)')

    # Build lookup: food_id (int) → FoodList
    id_to_list: dict[int, FoodList] = {}
    for fl in food_lists:
        for eq in fl.equivalents:
            id_to_list[eq.food_id] = fl

    new_meals: list[dict] = []
    daily_kcal = daily_protein = daily_carbs = daily_fat = daily_fiber = 0.0

    for meal in plan_dict.get("meals", []):
        meal_kcal = meal_protein = meal_carbs = meal_fat = meal_fiber = 0.0

        for item_str in meal.get("items", []):
            ids = [int(m.group(1)) for m in id_pattern.finditer(item_str)]
            matched_fl: FoodList | None = None
            for fid in ids:
                if fid in id_to_list:
                    matched_fl = id_to_list[fid]
                    break

            if matched_fl is None:
                print(f"[⚠️ ] Could not find food list for IDs {ids} in: {item_str[:80]}")
                continue

            # Check if LLM specified a dose multiplier (e.g., "(Multiplier: 2.0)")
            mult_pattern = re.compile(r'\(Multiplier:\s*([0-9.]+)\)', re.IGNORECASE)
            match_mult = mult_pattern.search(item_str)
            multiplier = float(match_mult.group(1)) if match_mult else 1.0

            # Parse energy: e.g. "90.19 kcal"
            try:
                kcal_val = float(matched_fl.energy.split()[0])
            except (ValueError, IndexError):
                kcal_val = 0.0

            meal_kcal    += kcal_val * multiplier
            meal_protein += matched_fl.macronutrients_in_grams.protein * multiplier
            meal_carbs   += matched_fl.macronutrients_in_grams.carbohydrate * multiplier
            meal_fat     += matched_fl.macronutrients_in_grams.fat * multiplier
            meal_fiber   += matched_fl.fiber_quantity_in_grams * multiplier

        # Overwrite the LLM's (unreliable) meal totals with code-computed values
        new_meal = dict(meal)
        
        # Clean up the items strings to visually match the multiplier
        new_meal["items"] = [_apply_multiplier_to_string(item) for item in meal.get("items", [])]
        
        new_meal["meal_totals"] = {
            "kcal":      round(meal_kcal,    2),
            "protein_g": round(meal_protein, 2),
            "carbs_g":   round(meal_carbs,   2),
            "fat_g":     round(meal_fat,     2),
            "fiber_g":   round(meal_fiber,   2),
        }
        new_meals.append(new_meal)

        daily_kcal    += meal_kcal
        daily_protein += meal_protein
        daily_carbs   += meal_carbs
        daily_fat     += meal_fat
        daily_fiber   += meal_fiber

    new_plan = dict(plan_dict)
    new_plan["meals"] = new_meals
    new_plan["daily_totals"] = {
        "kcal":      round(daily_kcal,    2),
        "protein_g": round(daily_protein, 2),
        "carbs_g":   round(daily_carbs,   2),
        "fat_g":     round(daily_fat,     2),
        "fiber_g":   round(daily_fiber,   2),
    }
    return new_plan



# ── LLM singleton ────────────────────────────────────────────────────────────

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=4096,
        )
    return _llm


# ── Node: load_and_filter ─────────────────────────────────────────────────────

def load_and_filter(state: Dict[str, Any]) -> Dict[str, Any]:
    """Filter food lists for the patient's safety constraints."""
    from src.models import GraphState

    gs = GraphState.model_validate(state)
    print(f"[📋] Loading & filtering food lists for {gs.patient.patient_name}...")

    filtered = filter_food_lists(gs.patient, gs.food_lists)

    # Report filtering results
    total_before = sum(len(fl.equivalents) for fl in gs.food_lists)
    total_after = sum(len(fl.equivalents) for fl in filtered)
    removed = total_before - total_after
    msg = f"Filtered food lists: {total_after} items remaining ({removed} removed for safety)"
    print(f"[✅] {msg}")

    empty_lists = [fl.name for fl in filtered if not fl.equivalents]
    if empty_lists:
        print(f"[⚠️] Fully excluded lists (no safe items): {empty_lists}")

    return {
        **state,
        "filtered_food_lists": [fl.model_dump() for fl in filtered],
        "status": "ready",
        "messages": gs.messages + [msg],
    }


# ── Node: generate_plan ──────────────────────────────────────────────────────

def generate_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM to generate a meal plan."""
    from src.models import GraphState, FoodList

    gs = GraphState.model_validate(state)
    gs.attempt += 1
    print(f"\n[🔄] Generating meal plan (attempt {gs.attempt}/{gs.max_attempts})...")

    food_lists = [FoodList.model_validate(fl) for fl in state["filtered_food_lists"]]
    user_prompt = build_generation_user_prompt(
        gs.patient,
        food_lists,
        prior_errors=gs.validation_errors if gs.attempt > 1 else [],
        critique_feedback=gs.critique_feedback if gs.attempt > 1 else "",
    )

    llm = _get_llm()
    response = llm.invoke([
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    raw_output = response.content.strip()
    # Strip markdown code fences if present
    if raw_output.startswith("```"):
        lines = raw_output.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_output = "\n".join(lines)

    msg = f"Attempt {gs.attempt}: LLM generation complete ({len(raw_output)} chars)"
    print(f"[📝] {msg}")

    return {
        **state,
        "raw_llm_output": raw_output,
        "attempt": gs.attempt,
        "status": "validating",
        "messages": gs.messages + [msg],
    }


# ── Node: validate_output ────────────────────────────────────────────────────

def validate_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the raw LLM output against schema and nutritional rules."""
    from src.models import GraphState

    gs = GraphState.model_validate(state)
    print(f"[🔍] Validating output...")

    plan, parse_errors = validate_json_schema(gs.raw_llm_output)
    if parse_errors:
        msg = f"Validation FAILED (parse): {parse_errors}"
        print(f"[❌] {msg}")
        return {
            **state,
            "current_plan": None,
            "validation_errors": parse_errors,
            "status": "validation_failed",
            "messages": gs.messages + [msg],
        }

    # ── Deterministic recalculation ──────────────────────────────────────────
    # The LLM's arithmetic is unreliable. Recompute all totals from the actual
    # food-list nutritional metadata so validation is deterministic.
    food_lists = [FoodList.model_validate(fl) for fl in state["filtered_food_lists"]]
    recalculated = _recalculate_plan_totals(plan.model_dump(), food_lists)
    # Re-parse the corrected dict back through the Pydantic schema
    plan, reparse_errors = validate_json_schema(json.dumps(recalculated))
    if reparse_errors:
        # Should never happen since we only changed numbers, but be safe
        msg = f"Re-parse after recalculation failed: {reparse_errors}"
        print(f"[❌] {msg}")
        return {
            **state,
            "current_plan": None,
            "validation_errors": reparse_errors,
            "status": "validation_failed",
            "messages": gs.messages + [msg],
        }
    print(f"[🔢] Totals recalculated from metadata: "
          f"{plan.daily_totals.kcal:.1f} kcal | "
          f"P={plan.daily_totals.protein_g:.1f}g | "
          f"C={plan.daily_totals.carbs_g:.1f}g | "
          f"F={plan.daily_totals.fat_g:.1f}g")
    # ─────────────────────────────────────────────────────────────────────────

    all_issues = run_all_validations(plan, gs.patient)

    # Separate hard errors (blocking) from soft warnings ([WARN] prefix)
    hard_errors = [e for e in all_issues if not e.startswith("[WARN]")]
    warnings    = [e for e in all_issues if e.startswith("[WARN]")]

    if warnings:
        print(f"[⚠️ ] Soft warnings (non-blocking): {warnings}")

    if hard_errors:
        msg = f"Validation FAILED: {hard_errors}"
        print(f"[❌] {msg}")
        
        plan_dict = plan.model_dump()
        if warnings:
            plan_dict["warnings"] = warnings
            
        return {
            **state,
            "current_plan": plan_dict,
            "validation_errors": hard_errors,
            "status": "validation_failed",
            "messages": gs.messages + [msg],
        }

    msg = "Validation PASSED – all checks OK"
    if warnings:
        msg += f" (with {len(warnings)} warning(s))"
    print(f"[✅] {msg}")
    
    plan_dict = plan.model_dump()
    if warnings:
        plan_dict["warnings"] = warnings
        
    return {
        **state,
        "current_plan": plan_dict,
        "validation_errors": warnings,  # pass warnings through for visibility
        "status": "critiquing",
        "messages": gs.messages + [msg],
    }


# ── Node: critique_plan ──────────────────────────────────────────────────────

def critique_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use the LLM to critique the validated plan."""
    from src.models import GraphState

    gs = GraphState.model_validate(state)
    print(f"[🧐] Critiquing meal plan...")

    # current_plan is stored as a plain dict in state (via model_dump()) —
    # never call .model_dump() on it directly, it's already serialisable.
    plan_data = gs.current_plan
    if hasattr(plan_data, "model_dump"):
        plan_json = json.dumps(plan_data.model_dump(), indent=2, ensure_ascii=False)
    else:
        plan_json = json.dumps(plan_data, indent=2, ensure_ascii=False)
    user_prompt = build_critique_user_prompt(plan_json, gs.patient)

    llm = _get_llm()
    response = llm.invoke([
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    raw_critique = response.content.strip()
    # Strip markdown code fences
    if raw_critique.startswith("```"):
        lines = raw_critique.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_critique = "\n".join(lines)

    try:
        critique_data = json.loads(raw_critique)
        approved = critique_data.get("approved", False)
        issues = critique_data.get("issues", [])
    except json.JSONDecodeError:
        approved = False
        issues = [f"Critique response was not valid JSON: {raw_critique[:200]}"]

    if approved:
        msg = "Critique APPROVED – meal plan is valid"
        print(f"[✅] {msg}")
        return {
            **state,
            "critique_feedback": raw_critique,
            "validation_errors": [],
            "status": "success",
            "messages": gs.messages + [msg],
        }
    else:
        msg = f"Critique REJECTED: {issues}"
        print(f"[❌] {msg}")
        return {
            **state,
            "critique_feedback": raw_critique,
            "validation_errors": issues,
            "status": "critique_failed",
            "messages": gs.messages + [msg],
        }


# ── Node: handle_failure ─────────────────────────────────────────────────────

def handle_failure(state: Dict[str, Any]) -> Dict[str, Any]:
    """Terminal node when max retries exhausted."""
    from src.models import GraphState

    gs = GraphState.model_validate(state)
    msg = f"FAILURE: Max attempts ({gs.max_attempts}) reached. Last errors: {gs.validation_errors}"
    print(f"\n[💀] {msg}")
    return {
        **state,
        "status": "failure",
        "messages": gs.messages + [msg],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def should_retry_after_validation(state: Dict[str, Any]) -> str:
    """Route after validation: pass → critique, fail → retry or failure."""
    gs = GraphState.model_validate(state)
    if gs.status == "critiquing":
        return "critique"
    if gs.attempt >= gs.max_attempts:
        return "failure"
    return "retry"


def should_retry_after_critique(state: Dict[str, Any]) -> str:
    """Route after critique: approved → end, rejected → retry or failure."""
    gs = GraphState.model_validate(state)
    if gs.status == "success":
        return "end"
    if gs.attempt >= gs.max_attempts:
        return "failure"
    return "retry"
