#!/usr/bin/env python3
"""
Meal Plan Generator – Main Entry Point
=======================================
Runs the LangGraph workflow for one or more patient profiles and prints
the progression of the graph along with the final meal plan.

Usage:
    python main.py                      # Run for all patients
    python main.py --patient "Paciente 1"  # Run for a specific patient
"""

from __future__ import annotations

import argparse
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env

from src.data_loader import load_food_lists, load_patient_profiles
from src.graph import build_graph
from src.models import FoodList


def run_for_patient(patient_name: str | None = None) -> None:
    """Run the meal plan generation pipeline."""

    # ── Load data ─────────────────────────────────────────────────────────
    patients = load_patient_profiles()
    food_lists = load_food_lists()

    if patient_name:
        patients = [p for p in patients if p.patient_name == patient_name]
        if not patients:
            print(f"❌ Patient '{patient_name}' not found.")
            sys.exit(1)

    graph = build_graph()

    for patient in patients:
        print("\n" + "=" * 70)
        print(f"  🏥 MEAL PLAN GENERATION — {patient.patient_name}")
        print(f"  Goal: {patient.dee_goal} {patient.dee_goal_unit}")
        print(
            f"  Macros: P={patient.macronutrient_distribution_in_grams.protein}g | "
            f"C={patient.macronutrient_distribution_in_grams.carbohydrate}g | "
            f"F={patient.macronutrient_distribution_in_grams.fat}g"
        )
        print(f"  Fiber: {patient.fiber_quantity_in_grams}g")
        print("=" * 70)

        initial_state = {
            "patient": patient.model_dump(),
            "food_lists": [fl.model_dump() for fl in food_lists],
            "filtered_food_lists": [],
            "current_plan": None,
            "raw_llm_output": "",
            "validation_errors": [],
            "critique_feedback": "",
            "attempt": 0,
            "max_attempts": 3,
            "status": "initialized",
            "messages": [],
        }

        # ── Run the graph ─────────────────────────────────────────────────
        final_state = graph.invoke(initial_state)

        # ── Print results ─────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("📊 GRAPH PROGRESSION:")
        for i, msg in enumerate(final_state.get("messages", []), 1):
            print(f"  {i}. {msg}")

        status = final_state.get("status", "unknown")
        print(f"\n🏁 Final Status: {status.upper()}")

        import os
        
        safe_name = patient.patient_name.replace(" ", "_").replace("/", "_")
        current_plan = final_state.get("current_plan")
        
        # Inject final fatal errors / critique feedback into the JSON's warnings array if it failed
        if status == "failure" and current_plan:
            final_errs = final_state.get("validation_errors", [])
            if final_errs:
                if "warnings" not in current_plan:
                    current_plan["warnings"] = []
                for err in final_errs:
                    if err not in current_plan["warnings"]:
                        current_plan["warnings"].append(err)
                        
        # Routing logic for folders
        if status == "success":
            if current_plan and current_plan.get("warnings"):
                folder = "llm_plans_success_w_warnings"
            else:
                folder = "llm_plans_success"
        else:
            folder = "llm_plans_failure"
            
        os.makedirs(folder, exist_ok=True)
        plan_filename = f"{folder}/{safe_name}.json"
        
        if status == "success" and current_plan:
            with open(plan_filename, "w", encoding="utf-8") as f:
                json.dump(current_plan, f, indent=2, ensure_ascii=False)
            print(f"\n📁 Plan successfully saved for analysis to: {plan_filename}")
            
            print("\n📋 GENERATED MEAL PLAN:")
            print(json.dumps(current_plan, indent=2, ensure_ascii=False))
            
        elif status == "failure":
            # Save the last failed output for debugging
            with open(plan_filename, "w", encoding="utf-8") as f:
                if current_plan:
                    json.dump(current_plan, f, indent=2, ensure_ascii=False)
                else:
                    f.write(final_state.get("raw_llm_output", ""))
            print(f"\n📁 Failed plan saved for debugging to: {plan_filename}")
            
            print("\n❌ Generation failed after all retry attempts.")
            print(f"   Last errors: {final_state.get('validation_errors', [])}")

        print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI Meal Plan Generator – Nutrium Challenge"
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help='Patient name to generate for (e.g. "Paciente 1"). Omit for all.',
    )
    args = parser.parse_args()
    run_for_patient(args.patient)


if __name__ == "__main__":
    main()
