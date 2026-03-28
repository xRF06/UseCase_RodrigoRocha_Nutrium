"""Pydantic data models for the Meal Plan Generator."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Output Schema Models ─────────────────────────────────────────────────────


class MealTotals(BaseModel):
    """Nutritional totals for a single meal or the whole day."""

    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class Meal(BaseModel):
    """A single meal in the day plan."""

    meal_type: str = Field(
        ...,
        description="One of: Breakfast, Morning Snack, Lunch, Afternoon Snack, Dinner",
    )
    time: str = Field(..., description="Time in HH:MM format")
    items: list[str] = Field(
        ...,
        description=(
            "Food choice groups. Each string contains alternatives separated "
            "by 'OR', with IDs in the format (ID: XXXXX)."
        ),
    )
    meal_totals: MealTotals


class MealPlan(BaseModel):
    """Complete 1-day meal plan – the expected final output."""

    daily_totals: MealTotals
    meals: list[Meal] = Field(..., min_length=2, max_length=6)


# ── Input Data Models ─────────────────────────────────────────────────────────


class Macronutrients(BaseModel):
    fat: float
    carbohydrate: float
    protein: float


class FoodEquivalent(BaseModel):
    """A single food item inside an alternatives list."""

    food_id: int
    description: str


class FoodList(BaseModel):
    """An alternatives list – all items are nutritionally equivalent."""

    name: str
    energy: str
    macronutrients_in_grams: Macronutrients
    fiber_quantity_in_grams: float
    equivalents: list[FoodEquivalent]


class DiseaseInfo(BaseModel):
    items: list[str] = Field(default_factory=list, alias="list")
    details: str = ""

    model_config = {"populate_by_name": True}


class MedicalHistory(BaseModel):
    diseases: DiseaseInfo = Field(default_factory=DiseaseInfo)
    medications: str = ""


class AllergyInfo(BaseModel):
    items: list[str] = Field(default_factory=list, alias="list")
    details: str = ""

    model_config = {"populate_by_name": True}


class IntoleranceInfo(BaseModel):
    items: list[str] = Field(default_factory=list, alias="list")
    details: str = ""

    model_config = {"populate_by_name": True}


class DeficiencyInfo(BaseModel):
    items: list[str] = Field(default_factory=list, alias="list")
    details: str = ""

    model_config = {"populate_by_name": True}


class BowelMovements(BaseModel):
    type: str = "normal"
    details: str = ""


class DietTypes(BaseModel):
    items: list[str] = Field(default_factory=list, alias="list")
    details: str = ""

    model_config = {"populate_by_name": True}


class DietaryHistory(BaseModel):
    wake_up_time_24h: str = ""
    bedtime_24h: str = ""
    diet_types: DietTypes = Field(default_factory=DietTypes)
    favorite_foods: str = ""
    disliked_foods: str = ""
    food_allergies: AllergyInfo = Field(default_factory=AllergyInfo)
    food_intolerances: IntoleranceInfo = Field(default_factory=IntoleranceInfo)
    nutritional_deficiencies: DeficiencyInfo = Field(default_factory=DeficiencyInfo)
    other_eating_histories: str = ""


class FoodDiaryMeal(BaseModel):
    meal_type: str
    text: str


class FoodDiaryEntry(BaseModel):
    date: str
    meals: list[FoodDiaryMeal] = Field(default_factory=list)
    observations: str = ""


class EatingBehaviour(BaseModel):
    date: str = ""
    text: str = ""


class PatientInfos(BaseModel):
    bowel_movements: BowelMovements = Field(default_factory=BowelMovements)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    dietary_history: DietaryHistory = Field(default_factory=DietaryHistory)
    eating_behaviours: list[EatingBehaviour] = Field(default_factory=list)
    food_diary_history_and_obs: list[FoodDiaryEntry] = Field(default_factory=list)


class PatientProfile(BaseModel):
    """A single patient profile from input_nutri_approval.jsonl."""

    patient_name: str
    dee_goal: float
    dee_goal_unit: str = "kcal/24h"
    macronutrient_distribution_in_grams: Macronutrients
    fiber_quantity_in_grams: float
    patient_infos: PatientInfos = Field(default_factory=PatientInfos)


# ── LangGraph State ──────────────────────────────────────────────────────────


class GraphState(BaseModel):
    """Shared state flowing through the LangGraph workflow."""

    patient: PatientProfile
    food_lists: list[FoodList] = Field(default_factory=list)
    filtered_food_lists: list[FoodList] = Field(default_factory=list)
    current_plan: Optional[MealPlan] = None
    raw_llm_output: str = ""
    validation_errors: list[str] = Field(default_factory=list)
    critique_feedback: str = ""
    attempt: int = 0
    max_attempts: int = 3
    status: str = "initialized"  # initialized | generating | validating | critiquing | success | failure
    messages: list[str] = Field(default_factory=list)
