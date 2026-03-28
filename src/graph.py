"""LangGraph workflow definition for the Meal Plan Generator."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.nodes import (
    critique_plan,
    generate_plan,
    handle_failure,
    load_and_filter,
    should_retry_after_critique,
    should_retry_after_validation,
    validate_output,
)


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Flow:
        START → load_and_filter → generate_plan → validate_output
            → (pass) → critique_plan → (approved) → END
            → (fail / rejected) → generate_plan (retry) or handle_failure
    """
    workflow = StateGraph(dict)

    # ── Add nodes ─────────────────────────────────────────────────────────
    workflow.add_node("load_and_filter", load_and_filter)
    workflow.add_node("generate_plan", generate_plan)
    workflow.add_node("validate_output", validate_output)
    workflow.add_node("critique_plan", critique_plan)
    workflow.add_node("handle_failure", handle_failure)

    # ── Set entry point ───────────────────────────────────────────────────
    workflow.set_entry_point("load_and_filter")

    # ── Edges ─────────────────────────────────────────────────────────────
    workflow.add_edge("load_and_filter", "generate_plan")
    workflow.add_edge("generate_plan", "validate_output")

    # After validation: pass → critique, fail → retry or give up
    workflow.add_conditional_edges(
        "validate_output",
        should_retry_after_validation,
        {
            "critique": "critique_plan",
            "retry": "generate_plan",
            "failure": "handle_failure",
        },
    )

    # After critique: approved → end, rejected → retry or give up
    workflow.add_conditional_edges(
        "critique_plan",
        should_retry_after_critique,
        {
            "end": END,
            "retry": "generate_plan",
            "failure": "handle_failure",
        },
    )

    workflow.add_edge("handle_failure", END)

    return workflow.compile()
