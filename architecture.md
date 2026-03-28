# Architecture – Nutrium Meal Plan Generator

## 1. LangGraph Workflow (Iterative Agent)

```mermaid
flowchart TD
    %% Global Styles
    classDef default fill:#1E1E1E,stroke:#444,stroke-width:1px,color:#FFF,rx:8px,ry:8px;
    classDef startEnd fill:#000000,stroke:#666,stroke-width:2px,color:#FFF,rx:20px,ry:20px;
    classDef ai fill:#2A2D34,stroke:#4A90E2,stroke-width:2px,color:#FFF;
    classDef python fill:#2A2D34,stroke:#E2B94A,stroke-width:2px,color:#FFF;
    classDef io fill:#1E1E1E,stroke:#4CAF50,stroke-width:1.5px,color:#FFF,stroke-dasharray: 5 5;
    classDef error fill:#3A1A1A,stroke:#E53935,stroke-width:2px,color:#FFF;
    classDef condition fill:#2A2A2A,stroke:#9E9E9E,stroke-width:1px,color:#FFF;

    %% Nodes
    START([🚀 START]):::startEnd --> FILTER["🛡️ load_and_filter<br/><small>Safety Firewall</small>"]:::python
    
    FILTER --> GEN["🧠 generate_plan<br/><small>LLM Food Selector</small>"]:::ai
    
    GEN --> VAL["🧮 validate_output<br/><small>Python Math Engine</small>"]:::python
    
    VAL -- "✅ Math & Schema OK" --> CRIT["👨‍⚕️ critique_plan<br/><small>Senior Reviewer</small>"]:::ai
    VAL -- "❌ FAILED" --> RETRY{"🔄 Retries<br/>exhausted?"}:::condition
    
    CRIT -- "✅ Approved" --> SUCC([🎯 SUCCESS]):::startEnd
    CRIT -- "❌ Rejected" --> RETRY
    
    RETRY -- "No (Inject Tip)" --> GEN
    RETRY -- "Yes" --> FAIL_NODE["🗑️ handle_failure"]:::error
    FAIL_NODE --> FAIL_END([💀 FAILURE]):::startEnd

    %% Output Routing
    subgraph Routing ["📁 Output Routing System"]
        SUCC -.->|"Zero Warnings"| J_S[/"/llm_plans_success"/]:::io
        SUCC -.->|"Macro mismatch"| J_W[/"/llm_plans_success_w_warnings"/]:::io
        FAIL_END -.->|"Fatal limits"| J_F[/"/llm_plans_failure"/]:::error
    end
```

## 2. State Schema (Graph Memory)

```mermaid
classDiagram
    class GraphState {
        +PatientProfile patient
        +List~FoodList~ food_lists
        +List~FoodList~ filtered_food_lists
        +MealPlan current_plan
        +str raw_llm_output
        +List~str~ validation_errors
        +str critique_feedback
        +int attempt
        +int max_attempts
        +str status
        +List~str~ messages
    }
```

## 3. Node Operations (Separation of Concerns)

### 🥑 `load_and_filter` (Pre-Flight Firewall)
- **Role**: Drops unwanted items before API usage.
- **Process**: Reads patient `disliked_foods`, `food_allergies`, and maps `food_intolerances` (e.g. "Lactose"). **Physically removes** offending equivalents from the dataset before feeding the prompt, ensuring zero-trust "Safe AI".

### 🧠 `generate_plan` (The Designer)
- **Role**: Orchestrates food pairings and variety.
- **Process**: LLM uses provided choices to draft a full valid JSON. It assigns fractional or integer multipliers `(Multiplier: 1.5)` without doing any math. 
- **Dynamic Prompting**: If retrying from a failure, the prompt receives a `ratio_tip` (e.g., "Change your Multipliers by x1.3!"), forcing the model to iteratively climb/descend scaling thresholds safely.

### 🧮 `validate_output` (The Calculator)
- **Role**: Unforgiving mathematical parser.
- **Process**: Intercepts the generated `(Multiplier: X)`, parses the strings, overrides LLM hallucinated macros, and constructs the absolute nutritional truth from the DB base limits. 
- **Rounding Logic**: A Python custom text-polisher snaps raw Multipliers (e.g., `0.75`) to standard clinical increments (`0.5, 1.0, 1.5`), fixing grammar strings and modifying internal weights deterministically to match the text.
- **Guardrails**: *Calories* falling outside **±10%** represent a `Hard Error` (loops back), while individual Macro discrepancies produce Soft `[WARN]` labels injected into the eventual JSON.

### 👨‍⚕️ `critique_plan` (The Senior Reviewer)
- **Role**: Common sense logic auditor.
- **Process**: Evaluates mathematically perfect plans against human nuance, verifying meal time distributions mapping to wake/sleep habits and flagging bizarre combinations for potential rejection/retry.

### 🗑️ `handle_failure` (Terminal Error State)
- **Role**: The cleanup crew.
- **Process**: Reached if attempts hit exactly `max_attempts(3)`. Forwards the original failing JSON file into `llm_plans_failure/` whilst dynamically appending the ultimate cause of rejection (e.g., Critique's `[WARN]` message) under a new array node within the document for easy debugging.
