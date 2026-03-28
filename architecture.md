# Architecture – Nutrium Meal Plan Generator

## 1. LangGraph Workflow (Iterative Agent)

```mermaid
graph TD
    %% Define styles
    classDef startEnd fill:#4CAF50,color:#fff,stroke:#388E3C,stroke-width:2px;
    classDef failure fill:#f44336,color:#fff,stroke:#d32f2f,stroke-width:2px;
    classDef action fill:#2196F3,color:#fff,stroke:#1976D2,stroke-width:2px;
    classDef api fill:#FF9800,color:#fff,stroke:#F57C00,stroke-width:2px;
    classDef logic fill:#9C27B0,color:#fff,stroke:#7B1FA2,stroke-width:2px;
    classDef review fill:#009688,color:#fff,stroke:#00796B,stroke-width:2px;
    classDef io fill:#607D8B,color:#fff,stroke:#455A64,stroke-width:2px;

    %% Nodes
    A([START]):::startEnd --> B["load_and_filter (Safety Firewall)"]:::action
    B --> C["generate_plan (LLM Food Selector)"]:::api
    C --> D["validate_output (Python Math Engine)"]:::logic
    
    D -->|"Math and Schema OK"| E["critique_plan (Senior Reviewer)"]:::review
    D -->|"Math or Schema FAILED"| F{"Retries limit reached?"}
    
    E -->|"Approved"| G([SUCCESS]):::startEnd
    E -->|"Rejected"| F
    
    F -->|"No (Dynamic Tip Injected)"| C
    F -->|"Yes (Attempts Exhausted)"| H["handle_failure"]:::failure
    H --> I([FAILURE]):::failure

    %% Output Routing
    G -.-> |"Zero Warnings"| J_S["llm_plans_success (Perfect)"]:::io
    G -.-> |"Macro mismatch"| J_W["llm_plans_success_w_warnings (Soft Warnings)"]:::io
    I -.-> |"Fatal limits"| J_F["llm_plans_failure (Full Error Logs)"]:::io
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
