"""
agents.py — LangGraph state machine for autonomous ML experimentation.

Design principle: LLM decides *what* to do; Python (tools.py) *does* the work.
Each node:
  1. Calls a pure Python tool to get real numbers.
  2. Passes the numbers to the LLM for interpretation / planning.
  3. Writes results into ExperimentState.
"""

import json
import os
from typing import Any, Literal

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import tools as T


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("MODEL", "openai/gpt-oss-20b"),
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _ask(system: str, user: str) -> str:
    response = _llm().invoke([
        {"role": "system",  "content": system},
        {"role": "user",    "content": user},
    ])
    return response.content.strip()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class ExperimentState(TypedDict):
    # Input
    dataframe:          Any
    target_column:      str
    problem_type:       str           # "" = auto-detect
    test_size:          float
    random_state:       int
    selected_models:    list[str]
    optimization_metric: str          # "" = auto

    # Computed
    inspection:         dict
    numerical_columns:  list[str]
    categorical_columns: list[str]

    data_quality:       dict
    eda_results:        dict
    cleaning_plan:      dict
    feature_plan:       dict
    feature_results:    dict

    split:              dict          # X_train / X_test etc.
    model_results:      dict
    trained_models:     dict

    best_model_name:    str
    best_metrics:       dict
    tuning_results:     dict

    final_report:       str
    progress_log:       list[str]
    experiment_id:      str


# ---------------------------------------------------------------------------
# Helper to append progress messages
# ---------------------------------------------------------------------------

def _log(state: ExperimentState, msg: str) -> dict:
    return {"progress_log": state.get("progress_log", []) + [msg]}


# ---------------------------------------------------------------------------
# Node 1 — Dataset Inspector
# ---------------------------------------------------------------------------

def inspect_dataset(state: ExperimentState) -> dict:
    df    = state["dataframe"]
    target = state["target_column"]

    inspection = T.inspect_dataset(df, target)

    explanation = _ask(
        "You are a data scientist. Interpret a dataset inspection summary concisely.",
        f"Dataset summary:\n{json.dumps(inspection, indent=2, default=str)}\n\n"
        "Highlight: row count, missing values, potential ID columns, and why the proposed target makes sense.",
    )

    return {
        **_log(state, "✓ Dataset inspection"),
        "inspection":          inspection,
        "numerical_columns":   inspection["numerical_cols"],
        "categorical_columns": inspection["categorical_cols"],
        "data_quality":        {"inspection_notes": explanation},
    }


# ---------------------------------------------------------------------------
# Node 2 — Problem Detector
# ---------------------------------------------------------------------------

def detect_problem(state: ExperimentState) -> dict:
    df     = state["dataframe"]
    target = state["target_column"]
    override = state.get("problem_type", "")

    detection = T.detect_problem_type(df, target)
    ptype = override if override else detection["problem_type"]

    explanation = _ask(
        "You are an ML engineer. Confirm problem type from detection data.",
        f"Detection result:\n{json.dumps(detection, indent=2)}\nChosen type: {ptype}\n\n"
        "Briefly explain why this problem type is correct.",
    )

    return {
        **_log(state, "✓ Problem type detected"),
        "problem_type": ptype,
        "data_quality": {**state.get("data_quality", {}), "problem_detection": explanation},
    }


# ---------------------------------------------------------------------------
# Node 3 — EDA Agent
# ---------------------------------------------------------------------------

def run_eda(state: ExperimentState) -> dict:
    df     = state["dataframe"]
    target = state["target_column"]

    eda = T.run_eda(df, target)

    insights = _ask(
        "You are a data analyst. Summarise EDA findings and flag issues.",
        f"EDA results:\n{json.dumps(eda, indent=2, default=str)}\n\n"
        "Note class imbalance, high-skew features, strong correlations, and rare categories.",
    )

    return {
        **_log(state, "✓ EDA complete"),
        "eda_results": {**eda, "llm_insights": insights},
    }


# ---------------------------------------------------------------------------
# Node 4 — Data Cleaning
# ---------------------------------------------------------------------------

def clean_data(state: ExperimentState) -> dict:
    df         = state["dataframe"]
    inspection = state["inspection"]

    # Ask LLM to produce a JSON cleaning plan
    raw_plan = _ask(
        "You are an ML engineer. Return ONLY valid JSON — no markdown, no preamble.",
        f"Missing values:\n{json.dumps(inspection.get('missing_values', {}), indent=2)}\n"
        f"Numerical columns: {inspection['numerical_cols']}\n"
        f"Categorical columns: {inspection['categorical_cols']}\n\n"
        'Return a JSON object like: {"col_name": {"action": "median_imputation"}}. '
        "Valid actions: median_imputation, mean_imputation, most_frequent, fill_unknown, drop.",
    )

    cleaned_raw = raw_plan.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        plan = json.loads(cleaned_raw)
        if not isinstance(plan, dict):
            raise ValueError
    except Exception:
        plan = {}

    cleaned_df = T.apply_cleaning_plan(df, plan)

    return {
        **_log(state, "✓ Data cleaned"),
        "cleaning_plan":  plan,
        "dataframe":      cleaned_df,
    }


# ---------------------------------------------------------------------------
# Node 5 — Feature Engineer
# ---------------------------------------------------------------------------

def engineer_features(state: ExperimentState) -> dict:
    df     = state["dataframe"]
    target = state["target_column"]
    eda    = state["eda_results"]

    raw_plan = _ask(
        "You are a feature engineer. Return ONLY valid JSON — no markdown, no preamble.",
        f"Columns: {df.columns.tolist()}\n"
        f"Target: {target}\n"
        f"Numerical stats: {json.dumps(list(eda.get('numerical_stats', {}).keys()))}\n"
        f"Categorical stats: {json.dumps(list(eda.get('categorical_stats', {}).keys()))}\n\n"
        "Return a JSON feature plan with keys:\n"
        '  "encode": {"col": "onehot"|"label"},\n'
        '  "interactions": [["col_a", "col_b", "ratio"]],\n'
        '  "date_cols": [],\n'
        '  "drop_cols": []',
    )

    # Strip markdown fences the LLM sometimes adds despite instructions
    cleaned = raw_plan.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        plan = json.loads(cleaned)
        # Validate expected keys exist
        if not isinstance(plan, dict):
            raise ValueError("plan is not a dict")
    except Exception:
        plan = {"encode": {}, "interactions": [], "date_cols": [], "drop_cols": []}

    # Safety: interactions must reference only numeric columns
    num_cols_set = set(df.select_dtypes(include="number").columns)
    safe_interactions = [
        pair for pair in plan.get("interactions", [])
        if len(pair) == 3 and pair[0] in num_cols_set and pair[1] in num_cols_set
    ]
    plan["interactions"] = safe_interactions

    engineered_df = T.apply_feature_plan(df, plan, target)

    return {
        **_log(state, "✓ Feature engineering done"),
        "feature_plan":    plan,
        "feature_results": {"new_shape": engineered_df.shape, "columns": engineered_df.columns.tolist()},
        "dataframe":       engineered_df,
    }


# ---------------------------------------------------------------------------
# Node 6 — Model Trainer
# ---------------------------------------------------------------------------

def train_models(state: ExperimentState) -> dict:
    df      = state["dataframe"]
    target  = state["target_column"]
    ptype   = state["problem_type"]
    models  = state.get("selected_models") or list(
        T.CLASSIFICATION_MODELS if "classification" in ptype else T.REGRESSION_MODELS
    )[:4]

    split = T.split_data(df, target, state.get("test_size", 0.2), state.get("random_state", 42))
    eval_result = T.train_and_evaluate(split, ptype, models)

    return {
        **_log(state, "✓ Models trained"),
        "split":          split,
        "model_results":  eval_result["metrics"],
        "trained_models": eval_result["trained_models"],
    }


# ---------------------------------------------------------------------------
# Node 7 — Model Evaluator
# ---------------------------------------------------------------------------

def evaluate_models(state: ExperimentState) -> dict:
    metrics  = state["model_results"]
    ptype    = state["problem_type"]
    opt      = state.get("optimization_metric", "")

    # Determine primary metric
    if not opt:
        opt = "f1" if "classification" in ptype else "rmse"

    # Find best model
    if "classification" in ptype:
        best_name = max(metrics, key=lambda m: metrics[m].get(opt, metrics[m].get("f1", 0)))
    else:
        best_name = min(metrics, key=lambda m: metrics[m].get(opt, metrics[m].get("rmse", float("inf"))))

    best_metrics = metrics[best_name]

    analysis = _ask(
        "You are an ML evaluator. Analyse model results and justify the best model choice.",
        f"Problem: {ptype}\nOptimisation metric: {opt}\n"
        f"Results:\n{json.dumps(metrics, indent=2)}\n"
        f"Best model: {best_name}\n\n"
        "Give a concise 3–5 sentence evaluation.",
    )

    return {
        **_log(state, "✓ Models evaluated"),
        "best_model_name": best_name,
        "best_metrics":    best_metrics,
        "data_quality":    {**state.get("data_quality", {}), "evaluation_analysis": analysis},
    }


# ---------------------------------------------------------------------------
# Node 8 — Hyperparameter Tuner
# ---------------------------------------------------------------------------

def tune_model(state: ExperimentState) -> dict:
    best_name     = state["best_model_name"]
    trained       = state["trained_models"]
    split         = state["split"]
    ptype         = state["problem_type"]
    opt           = state.get("optimization_metric", "f1" if "classification" in ptype else "r2")

    if best_name not in trained:
        return {**_log(state, "⚠ Tuning skipped (model not found)"), "tuning_results": {}}

    tuning = T.tune_model(
        model=trained[best_name],
        X_train=split["X_train"],
        y_train=split["y_train"],
        model_name=best_name,
        scoring=opt,
    )

    summary = _ask(
        "You are an ML engineer. Interpret hyperparameter tuning results.",
        f"Model: {best_name}\n"
        f"Best params: {json.dumps(tuning.get('best_params', {}))}\n"
        f"Tuned score ({opt}): {tuning.get('best_score')}\n"
        f"Previous best score: {state['best_metrics'].get(opt, 'N/A')}\n\n"
        "Explain what improved and recommend whether the tuned model should be used.",
    )

    return {
        **_log(state, "✓ Hyperparameter tuning done"),
        "tuning_results": {**tuning, "summary": summary},
    }


# ---------------------------------------------------------------------------
# Conditional edge — should we tune?
# ---------------------------------------------------------------------------

def should_tune(state: ExperimentState) -> Literal["tune", "generate_report"]:
    ptype = state["problem_type"]
    opt   = state.get("optimization_metric", "f1" if "classification" in ptype else "r2")
    score = state["best_metrics"].get(opt, 0)

    threshold = 0.80 if "classification" in ptype else 0.75
    return "tune" if score < threshold else "generate_report"


# ---------------------------------------------------------------------------
# Node 9 — Report Generator
# ---------------------------------------------------------------------------

def generate_report(state: ExperimentState) -> dict:
    inspection = state["inspection"]
    eda        = state["eda_results"]
    ptype      = state["problem_type"]
    best_name  = state["best_model_name"]
    best_m     = state["best_metrics"]
    tuning     = state.get("tuning_results", {})
    feat_imp   = T.get_feature_importance(
        state["trained_models"].get(best_name),
        state["split"].get("feature_names", []),
    )

    report = _ask(
        "You are a senior data scientist writing a professional ML experiment report.",
        f"""
Dataset: {inspection['rows']} rows, {inspection['columns']} columns
Problem: {ptype}
Target: {state['target_column']}

EDA insights: {eda.get('llm_insights', '')}
Class imbalance: {eda.get('class_imbalance_detected', False)}

Best model: {best_name}
Metrics: {json.dumps(best_m)}
Tuning improvement: {tuning.get('best_score', 'N/A')}

Top features: {json.dumps(feat_imp[:5])}

Write a structured report with sections:
## Dataset, ## Problem, ## Data Quality, ## Best Model, ## Performance,
## Important Features, ## Model Selection Rationale, ## Recommendations,
## Limitations, ## Next Experiments
""",
    )

    # Persist experiment
    record = {
        "dataset_name":   "uploaded_dataset",
        "dataset_hash":   inspection.get("dataset_hash"),
        "target":         state["target_column"],
        "problem_type":   ptype,
        "features":       state["split"].get("feature_names", []),
        "models_trained": list(state["model_results"].keys()),
        "best_model":     best_name,
        "best_metrics":   best_m,
        "tuning":         {k: v for k, v in tuning.items() if k != "best_model"},
        "feature_importance": feat_imp,
    }
    exp_id = T.save_experiment(record, db_url=os.getenv("DATABASE_URL"))

    return {
        **_log(state, "✓ Report generated"),
        "final_report":   report,
        "experiment_id":  exp_id,
    }


# ---------------------------------------------------------------------------
# Build the LangGraph state machine
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ExperimentState)

    graph.add_node("inspect_dataset",    inspect_dataset)
    graph.add_node("detect_problem",     detect_problem)
    graph.add_node("eda",                run_eda)
    graph.add_node("clean_data",         clean_data)
    graph.add_node("feature_engineering", engineer_features)
    graph.add_node("train_models",       train_models)
    graph.add_node("evaluate",           evaluate_models)
    graph.add_node("tune",               tune_model)
    graph.add_node("generate_report",    generate_report)

    graph.add_edge(START,                "inspect_dataset")
    graph.add_edge("inspect_dataset",    "detect_problem")
    graph.add_edge("detect_problem",     "eda")
    graph.add_edge("eda",                "clean_data")
    graph.add_edge("clean_data",         "feature_engineering")
    graph.add_edge("feature_engineering","train_models")
    graph.add_edge("train_models",       "evaluate")

    # Conditional routing: tune if performance is below threshold
    graph.add_conditional_edges("evaluate", should_tune, {
        "tune":            "tune",
        "generate_report": "generate_report",
    })
    graph.add_edge("tune",          "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# Singleton compiled graph
experiment_graph = build_graph()