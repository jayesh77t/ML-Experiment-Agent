"""
app.py — Streamlit UI for the ML Experiment Agent.

Run:
    streamlit run app.py

Environment variables (in .env or shell):
    GROQ_API_KEY=...
    MODEL=openai/gpt-oss-20b       # default
    DATABASE_URL=postgresql://...  # optional; falls back to local JSON
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import experiment_graph, ExperimentState
import tools as T

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🧪 ML Experiment Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧪 ML Experiment Agent")
st.caption("Autonomous Machine Learning Experimentation")

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    target_col = st.text_input("🎯 Target Column", placeholder="e.g. Exited")

    st.header("⚙️ Experiment")
    problem_override = st.selectbox(
        "Problem Type",
        ["Auto Detect", "binary_classification", "multiclass_classification", "regression"],
    )
    test_size    = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random State", value=42, step=1)

    st.header("🎯 Optimisation Metric")
    metric = st.selectbox(
        "Primary Metric",
        ["Auto", "accuracy", "f1", "recall", "precision", "roc_auc", "rmse", "r2", "mae"],
    )

    st.header("🤖 Models")
    all_clf = list(T.CLASSIFICATION_MODELS.keys())
    all_reg = list(T.REGRESSION_MODELS.keys())

    use_clf = "classification" in problem_override or problem_override == "Auto Detect"
    model_options = all_clf if use_clf else all_reg
    selected_models = st.multiselect("Select models", model_options, default=model_options[:3])

    run_btn = st.button("🚀 Run Experiment", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "📊 Overview", "🔎 EDA", "🧹 Data Quality",
    "🧠 Features", "🤖 Models", "📈 Evaluation",
    "⚙️ Tuning", "🏆 Best Model", "📝 AI Report", "🧪 History",
])

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

if run_btn:
    if uploaded is None:
        st.error("Please upload a CSV file.")
        st.stop()
    if not target_col:
        st.error("Please enter the target column name.")
        st.stop()

    df = pd.read_csv(uploaded)
    df = T.sanitize_dataframe(df)   # strip PyArrow-backed dtypes → plain numpy
    if target_col not in df.columns:
        st.error(f"Column '{target_col}' not found. Available: {df.columns.tolist()}")
        st.stop()

    initial_state: ExperimentState = {
        "dataframe":           df,
        "target_column":       target_col,
        "problem_type":        "" if problem_override == "Auto Detect" else problem_override,
        "test_size":           test_size,
        "random_state":        int(random_state),
        "selected_models":     selected_models,
        "optimization_metric": "" if metric == "Auto" else metric,
        # will be filled by nodes
        "inspection": {}, "numerical_columns": [], "categorical_columns": [],
        "data_quality": {}, "eda_results": {}, "cleaning_plan": {}, "feature_plan": {},
        "feature_results": {}, "split": {}, "model_results": {}, "trained_models": {},
        "best_model_name": "", "best_metrics": {}, "tuning_results": {},
        "final_report": "", "progress_log": [], "experiment_id": "",
    }

    progress_placeholder = st.empty()

    with st.spinner("Running experiment…"):
        final_state = experiment_graph.invoke(
            initial_state,
            config={"recursion_limit": 50},
        )

    progress_placeholder.empty()
    st.session_state["state"] = final_state
    st.success(f"✅ Experiment complete — ID: {final_state.get('experiment_id', 'N/A')}")


# ---------------------------------------------------------------------------
# Display results (only if an experiment has been run)
# ---------------------------------------------------------------------------

state: ExperimentState | None = st.session_state.get("state")

# ---------- Tab 0: Overview -------------------------------------------------
with tabs[0]:
    if state:
        insp = state["inspection"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows",    f"{insp.get('rows', 0):,}")
        col2.metric("Columns", insp.get("columns", 0))
        col3.metric("Missing", sum(insp.get("missing_values", {}).values()))
        col4.metric("Dupes",   insp.get("duplicate_rows", 0))

        st.subheader("Progress")
        for msg in state.get("progress_log", []):
            st.write(msg)

        st.subheader("Detected Problem Type")
        st.info(state.get("problem_type", "—"))
    else:
        st.info("Upload a dataset and click **Run Experiment** to begin.")

# ---------- Tab 1: EDA ------------------------------------------------------
with tabs[1]:
    if state and state.get("eda_results"):
        eda = state["eda_results"]
        st.subheader("Target Distribution")
        td = eda.get("target_distribution", {})
        if td:
            fig = px.bar(x=list(td.keys()), y=list(td.values()), labels={"x": "Class", "y": "Proportion"})
            st.plotly_chart(fig, use_container_width=True)
        if eda.get("class_imbalance_detected"):
            st.warning("⚠️ Significant class imbalance detected.")

        st.subheader("Numerical Features")
        ns = eda.get("numerical_stats", {})
        if ns:
            st.dataframe(pd.DataFrame(ns).T, use_container_width=True)

        st.subheader("LLM Insights")
        st.write(eda.get("llm_insights", "—"))
    else:
        st.info("Run an experiment to see EDA results.")

# ---------- Tab 2: Data Quality ---------------------------------------------
with tabs[2]:
    if state and state.get("cleaning_plan"):
        st.subheader("Cleaning Plan")
        st.json(state["cleaning_plan"])
        st.subheader("Inspector Notes")
        st.write(state["data_quality"].get("inspection_notes", "—"))
    else:
        st.info("Run an experiment to see data quality results.")

# ---------- Tab 3: Features -------------------------------------------------
with tabs[3]:
    if state and state.get("feature_plan"):
        st.subheader("Feature Engineering Plan")
        st.json(state["feature_plan"])
        fr = state.get("feature_results", {})
        if fr:
            st.write(f"Shape after engineering: {fr.get('new_shape')}")
    else:
        st.info("Run an experiment to see feature engineering results.")

# ---------- Tab 4: Models ---------------------------------------------------
with tabs[4]:
    if state and state.get("model_results"):
        st.subheader("Model Comparison")

        # Filter out failed models and non-numeric columns (confusion_matrix, error, roc_auc=None)
        clean_results = {}
        for model_name, metrics in state["model_results"].items():
            if "error" in metrics:
                st.warning(f"⚠️ {model_name} failed: {metrics['error']}")
                continue
            clean_results[model_name] = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None
            }

        if clean_results:
            df_metrics = pd.DataFrame(clean_results).T
            # Drop columns that still have any NaN (e.g. roc_auc for multiclass)
            df_metrics = df_metrics.dropna(axis=1, how="any")
            df_metrics = df_metrics.astype(float)

            try:
                styled = df_metrics.style.highlight_max(axis=0, color="#d4edda")
                st.dataframe(styled, use_container_width=True)
            except Exception:
                # Fallback: plain dataframe with no styling
                st.dataframe(df_metrics, use_container_width=True)
        else:
            st.warning("No models completed successfully.")
    else:
        st.info("Run an experiment to see model results.")

# ---------- Tab 5: Evaluation -----------------------------------------------
with tabs[5]:
    if state and state.get("best_model_name"):
        st.subheader(f"🏅 Best Model: **{state['best_model_name']}**")

        # Show only scalar metrics in a clean table
        bm = state["best_metrics"]
        scalar_metrics = {
            k: round(float(v), 4)
            for k, v in bm.items()
            if isinstance(v, (int, float)) and v is not None
        }
        if scalar_metrics:
            st.dataframe(
                pd.DataFrame(scalar_metrics, index=["Score"]),
                use_container_width=True,
            )

        st.write(state["data_quality"].get("evaluation_analysis", "—"))

        # Confusion matrix (classification)
        cm_data = bm.get("confusion_matrix")
        if cm_data and isinstance(cm_data, list):
            try:
                fig = px.imshow(
                    cm_data, text_auto=True,
                    labels={"x": "Predicted", "y": "Actual"},
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render confusion matrix: {e}")
    else:
        st.info("Run an experiment to see evaluation results.")

# ---------- Tab 6: Tuning ---------------------------------------------------
with tabs[6]:
    if state and state.get("tuning_results"):
        tr = state["tuning_results"]
        st.subheader("Best Hyperparameters")
        st.json(tr.get("best_params", {}))
        st.metric("Tuned Score", tr.get("best_score", "—"))
        st.write(tr.get("summary", "—"))
    else:
        st.info("Run an experiment to see tuning results.")

# ---------- Tab 7: Best Model -----------------------------------------------
with tabs[7]:
    if state and state.get("best_model_name"):
        best = state["best_model_name"]
        st.subheader(f"🏆 {best}")

        # Safe scalar metrics only
        bm = state.get("best_metrics", {})
        scalar_metrics = {
            k: round(float(v), 4)
            for k, v in bm.items()
            if isinstance(v, (int, float)) and v is not None
        }
        if scalar_metrics:
            cols = st.columns(len(scalar_metrics))
            for col, (k, v) in zip(cols, scalar_metrics.items()):
                col.metric(k.upper(), v)

        # Feature importance chart
        try:
            model_obj  = state.get("trained_models", {}).get(best)
            feat_names = state.get("split", {}).get("feature_names", [])
            if model_obj and feat_names:
                imp = T.get_feature_importance(model_obj, feat_names)
                if imp:
                    fig = px.bar(
                        pd.DataFrame(imp[:15]),
                        x="importance", y="feature",
                        orientation="h",
                        title="Top 15 Feature Importances",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")
    else:
        st.info("Run an experiment to see the best model.")

# ---------- Tab 8: AI Report ------------------------------------------------
with tabs[8]:
    if state and state.get("final_report"):
        st.markdown(state["final_report"])
        st.download_button(
            "⬇️ Download Report",
            data=state["final_report"],
            file_name=f"{state.get('experiment_id', 'report')}.md",
            mime="text/markdown",
        )
    else:
        st.info("Run an experiment to generate the AI report.")

# ---------- Tab 9: History --------------------------------------------------
with tabs[9]:
    st.subheader("Experiment History")
    history = T.load_experiment_history(db_url=os.getenv("DATABASE_URL"))
    if history:
        rows = []
        for h in history:
            rows.append({
                "ID":        h.get("experiment_id", "—"),
                "Dataset":   h.get("dataset_name", "—"),
                "Target":    h.get("target", "—"),
                "Problem":   h.get("problem_type", "—"),
                "Best Model": h.get("best_model", "—"),
                "Timestamp": h.get("timestamp", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No experiments recorded yet.")