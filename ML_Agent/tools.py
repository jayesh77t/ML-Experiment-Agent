"""
tools.py — Pure Python/sklearn computation tools.
The LLM never performs calculations; it only decides *which* tool to call and interprets results.
"""

import hashlib
import json
import uuid
import copy
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS: dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree":       DecisionTreeClassifier(random_state=42),
    "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting":   GradientBoostingClassifier(random_state=42),
}
if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["xgboost"] = XGBClassifier(
        eval_metric="logloss", random_state=42, verbosity=0
    )
if LIGHTGBM_AVAILABLE:
    CLASSIFICATION_MODELS["lightgbm"] = LGBMClassifier(random_state=42, verbose=-1)

REGRESSION_MODELS: dict[str, Any] = {
    "linear_regression": LinearRegression(),
    "ridge":             Ridge(random_state=42),
    "lasso":             Lasso(random_state=42),
    "decision_tree":     DecisionTreeRegressor(random_state=42),
    "random_forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}
if XGBOOST_AVAILABLE:
    REGRESSION_MODELS["xgboost"] = XGBRegressor(random_state=42, verbosity=0)
if LIGHTGBM_AVAILABLE:
    REGRESSION_MODELS["lightgbm"] = LGBMRegressor(random_state=42, verbose=-1)

TUNING_GRIDS: dict[str, dict] = {
    "random_forest":       {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "gradient_boosting":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5, 7]},
    "xgboost":             {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5, 7], "subsample": [0.8, 1.0]},
    "lightgbm":            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63, 127]},
    "logistic_regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
}


# ---------------------------------------------------------------------------
# CORE UTILITY — strip PyArrow / exotic dtypes to plain numpy types
# ---------------------------------------------------------------------------

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every column to plain numpy-backed dtype.
    - Fixes pandas 3.x PyArrow-backed string columns (large_string, etc.)
    - Converts numeric-looking strings (e.g. TotalCharges) to float
    - Keeps object dtype for genuine string columns
    Call this ONCE right after pd.read_csv().
    """
    df = df.copy()
    for col in df.columns:
        # Convert arrow-backed or extension types to plain Python objects first
        try:
            if hasattr(df[col], "dtype") and hasattr(df[col].dtype, "pyarrow_dtype"):
                df[col] = df[col].astype(object)
        except Exception:
            pass

        # Try numeric conversion (handles "1234.5" strings, spaces, etc.)
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only use numeric result if > 80% of non-null values converted cleanly
        if converted.notna().sum() > 0.8 * df[col].notna().sum():
            df[col] = converted
        else:
            # Keep as plain Python str object dtype
            df[col] = df[col].astype(object)

    return df


def _safe_numeric_col(series: pd.Series) -> pd.Series:
    """Ensure a series is float64, coercing anything non-numeric to NaN."""
    try:
        return pd.to_numeric(series, errors="coerce").astype("float64")
    except Exception:
        return series


# ---------------------------------------------------------------------------
# 1. Dataset inspection
# ---------------------------------------------------------------------------

def inspect_dataset(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """Return a structured summary of the dataset for the LLM to interpret."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    missing = df.isnull().sum()
    missing = {k: int(v) for k, v in missing.items() if v > 0}

    potential_ids = [
        c for c in df.columns
        if df[c].nunique() == len(df) or "id" in c.lower()
    ]
    potential_targets = [
        c for c in num_cols + cat_cols
        if df[c].nunique() < 20 and c not in potential_ids
    ]

    return {
        "rows":              int(df.shape[0]),
        "columns":           int(df.shape[1]),
        "numerical_cols":    num_cols,
        "categorical_cols":  cat_cols,
        "missing_values":    missing,
        "duplicate_rows":    int(df.duplicated().sum()),
        "potential_ids":     potential_ids,
        "potential_targets": potential_targets,
        "target_col":        target_col,
        "dataset_hash":      hashlib.md5(
            pd.util.hash_pandas_object(df).values
        ).hexdigest(),
    }


# ---------------------------------------------------------------------------
# 2. Problem type detection
# ---------------------------------------------------------------------------

def detect_problem_type(df: pd.DataFrame, target_col: str) -> dict:
    """Heuristically determine classification vs regression."""
    series   = df[target_col].dropna()
    n_unique = series.nunique()
    dtype    = str(series.dtype)

    if dtype in ("object", "bool", "category") or n_unique <= 2:
        ptype = "binary_classification" if n_unique <= 2 else "multiclass_classification"
    elif n_unique <= 20:
        ptype = "multiclass_classification"
    else:
        ptype = "regression"

    return {
        "target_col":    target_col,
        "dtype":         dtype,
        "unique_values": series.unique().tolist()[:20],
        "n_unique":      int(n_unique),
        "problem_type":  ptype,
    }


# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, target_col: str) -> dict:
    """Compute numerical stats, categorical summaries and target distribution."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- Numerical stats ---
    numerical_stats: dict = {}
    for col in num_cols:
        try:
            s = _safe_numeric_col(df[col]).dropna()
            if len(s) == 0:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr     = q3 - q1
            outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            numerical_stats[col] = {
                "mean":     round(float(s.mean()), 4),
                "median":   round(float(s.median()), 4),
                "std":      round(float(s.std()), 4),
                "min":      round(float(s.min()), 4),
                "max":      round(float(s.max()), 4),
                "skewness": round(float(s.skew()), 4),
                "outliers": outliers,
            }
        except Exception:
            continue

    # --- Categorical stats ---
    categorical_stats: dict = {}
    for col in cat_cols:
        try:
            vc = df[col].astype(str).value_counts()
            categorical_stats[col] = {
                "unique":     int(df[col].nunique()),
                "top_values": vc.head(5).to_dict(),
                "rare_count": int((vc < 10).sum()),
            }
        except Exception:
            continue

    # --- Target distribution ---
    try:
        target_dist = (
            df[target_col].astype(str)
            .value_counts(normalize=True)
            .round(4)
            .to_dict()
        )
    except Exception:
        target_dist = {}

    imbalanced = (
        len(target_dist) == 2
        and min(target_dist.values()) < 0.2
    )

    # --- Correlation with target (encode if string) ---
    try:
        target_series = df[target_col].copy()
        if target_series.dtype == object or str(target_series.dtype) in ("bool", "boolean"):
            target_series = pd.Series(
                LabelEncoder().fit_transform(target_series.astype(str)),
                index=target_series.index,
                name=target_col,
                dtype="float64",
            )
        else:
            target_series = _safe_numeric_col(target_series)

        num_df = df[num_cols].apply(_safe_numeric_col)
        corr_with_target = num_df.corrwith(target_series).round(4).to_dict()
        corr_with_target = {k: (float(v) if pd.notna(v) else None) for k, v in corr_with_target.items()}
    except Exception:
        corr_with_target = {}

    return {
        "numerical_stats":          numerical_stats,
        "categorical_stats":        categorical_stats,
        "target_distribution":      target_dist,
        "class_imbalance_detected": imbalanced,
        "correlation_with_target":  corr_with_target,
    }


# ---------------------------------------------------------------------------
# 4. Data cleaning
# ---------------------------------------------------------------------------

def apply_cleaning_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """
    Execute a cleaning plan produced by the LLM.
    plan: {"col_name": {"action": "median_imputation"|"most_frequent"|"drop"|"fill_unknown"}}
    """
    df = df.copy()
    df = df.drop_duplicates()

    for col, cfg in plan.items():
        if col not in df.columns:
            continue
        action = cfg.get("action", "")
        try:
            if action == "median_imputation":
                df[col] = _safe_numeric_col(df[col]).fillna(_safe_numeric_col(df[col]).median())
            elif action == "mean_imputation":
                df[col] = _safe_numeric_col(df[col]).fillna(_safe_numeric_col(df[col]).mean())
            elif action == "most_frequent":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif action == "fill_unknown":
                df[col] = df[col].fillna("Unknown")
            elif action == "drop":
                df = df.dropna(subset=[col])
        except Exception:
            continue

    return df


# ---------------------------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------------------------

def apply_feature_plan(df: pd.DataFrame, plan: dict, target_col: str) -> pd.DataFrame:
    """
    Execute a feature-engineering plan from the LLM.
    plan keys: encode, interactions, date_cols, drop_cols
    """
    df = df.copy()

    # --- Date decomposition ---
    for col in plan.get("date_cols", []):
        if col not in df.columns:
            continue
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"]       = dt.dt.year.astype("float64")
            df[f"{col}_month"]      = dt.dt.month.astype("float64")
            df[f"{col}_day"]        = dt.dt.day.astype("float64")
            df[f"{col}_days_since"] = (pd.Timestamp.now() - dt).dt.days.astype("float64")
            df.drop(columns=[col], inplace=True)
        except Exception:
            continue

    # --- Interaction features (only between numeric columns) ---
    for pair in plan.get("interactions", []):
        if len(pair) != 3:
            continue
        a, b, kind = pair
        if a not in df.columns or b not in df.columns:
            continue
        try:
            col_a = _safe_numeric_col(df[a])
            col_b = _safe_numeric_col(df[b])
            # Skip if either column is not actually numeric
            if col_a.isna().all() or col_b.isna().all():
                continue
            if kind == "ratio":
                df[f"{a}_per_{b}"] = col_a / col_b.replace(0, np.nan)
            elif kind == "product":
                df[f"{a}_x_{b}"] = col_a * col_b
        except Exception:
            continue

    # --- Encoding ---
    for col, enc_type in plan.get("encode", {}).items():
        if col not in df.columns or col == target_col:
            continue
        try:
            if enc_type == "onehot":
                dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            elif enc_type in ("label", "ordinal"):
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        except Exception:
            continue

    # --- Drop columns ---
    for col in plan.get("drop_cols", []):
        if col in df.columns and col != target_col:
            df.drop(columns=[col], inplace=True)

    # --- Auto-encode any leftover object/category columns (except target) ---
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == target_col:
            continue
        try:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        except Exception:
            df.drop(columns=[col], inplace=True)

    return df


# ---------------------------------------------------------------------------
# 6. Train / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = df.copy()
    X  = df.drop(columns=[target_col])
    y  = df[target_col]

    # --- Encode target ---
    if y.dtype == object or str(y.dtype) in ("bool", "boolean", "string"):
        y = pd.Series(
            LabelEncoder().fit_transform(y.astype(str)),
            index=y.index, name=target_col, dtype="int64",
        )
    else:
        y = _safe_numeric_col(y)

    # --- Encode remaining string/bool columns in X ---
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) in ("bool", "boolean", "string"):
            try:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            except Exception:
                X.drop(columns=[col], inplace=True)

    # --- Force all columns to float64 (catches arrow-backed numerics) ---
    for col in X.columns:
        try:
            X[col] = _safe_numeric_col(X[col])
        except Exception:
            X.drop(columns=[col], inplace=True)

    # --- Drop anything still non-numeric ---
    bad_cols = X.select_dtypes(exclude="number").columns.tolist()
    if bad_cols:
        X.drop(columns=bad_cols, inplace=True)

    # --- Drop columns that are all NaN after conversion ---
    X.dropna(axis=1, how="all", inplace=True)

    # --- Fill remaining NaNs with column median ---
    X.fillna(X.median(numeric_only=True), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "feature_names": X.columns.tolist(),
    }


# ---------------------------------------------------------------------------
# 7. Model training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    split: dict, problem_type: str, model_names: list[str]
) -> dict:
    registry = (
        CLASSIFICATION_MODELS if "classification" in problem_type
        else REGRESSION_MODELS
    )
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    results: dict        = {}
    trained_models: dict = {}

    for name in model_names:
        if name not in registry:
            continue
        try:
            model = copy.deepcopy(registry[name])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_models[name] = model

            if "classification" in problem_type:
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba") else y_pred
                )
                multi = "multiclass" in problem_type
                avg   = "macro" if multi else "binary"
                try:
                    roc = float(roc_auc_score(
                        y_test, y_prob,
                        multi_class="ovr" if multi else "raise",
                    ))
                except Exception:
                    roc = None
                results[name] = {
                    "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
                    "precision":        round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "recall":           round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "f1":               round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "roc_auc":          round(roc, 4) if roc is not None else None,
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                }
            else:
                results[name] = {
                    "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "mse":  round(float(mean_squared_error(y_test, y_pred)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                    "r2":   round(float(r2_score(y_test, y_pred)), 4),
                    "mape": round(float(np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-9)))), 4),
                }
        except Exception as exc:
            results[name] = {"error": str(exc)}

    return {"metrics": results, "trained_models": trained_models}


# ---------------------------------------------------------------------------
# 8. Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_model(
    model,
    X_train,
    y_train,
    model_name: str,
    scoring: str = "f1",
    n_iter: int = 20,
) -> dict:
    param_grid = TUNING_GRIDS.get(model_name, {})
    if not param_grid:
        return {"status": "no_grid_available", "best_model": model, "best_score": None, "best_params": {}}

    try:
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=min(n_iter, 10),
            scoring=scoring,
            cv=3,
            random_state=42,
            n_jobs=-1,
            error_score=0.0,
        )
        search.fit(X_train, y_train)
        return {
            "best_params": search.best_params_,
            "best_score":  round(float(search.best_score_), 4),
            "best_model":  search.best_estimator_,
        }
    except Exception as exc:
        return {"status": f"tuning_failed: {exc}", "best_model": model, "best_score": None, "best_params": {}}


# ---------------------------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names: list[str]) -> list[dict]:
    if model is None:
        return []
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            imp  = np.abs(coef[0] if coef.ndim > 1 else coef)
        else:
            return []
        pairs = sorted(zip(feature_names, imp.tolist()), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 10. Experiment persistence (PostgreSQL optional, JSON fallback)
# ---------------------------------------------------------------------------

def save_experiment(record: dict, db_url: str | None = None) -> str:
    exp_id = f"EXP-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    record["experiment_id"] = exp_id
    record["timestamp"]     = datetime.now().isoformat()

    if db_url:
        try:
            import sqlalchemy as sa
            engine = sa.create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(
                    sa.text(
                        "INSERT INTO experiments (id, data, timestamp) "
                        "VALUES (:id, :data, :ts)"
                    ),
                    {
                        "id":   exp_id,
                        "data": json.dumps(record, default=str),
                        "ts":   record["timestamp"],
                    },
                )
                conn.commit()
        except Exception as exc:
            print(f"[DB] PostgreSQL save failed: {exc}. Using JSON fallback.")

    import pathlib
    log_dir = pathlib.Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{exp_id}.json").write_text(
        json.dumps(record, indent=2, default=str)
    )
    return exp_id


def load_experiment_history(db_url: str | None = None) -> list[dict]:
    if db_url:
        try:
            import sqlalchemy as sa
            engine = sa.create_engine(db_url)
            with engine.connect() as conn:
                rows = conn.execute(
                    sa.text("SELECT data FROM experiments ORDER BY timestamp DESC")
                ).fetchall()
            return [json.loads(r[0]) for r in rows]
        except Exception:
            pass

    import pathlib
    log_dir = pathlib.Path("experiment_logs")
    if not log_dir.exists():
        return []
    records = []
    for f in sorted(log_dir.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(f.read_text()))
        except Exception:
            continue
    return records