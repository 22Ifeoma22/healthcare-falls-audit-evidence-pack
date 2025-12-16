"""
02_shap_explainability.py

Purpose:
- Train a simple baseline model for a falls-risk use case (demo)
- Produce SHAP global + local explanations
- Save audit-ready visuals to ../assets/explainability/

How to run (from repo root):
    python notebooks/02_shap_explainability.py

Dataset:
- Expected (recommended): a CSV you place at: data/falls.csv
- Must include a target column (default): fall_risk (0/1)
- If not found, this script generates a synthetic demo dataset so it still runs.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ASSETS_DIR = REPO_ROOT / "assets" / "explainability"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET_PATH = DATA_DIR / "falls.csv"
DEFAULT_TARGET_COL = "fall_risk"


# -----------------------------
# Synthetic demo dataset (fallback)
# -----------------------------
def make_synthetic_falls(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 100, size=n)
    prior_falls = rng.poisson(lam=0.6, size=n).clip(0, 6)
    meds_count = rng.poisson(lam=4.0, size=n).clip(0, 20)
    mobility_score = rng.normal(loc=50, scale=15, size=n).clip(0, 100)
    cognitive_score = rng.normal(loc=60, scale=18, size=n).clip(0, 100)
    bp_systolic = rng.normal(loc=125, scale=18, size=n).clip(80, 220)
    dizziness = rng.choice(["no", "yes"], size=n, p=[0.75, 0.25])
    ward_type = rng.choice(["medical", "surgical", "rehab"], size=n, p=[0.5, 0.35, 0.15])
    assist_device = rng.choice(["none", "cane", "walker", "wheelchair"], size=n, p=[0.45, 0.2, 0.25, 0.1])

    # Logistic-ish risk score (demo only)
    logits = (
        0.03 * (age - 60)
        + 0.55 * prior_falls
        + 0.08 * meds_count
        - 0.03 * (mobility_score - 50)
        - 0.015 * (cognitive_score - 60)
        + 0.02 * (120 - bp_systolic)  # lower BP -> slightly higher risk (proxy for hypotension)
        + (dizziness == "yes") * 0.6
        + (assist_device != "none") * 0.35
        + (ward_type == "rehab") * 0.25
    )
    probs = 1 / (1 + np.exp(-logits))
    fall_risk = rng.binomial(1, p=np.clip(probs, 0.02, 0.98), size=n)

    df = pd.DataFrame(
        {
            "age": age,
            "prior_falls_12m": prior_falls,
            "meds_count": meds_count,
            "mobility_score": mobility_score.round(1),
            "cognitive_score": cognitive_score.round(1),
            "bp_systolic": bp_systolic.round(0),
            "dizziness": dizziness,
            "ward_type": ward_type,
            "assist_device": assist_device,
            DEFAULT_TARGET_COL: fall_risk,
        }
    )
    return df


# -----------------------------
# Load dataset (CSV or synthetic)
# -----------------------------
def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        return df
    print(f"[INFO] Dataset not found at {path}. Creating synthetic demo dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_falls()
    df.to_csv(path, index=False)
    print(f"[INFO] Wrote synthetic dataset to {path}")
    return df


# -----------------------------
# Build model pipeline
# -----------------------------
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Baseline model that works well with SHAP TreeExplainer
    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    return pipe


# -----------------------------
# Evaluation helpers
# -----------------------------
def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    return metrics


# -----------------------------
# SHAP explainability
# -----------------------------
def run_shap(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    try:
        import shap  # noqa
        import matplotlib.pyplot as plt  # noqa
    except Exception as e:
        print("[ERROR] SHAP or matplotlib not available.")
        print("Install with: pip install shap matplotlib")
        raise e

    # Transform features so SHAP sees the actual model inputs
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)

    # Feature names after preprocessing
    feature_names = pre.get_feature_names_out()
    X_test_t_df = pd.DataFrame(X_test_t, columns=feature_names)

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_t_df)

    # Binary classification: shap_values is list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        sv = shap_values
        expected_value = explainer.expected_value

    # --- Global: summary plot (bar)
    plt.figure()
    shap.summary_plot(sv, X_test_t_df, plot_type="bar", show=False)
    out1 = ASSETS_DIR / "shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out1}")

    # --- Global: summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(sv, X_test_t_df, show=False)
    out2 = ASSETS_DIR / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out2}")

    # --- Local: waterfall for one “high-risk” example
    # pick the highest predicted probability sample
    proba = pipe.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(proba))
    x_row = X_test_t_df.iloc[idx]

    plt.figure()
    # pick positive class if expected_value is an array (common in classifiers)
    ev = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
    sv_row = sv[idx][1] if hasattr(sv[idx], "__len__") else sv[idx]

    shap.plots._waterfall.waterfall_legacy(ev, sv_row, x_row, show=False)

    out3 = ASSETS_DIR / "shap_local_waterfall_highrisk.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out3}")

    # Save a short markdown summary for your evidence pack
    md = ASSETS_DIR / "shap_results_summary.md"
    sv_arr = np.array(sv)

    # If classifier gives (n_features, 2) or (n_rows, n_features, 2), pick positive class (1)
if sv_arr.ndim == 2 and sv_arr.shape[1] == 2:
    sv_for_rank = sv_arr[:, 1]  # (n_features,)
elif sv_arr.ndim == 3 and sv_arr.shape[-1] == 2:
    sv_for_rank = np.abs(sv_arr[:, :, 1]).mean(axis=0)  # (n_features,)
else:
    # already 1D or 2D (n_rows, n_features)
    sv_for_rank = np.abs(sv_arr).mean(axis=0) if sv_arr.ndim > 1 else np.abs(sv_arr)

top_features = (
    pd.Series(np.abs(sv_for_rank), index=feature_names)
      .sort_values(ascending=False)
      .head(10)
)

with md.open("w", encoding="utf-8") as f:
        f.write("# SHAP Results Summary (Demo)\n\n")
        f.write("## Global explanation (top 10 features by mean |SHAP|)\n\n")
        for name, val in top_features.items():
            f.write(f"- **{name}**: {val:.4f}\n")
        f.write("\n## Artefacts generated\n\n")
        f.write("- `assets/explainability/shap_summary_bar.png`\n")
        f.write("- `assets/explainability/shap_summary_beeswarm.png`\n")
        f.write("- `assets/explainability/shap_local_waterfall_highrisk.png`\n")

    print(f"[OK] Saved {md}")


def main() -> None:
    df = load_dataset(DEFAULT_DATASET_PATH)

    if DEFAULT_TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{DEFAULT_TARGET_COL}' not found. "
            f"Either rename your target to '{DEFAULT_TARGET_COL}' or change DEFAULT_TARGET_COL."
        )

    y = df[DEFAULT_TARGET_COL].astype(int)
    X = df.drop(columns=[DEFAULT_TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    metrics = evaluate(pipe, X_test, y_test)
    print("\n=== Baseline evaluation (demo) ===")
    print(f"ROC AUC:       {metrics['roc_auc']:.3f}")
    print(f"Avg Precision: {metrics['avg_precision']:.3f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    # Save metrics for the evidence pack
    metrics_path = ASSETS_DIR / "model_metrics.json"
    import json
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved {metrics_path}")

    # SHAP
    run_shap(pipe, X_train, X_test, y_test)

    print("\nDone ✅")


if __name__ == "__main__":
    main()

