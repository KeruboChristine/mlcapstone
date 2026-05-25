from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Heart Disease Prediction - Capstone Learning App",
    page_icon="H",
    layout="wide",
)


DATA_PATH = Path(__file__).resolve().parent / "raw_merged_heart_dataset.csv"
RANDOM_STATE = 42
TARGET_COL = "target"

TO_NUMERIC_COLS = [
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalachh",
    "exang",
    "slope",
    "ca",
    "thal",
]

OUTLIER_COLS = ["chol", "trestbps", "oldpeak"]

NUM_IMPUTE_COLS = [
    "trestbps",
    "chol",
    "thalachh",
    "oldpeak",
    "ca",
    "stress_risk",
    "risk_score",
]

CAT_IMPUTE_COLS = ["fbs", "exang"]

ENCODE_COLS = ["cp", "thal", "slope", "restecg"]

NOTEBOOK_RESULTS = pd.DataFrame(
    [
        ["Logistic Regression", 0.790000, 0.810000, 0.760000, 0.780000, 0.875000],
        ["Tuned Logistic Regression", 0.783000, 0.799000, 0.751000, 0.774000, 0.875000],
        ["Random Forest", 0.931000, 0.935000, 0.926000, 0.931000, 0.986000],
        ["Tuned Random Forest", 0.936000, 0.940000, 0.931000, 0.935000, 0.986000],
        ["XGBoost (Baseline)", 0.926773, 0.930233, 0.921659, 0.925926, 0.977901],
        ["XGBoost (Tuned)", 0.929062, 0.930556, 0.926267, 0.928406, 0.975597],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
)

PROBLEM_STATEMENT = (
    "Given a patient's clinical measurements, predict whether they are likely to have "
    "heart disease (1) or not (0), while minimizing missed disease cases."
)

TARGET_AUDIENCE = [
    "Clinical teams who need early screening support.",
    "Hospital operations teams tracking risk trends.",
    "Data/IT interns learning end-to-end medical ML workflows.",
    "Healthcare managers comparing model safety trade-offs.",
]

BASE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalachh",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

ORIGINAL_COLUMN_MEANINGS = [
    ("age", "Age in years"),
    ("sex", "Biological sex (1=male, 0=female)"),
    ("cp", "Chest pain type category"),
    ("trestbps", "Resting blood pressure"),
    ("chol", "Serum cholesterol level"),
    ("fbs", "Fasting blood sugar flag"),
    ("restecg", "Resting ECG category"),
    ("thalachh", "Maximum heart rate achieved"),
    ("exang", "Exercise-induced angina flag"),
    ("oldpeak", "ST depression after exercise"),
    ("slope", "Slope of exercise ST segment"),
    ("ca", "Number of major vessels"),
    ("thal", "Thalassemia stress test category"),
    ("target", "Outcome (1=disease, 0=no disease)"),
]

ENGINEERED_COLUMN_MEANINGS = [
    ("high_bp", "High blood pressure risk flag"),
    ("high_chol", "High cholesterol risk flag"),
    ("stress_risk", "Exercise stress burden (oldpeak * exang)"),
    ("low_hr", "Low heart rate risk flag"),
    ("vessel_risk", "Any vessel involvement risk flag"),
    ("risk_score", "Combined cardiovascular burden score"),
]

ENCODED_COLUMN_MEANINGS = [
    ("cp_*", "One-hot encoded chest pain categories"),
    ("thal_*", "One-hot encoded thalassemia categories"),
    ("slope_*", "One-hot encoded ST slope categories"),
    ("restecg_*", "One-hot encoded ECG categories"),
]


def apply_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Source+Sans+3:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] * {
    font-family: "Source Sans 3", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% 10%, rgba(147, 197, 253, 0.26), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(251, 191, 36, 0.18), transparent 30%),
        linear-gradient(180deg, #f7fbff 0%, #fefefe 48%, #f7f8ff 100%);
}

.hero {
    border-radius: 18px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
    background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 48%, #2563eb 100%);
    color: #ffffff;
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.18);
}

.hero h1 {
    font-family: "Manrope", sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0 0 0.2rem 0;
}

.hero p {
    margin: 0.12rem 0;
    font-size: 1rem;
}

div[data-testid="metric-container"] {
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(59, 130, 246, 0.22);
    box-shadow: 0 8px 18px rgba(30, 64, 175, 0.08);
    padding: 0.7rem 0.9rem;
}

div[data-testid="stExpander"] details {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.28);
}

.tag {
    display: inline-block;
    background: #e2e8f0;
    color: #1e293b;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.82rem;
    margin-right: 0.25rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TO_NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def compute_outlier_bounds(
    df: pd.DataFrame, cols: list[str]
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds[col] = (lower, upper)
    return bounds


def count_outliers(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col, (lower, upper) in bounds.items():
        counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return counts


def apply_outlier_capping(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lower, upper) in bounds.items():
        out[col] = out[col].clip(lower, upper)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["high_bp"] = np.where(out["trestbps"] > 140, 1, 0)
    out["high_chol"] = np.where(out["chol"] > 240, 1, 0)
    out["stress_risk"] = out["oldpeak"] * out["exang"]
    out["low_hr"] = np.where(out["thalachh"] < 120, 1, 0)
    out["vessel_risk"] = np.where(out["ca"] > 0, 1, 0)
    out["risk_score"] = (
        out["high_bp"] + out["high_chol"] + out["exang"] + out["vessel_risk"]
    )
    return out


def apply_categorical_levels(df: pd.DataFrame, levels: dict[str, list[float]]) -> pd.DataFrame:
    out = df.copy()
    for col, vals in levels.items():
        out[col] = pd.Categorical(out[col], categories=vals)
    return out


def encode_impute_and_cast(
    df: pd.DataFrame,
    levels: dict[str, list[float]],
    num_medians: dict[str, float],
    cat_modes: dict[str, float],
) -> pd.DataFrame:
    out = apply_categorical_levels(df, levels)
    out = pd.get_dummies(out, columns=ENCODE_COLS, drop_first=True)

    for col in NUM_IMPUTE_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(num_medians[col])

    for col in CAT_IMPUTE_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(cat_modes[col])

    bool_cols = out.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        out[bool_cols] = out[bool_cols].astype(int)

    return out


@st.cache_data(show_spinner=False)
def build_artifacts(data_path: str) -> dict[str, Any]:
    raw_df = pd.read_csv(data_path)
    numeric_df = convert_to_numeric(raw_df)

    missing_summary = pd.DataFrame(
        {
            "Missing Count": numeric_df.isnull().sum(),
            "Missing %": (numeric_df.isnull().sum() / len(numeric_df) * 100).round(3),
        }
    )

    category_levels = {
        col: sorted(numeric_df[col].dropna().unique().tolist()) for col in ENCODE_COLS
    }

    bounds = compute_outlier_bounds(numeric_df, OUTLIER_COLS)
    outliers_before = count_outliers(numeric_df, bounds)
    capped_df = apply_outlier_capping(numeric_df, bounds)
    outliers_after = count_outliers(capped_df, bounds)

    engineered_df = add_engineered_features(capped_df)

    num_medians = {col: float(engineered_df[col].median()) for col in NUM_IMPUTE_COLS}
    cat_modes = {col: float(engineered_df[col].mode()[0]) for col in CAT_IMPUTE_COLS}

    model_df = encode_impute_and_cast(
        engineered_df,
        levels=category_levels,
        num_medians=num_medians,
        cat_modes=cat_modes,
    )

    X = model_df.drop(columns=[TARGET_COL])
    y = model_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return {
        "raw_df": raw_df,
        "numeric_df": numeric_df,
        "capped_df": capped_df,
        "engineered_df": engineered_df,
        "model_df": model_df,
        "missing_summary": missing_summary,
        "outliers_before": outliers_before,
        "outliers_after": outliers_after,
        "bounds": bounds,
        "category_levels": category_levels,
        "num_medians": num_medians,
        "cat_modes": cat_modes,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": X.columns.tolist(),
    }


@st.cache_resource(show_spinner=False)
def train_models(data_path: str) -> dict[str, Any]:
    artifacts = build_artifacts(data_path)
    X_train = artifacts["X_train"]
    y_train = artifacts["y_train"]

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Tuned Logistic Regression": LogisticRegression(C=10, solver="lbfgs", max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Tuned Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="log2",
            max_depth=None,
            random_state=RANDOM_STATE,
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def evaluate_single_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    y_pred = pd.Series(model.predict(X_test), index=y_test.index, name="predicted")
    y_prob = pd.Series(model.predict_proba(X_test)[:, 1], index=y_test.index, name="prob")
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T

    return {
        "metrics": {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred)),
            "Recall": float(recall_score(y_test, y_pred)),
            "F1-Score": float(f1_score(y_test, y_pred)),
            "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
        },
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
    }


def build_model_outputs(models: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for name, model in models.items():
        outputs[name] = evaluate_single_model(model, artifacts["X_test"], artifacts["y_test"])
    return outputs


def preprocess_single_patient(patient_df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    df = convert_to_numeric(patient_df)
    df = apply_outlier_capping(df, artifacts["bounds"])
    df = add_engineered_features(df)
    df = encode_impute_and_cast(
        df,
        levels=artifacts["category_levels"],
        num_medians=artifacts["num_medians"],
        cat_modes=artifacts["cat_modes"],
    )
    df = df.reindex(columns=artifacts["feature_columns"], fill_value=0)
    return df


def plot_confusion(cm: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curves(outputs: dict[str, Any], model_names: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for model_name in model_names:
        auc_value = outputs[model_name]["metrics"]["ROC-AUC"]
        ax.plot(
            outputs[model_name]["fpr"],
            outputs[model_name]["tpr"],
            label=f"{model_name} (AUC={auc_value:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def show_project_story(artifacts: dict[str, Any]) -> None:
    st.markdown(
        """
<div class="hero">
  <h1>Heart Disease Prediction Project</h1>
  <p>Beginner-friendly walkthrough of my full capstone: from raw data to model decisions.</p>
  <p><span class="tag">Medical ML</span><span class="tag">Error Analysis</span><span class="tag">Recall Focus</span></p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{artifacts['raw_df'].shape[0]:,}")
    c2.metric("Original Columns", artifacts["raw_df"].shape[1])
    c3.metric("Model Features", len(artifacts["feature_columns"]))
    c4.metric("Disease Rate", f"{(artifacts['numeric_df'][TARGET_COL].mean() * 100):.1f}%")

    st.subheader("Problem Statement")
    st.markdown(f"- {PROBLEM_STATEMENT}")

    st.subheader("Target Audience")
    for item in TARGET_AUDIENCE:
        st.markdown(f"- {item}")

    st.subheader("Step-by-Step: How Analysis Was Done")
    st.markdown(
        """
1. Load merged heart dataset and inspect shape, datatypes, and summary statistics.
2. Convert object-like clinical fields to numeric safely using `errors='coerce'`.
3. Check missing values and percentages; keep all rows because medical records are valuable.
4. Cap outliers in `chol`, `trestbps`, and `oldpeak` using IQR winsorization.
5. Engineer medical-risk features: `high_bp`, `high_chol`, `stress_risk`, `low_hr`, `vessel_risk`, and `risk_score`.
6. One-hot encode `cp`, `thal`, `slope`, and `restecg`.
7. Impute remaining missing values (median for numeric, mode for selected binary/categorical columns).
8. Split train/test with stratification.
9. Train and compare Logistic Regression, Random Forest, and tuned variants.
10. Prioritize recall in final model choice due the medical context.
11. Perform deep error analysis on misclassified rows (FP/FN patterns and confusion metrics).
"""
    )

    st.subheader("Beginner Guide: What Metrics Mean")
    st.markdown(
        """
- **Recall**: Of actual disease cases, how many the model catches. Most critical for medical screening.
- **Precision**: Of predicted disease cases, how many are truly disease.
- **F1-score**: Balance of recall and precision.
- **Accuracy**: Overall correct predictions.
- **ROC-AUC**: Overall ability to separate disease vs no disease across thresholds.
"""
    )


def show_columns_meanings() -> None:
    st.header("Columns and Short Meanings")
    st.caption("Quick data dictionary for new readers.")

    st.subheader("Original Dataset Columns")
    original_df = pd.DataFrame(ORIGINAL_COLUMN_MEANINGS, columns=["Column", "Short Meaning"])
    st.dataframe(original_df, use_container_width=True, hide_index=True)

    st.subheader("Engineered Features I Added")
    engineered_df = pd.DataFrame(ENGINEERED_COLUMN_MEANINGS, columns=["Column", "Short Meaning"])
    st.dataframe(engineered_df, use_container_width=True, hide_index=True)

    st.subheader("Encoded Groups (Modeling)")
    encoded_df = pd.DataFrame(ENCODED_COLUMN_MEANINGS, columns=["Columns", "Short Meaning"])
    st.dataframe(encoded_df, use_container_width=True, hide_index=True)


def show_all_eda_visuals(artifacts: dict[str, Any]) -> None:
    st.header("EDA: Every Key Visual From The Notebook")
    st.markdown(
        """
I explain each visual in simple terms below:
- What the chart shows.
- What pattern I observed.
- Why that pattern matters for prediction.
"""
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Data Profile",
            "Missing + Outliers",
            "Numerical Visuals",
            "Categorical Visuals",
            "Relationship Visuals",
        ]
    )

    with tab1:
        st.subheader("Dataset Snapshot")
        st.dataframe(artifacts["raw_df"].head(10), use_container_width=True)
        st.caption(
            "This first table helps me confirm the columns, units, and real patient-record structure before modeling."
        )

        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=artifacts["numeric_df"], x=TARGET_COL, palette=["#ef4444", "#3b82f6"], ax=ax)
        ax.set_title("Heart Disease Target Distribution")
        st.pyplot(fig)
        plt.close(fig)
        target_counts = artifacts["numeric_df"][TARGET_COL].value_counts().sort_index()
        target_pct = (
            artifacts["numeric_df"][TARGET_COL].value_counts(normalize=True).sort_index() * 100
        )
        st.markdown(
            """
- This bar chart shows how many records are class `0` (no disease) versus class `1` (disease).
- A near-balanced dataset reduces bias toward one class and makes evaluation more reliable.
- Even with balance, recall remains the priority because missing true disease cases is high risk.
"""
        )
        st.info(
            f"Class counts: no disease = {int(target_counts.get(0, 0))} ({target_pct.get(0, 0):.2f}%), "
            f"disease = {int(target_counts.get(1, 0))} ({target_pct.get(1, 0):.2f}%)."
        )

    with tab2:
        st.subheader("Missing Values")
        st.dataframe(
            artifacts["missing_summary"].sort_values("Missing Count", ascending=False),
            use_container_width=True,
        )
        st.markdown(
            """
- The table quantifies data completeness feature by feature.
- `ca` and `thal` have the highest missing percentages, so they need careful handling.
- I used imputation instead of dropping rows to preserve important medical patterns.
"""
        )

        st.subheader("Missing Matrix (Visual)")
        ax_missing = msno.matrix(
            artifacts["numeric_df"],
            figsize=(12, 4.2),
            fontsize=8,
            color=(0.12, 0.27, 0.54),
        )
        st.pyplot(ax_missing.figure)
        plt.close(ax_missing.figure)
        st.caption(
            "In this matrix, white gaps represent missing entries. Denser gaps on a column mean that feature needs stronger preprocessing attention."
        )

        st.subheader("Outliers Before vs After Winsorization")
        outlier_table = pd.DataFrame(
            {
                "Feature": list(artifacts["outliers_before"].keys()),
                "Before": list(artifacts["outliers_before"].values()),
                "After": [artifacts["outliers_after"][c] for c in artifacts["outliers_before"]],
            }
        )
        st.dataframe(outlier_table, use_container_width=True)
        st.markdown(
            """
- This table confirms that extreme values were capped in selected columns.
- Capping protects models from being overly influenced by rare extreme measurements.
- It keeps all rows while reducing instability from outliers.
"""
        )

        skew_cols = ["age", "trestbps", "chol", "thalachh", "oldpeak"]
        skew_df = artifacts["numeric_df"][skew_cols].skew().rename("Skewness").to_frame()
        st.subheader("Skewness Table")
        st.dataframe(skew_df, use_container_width=True)
        st.caption(
            "Positive skew means long right tail. Here, `chol` and `oldpeak` are more right-skewed, supporting the outlier-handling step."
        )

    with tab3:
        st.subheader("Histogram Distribution of Numerical Features")
        num_cols = ["age", "trestbps", "chol", "thalachh", "oldpeak"]
        fig = artifacts["capped_df"][num_cols].hist(figsize=(12, 7), bins=20)
        plt.suptitle("Distribution of Numerical Features", fontsize=14)
        st.pyplot(plt.gcf())
        plt.close(plt.gcf())
        st.markdown(
            """
- Histograms show spread, central concentration, and shape of each numeric feature.
- I use them to spot skewness and verify whether value ranges look clinically reasonable.
"""
        )

        st.subheader("KDE Distributions by Target")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(["chol", "thalachh", "oldpeak", "trestbps"]):
            sns.kdeplot(
                data=artifacts["capped_df"],
                x=col,
                hue=TARGET_COL,
                fill=True,
                common_norm=False,
                ax=axes[i],
            )
            axes[i].set_title(f"{col} Distribution by Target")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            """
- KDE compares class `0` and class `1` smooth distributions on the same axis.
- `oldpeak` and `thalachh` show clearer separation, so they carry stronger predictive signal.
- `chol` and `trestbps` overlap more, meaning they are weaker alone but still helpful when combined with other features.
"""
        )

        st.subheader("Boxplots (Numerical Features)")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(["chol", "trestbps", "oldpeak", "thalachh"]):
            sns.boxplot(x=artifacts["capped_df"][col], ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Boxplots summarize median, spread, and extreme points. They make outlier presence and variability differences easier to compare quickly."
        )

    with tab4:
        st.subheader("Categorical Countplots vs Target")
        cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            sns.countplot(data=artifacts["numeric_df"], x=col, hue=TARGET_COL, ax=axes[i])
            axes[i].set_title(f"{col} vs target")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            """
- Each subplot compares category counts by target class.
- `cp`, `slope`, `ca`, and `thal` show stronger class contrast, so they are likely high-value predictors.
- `fbs` has more overlap between classes, so it behaves as a weaker standalone signal.
- This explains why tree-based models that combine many interactions performed well.
"""
        )

    with tab5:
        st.subheader("Violin Plots by Target")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(["oldpeak", "chol", "thalachh", "trestbps"]):
            sns.violinplot(x=TARGET_COL, y=col, data=artifacts["capped_df"], ax=axes[i])
            axes[i].set_title(f"{col} by target")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            """
- Violin plots show both density and spread by target class.
- They help verify whether one class concentrates in different value bands than the other.
"""
        )

        st.subheader("Age Distribution by Target")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        sns.histplot(
            data=artifacts["capped_df"],
            x="age",
            hue=TARGET_COL,
            kde=True,
            bins=20,
            ax=ax,
        )
        ax.set_title("Age Distribution by Target")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "This overlay checks whether age alone separates classes. Age contributes, but the overlap shows why multi-feature modeling is necessary."
        )

        st.subheader("Pairplot (Notebook Style)")
        st.caption("This is heavier to render. Turn on when I need to inspect pairwise relations.")
        if st.checkbox("Render Pairplot", key="pairplot_toggle", value=False):
            sample_n = st.slider("Pairplot sample size", min_value=300, max_value=2181, value=900, step=100)
            pair_cols = ["age", "chol", "thalachh", "oldpeak", "trestbps", TARGET_COL]
            pair_df = artifacts["capped_df"][pair_cols].sample(
                n=min(sample_n, len(artifacts["capped_df"])),
                random_state=RANDOM_STATE,
            )
            pair = sns.pairplot(pair_df, hue=TARGET_COL)
            st.pyplot(pair.fig)
            plt.close(pair.fig)
            st.markdown(
                """
- Pairplot reveals pairwise feature relationships and target separation at once.
- Off-diagonal scatter plots show interaction patterns; diagonal plots show each feature distribution.
- I use this as a final EDA check before modeling.
"""
            )


def show_modeling(artifacts: dict[str, Any], models: dict[str, Any], outputs: dict[str, Any]) -> None:
    st.header("Modeling Results and Explanations")
    st.markdown(
        """
Modeling intuition:
- I am not choosing a model by accuracy alone.
- Because this is medical data, I prioritize **recall** to reduce missed disease cases.
- I still check precision, F1-score, and ROC-AUC to keep decisions balanced and clinically useful.
"""
    )

    st.subheader("Notebook Results Table")
    st.dataframe(NOTEBOOK_RESULTS, use_container_width=True)

    st.subheader("Re-run Metrics In This App")
    rerun_rows = []
    for name, pack in outputs.items():
        row = {"Model": name}
        row.update(pack["metrics"])
        rerun_rows.append(row)
    rerun_df = pd.DataFrame(rerun_rows).sort_values("Recall", ascending=False).reset_index(drop=True)
    st.dataframe(rerun_df, use_container_width=True)

    st.subheader("Clinical Trade-off Table (What Matters Most)")
    clinical_rows = []
    for name, pack in outputs.items():
        tn, fp, fn, tp = pack["cm"].ravel()
        disease_total = tp + fn
        healthy_total = tn + fp
        clinical_rows.append(
            {
                "Model": name,
                "Recall": pack["metrics"]["Recall"],
                "Precision": pack["metrics"]["Precision"],
                "FN (Missed disease)": fn,
                "TP (Caught disease)": tp,
                "FNR": fn / disease_total if disease_total > 0 else 0.0,
                "FPR": fp / healthy_total if healthy_total > 0 else 0.0,
                "Missed per 100 disease patients": (fn / disease_total * 100) if disease_total > 0 else 0.0,
            }
        )
    clinical_df = (
        pd.DataFrame(clinical_rows)
        .sort_values(["Recall", "Precision"], ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(clinical_df, use_container_width=True)
    st.caption(
        "I use this table to see direct clinical impact: how many true disease cases the model catches vs misses."
    )

    best_note = NOTEBOOK_RESULTS.sort_values("Recall", ascending=False).iloc[0]
    st.subheader("Best Performing Model (Recall Focus)")
    st.markdown(
        f"""
- **Notebook winner:** {best_note['Model']}
- **Recall:** {best_note['Recall']:.3f} (highest among compared models)
- **Why best for medical screening:** highest recall means fewer missed true disease cases.
- **Precision also strong:** {best_note['Precision']:.3f}, so false alarms are still controlled.
- **F1-score:** {best_note['F1-Score']:.3f}, showing balanced performance.
- **ROC-AUC:** {best_note['ROC-AUC']:.3f}, indicating excellent class separation.
"""
    )
    st.markdown(
        """
Practical intuition:
- Higher recall means fewer dangerous misses (false negatives).
- Precision ensures I do not overwhelm clinicians with too many false alarms.
- F1-score confirms whether recall gains are balanced.
- ROC-AUC confirms overall separability across thresholds, not only one cutoff.
"""
    )

    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_confusion(outputs["Logistic Regression"]["cm"], "Logistic Regression")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig = plot_confusion(outputs["Tuned Random Forest"]["cm"], "Tuned Random Forest")
        st.pyplot(fig)
        plt.close(fig)
    st.markdown(
        """
How to read these matrices:
- Bottom-right (TP): correctly caught disease cases.
- Bottom-left (FN): missed disease cases (most risky error).
- Top-right (FP): false alarms.
- Top-left (TN): correctly identified healthy cases.
"""
    )

    st.subheader("ROC Curves")
    fig = plot_roc_curves(
        outputs,
        ["Logistic Regression", "Random Forest", "Tuned Random Forest"],
        "ROC Comparison",
    )
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "A curve closer to the top-left corner is better. Higher AUC means stronger class discrimination."
    )

    st.subheader("Logistic Regression Classification Report")
    st.dataframe(outputs["Logistic Regression"]["report"], use_container_width=True)
    st.caption(
        "This report breaks performance by each class (`0` and `1`) and helps verify that disease-class recall is acceptable."
    )

    st.subheader("Feature Importance")
    log_model = models["Logistic Regression"]
    rf_model = models["Random Forest"]

    log_importance = pd.DataFrame(
        {
            "Feature": artifacts["feature_columns"],
            "Coefficient": log_model.coef_[0],
        }
    )
    log_importance["Absolute"] = log_importance["Coefficient"].abs()
    log_importance = log_importance.sort_values("Absolute", ascending=False)

    rf_importance = pd.DataFrame(
        {
            "Feature": artifacts["feature_columns"],
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.barplot(data=log_importance.head(15), x="Coefficient", y="Feature", ax=ax)
        ax.set_title("Top Logistic Regression Features")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.barplot(data=rf_importance.head(15), x="Importance", y="Feature", ax=ax)
        ax.set_title("Top Random Forest Features")
        st.pyplot(fig)
        plt.close(fig)
    st.markdown(
        """
Feature-importance intuition:
- Logistic coefficients show direction: positive values push prediction toward disease.
- Random Forest importance shows contribution strength, not direction.
- If the same features are repeatedly strong across models, confidence in their signal increases.
"""
    )


def build_error_sets(
    artifacts: dict[str, Any],
    model_output: dict[str, Any],
) -> dict[str, Any]:
    y_test = artifacts["y_test"]
    y_pred = model_output["y_pred"]
    y_prob = model_output["y_prob"]
    X_test = artifacts["X_test"]

    fn_idx = y_test[(y_test == 1) & (y_pred == 0)].index
    fp_idx = y_test[(y_test == 0) & (y_pred == 1)].index
    tp_idx = y_test[(y_test == 1) & (y_pred == 1)].index
    tn_idx = y_test[(y_test == 0) & (y_pred == 0)].index

    ref_df = artifacts["engineered_df"].copy()
    selected_cols = BASE_COLUMNS + [
        "high_bp",
        "high_chol",
        "stress_risk",
        "low_hr",
        "vessel_risk",
        "risk_score",
    ]

    error_rows = ref_df.loc[fn_idx.union(fp_idx), selected_cols].copy()
    error_rows["Actual"] = y_test.loc[error_rows.index].values
    error_rows["Predicted"] = y_pred.loc[error_rows.index].values
    error_rows["Disease Probability"] = y_prob.loc[error_rows.index].round(4).values
    error_rows["Distance From 0.5"] = (
        np.abs(y_prob.loc[error_rows.index].values - 0.5).round(4)
    )
    error_rows["Error Type"] = np.where(error_rows["Actual"] == 1, "False Negative", "False Positive")

    false_negatives = X_test.loc[fn_idx]
    true_positives = X_test.loc[tp_idx]

    comparison = pd.DataFrame()
    if len(false_negatives) > 0 and len(true_positives) > 0:
        comparison = pd.DataFrame(
            {
                "False Negatives Mean": false_negatives.mean(numeric_only=True),
                "True Positives Mean": true_positives.mean(numeric_only=True),
            }
        )
        comparison["Absolute Gap"] = (
            comparison["False Negatives Mean"] - comparison["True Positives Mean"]
        ).abs()
        comparison = comparison.sort_values("Absolute Gap", ascending=False)

    cm = model_output["cm"]
    tn, fp, fn, tp = cm.ravel()
    misclassification_rate = (fp + fn) / (tn + fp + fn + tp)
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    outcome_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred,
            "Disease Probability": y_prob,
        }
    )
    outcome_df["Outcome Type"] = "TN"
    outcome_df.loc[(outcome_df["Actual"] == 1) & (outcome_df["Predicted"] == 1), "Outcome Type"] = "TP"
    outcome_df.loc[(outcome_df["Actual"] == 1) & (outcome_df["Predicted"] == 0), "Outcome Type"] = "FN"
    outcome_df.loc[(outcome_df["Actual"] == 0) & (outcome_df["Predicted"] == 1), "Outcome Type"] = "FP"

    probability_summary = (
        outcome_df.groupby("Outcome Type")["Disease Probability"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    probability_summary.columns = [
        "Outcome Type",
        "Count",
        "Mean Probability",
        "Median Probability",
        "Min Probability",
        "Max Probability",
    ]

    return {
        "fn_idx": fn_idx,
        "fp_idx": fp_idx,
        "tp_idx": tp_idx,
        "tn_idx": tn_idx,
        "error_rows": error_rows,
        "comparison": comparison,
        "misclassification_rate": misclassification_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "outcome_df": outcome_df,
        "probability_summary": probability_summary,
    }


def show_error_analysis(artifacts: dict[str, Any], outputs: dict[str, Any]) -> None:
    st.header("Deep Error Analysis: Misclassified Rows and Patterns")
    st.caption("This section repeats and expands my notebook error-analysis workflow.")
    st.markdown(
        """
Error-analysis intuition:
- Not all errors are equally risky.
- **False negatives (FN)** are most dangerous because disease cases are missed.
- I use this section to understand exactly which patients are missed and what feature patterns are common.
"""
    )

    model_name = st.selectbox(
        "Select model for error analysis",
        ["Logistic Regression", "Tuned Random Forest"],
        index=0,
    )
    selected_output = outputs[model_name]
    error_sets = build_error_sets(artifacts, selected_output)

    tn, fp, fn, tp = selected_output["cm"].ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negatives", tn)
    c2.metric("False Positives", fp)
    c3.metric("False Negatives", fn)
    c4.metric("True Positives", tp)

    c5, c6, c7 = st.columns(3)
    c5.metric("Misclassification Rate", f"{error_sets['misclassification_rate']:.3f}")
    c6.metric("False Positive Rate", f"{error_sets['false_positive_rate']:.3f}")
    c7.metric("False Negative Rate", f"{error_sets['false_negative_rate']:.3f}")
    st.info(
        f"Clinical readout: FN rate is {error_sets['false_negative_rate']:.3f}. "
        "This is the key number I watch first because it reflects missed true disease cases."
    )

    fig = plot_confusion(selected_output["cm"], f"Confusion Matrix - {model_name}")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "I read confusion matrix cells as real patient outcomes, not just abstract counts."
    )

    st.subheader("Error Type Table")
    error_table = pd.DataFrame(
        {
            "Prediction Type": ["True Negative", "False Positive", "False Negative", "True Positive"],
            "Count": [tn, fp, fn, tp],
            "Meaning": [
                "Correctly predicted no heart disease",
                "Predicted disease when patient is healthy",
                "Missed actual heart disease case",
                "Correctly predicted heart disease",
            ],
        }
    )
    st.dataframe(error_table, use_container_width=True)

    st.subheader("Probability Behavior by Outcome Type")
    st.dataframe(error_sets["probability_summary"], use_container_width=True)
    st.markdown(
        """
- `TP` rows should usually have higher disease probabilities.
- `FN` rows often sit closer to or below the 0.5 decision threshold.
- This helps me see whether misses happen because scores are borderline or because patterns are truly ambiguous.
"""
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.boxplot(
        data=error_sets["outcome_df"],
        x="Outcome Type",
        y="Disease Probability",
        order=["TN", "FP", "FN", "TP"],
        ax=ax,
    )
    ax.set_title("Predicted Disease Probability by Outcome Type")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Misclassified Rows")
    view_type = st.radio(
        "Choose rows",
        ["All misclassified rows", "False negatives only", "False positives only"],
        horizontal=True,
    )
    error_rows = error_sets["error_rows"]
    sort_choice = st.selectbox(
        "Sort misclassified rows by",
        ["Most confident errors", "Most borderline errors"],
        index=0,
    )
    if sort_choice == "Most confident errors":
        error_rows = error_rows.sort_values("Distance From 0.5", ascending=False)
    else:
        error_rows = error_rows.sort_values("Distance From 0.5", ascending=True)

    if view_type == "False negatives only":
        show_rows = error_rows[error_rows["Error Type"] == "False Negative"]
    elif view_type == "False positives only":
        show_rows = error_rows[error_rows["Error Type"] == "False Positive"]
    else:
        show_rows = error_rows

    st.dataframe(show_rows, use_container_width=True)
    st.caption(
        "Distance from 0.5 helps distinguish borderline mistakes from high-confidence mistakes."
    )

    st.subheader("Pattern Comparison: False Negatives vs True Positives")
    if not error_sets["comparison"].empty:
        st.dataframe(error_sets["comparison"], use_container_width=True)
        top_patterns = error_sets["comparison"].head(8).reset_index()
        top_patterns.columns = ["Feature", "False Negatives Mean", "True Positives Mean", "Absolute Gap"]
        st.markdown("Top gap features suggesting where misses happen most:")
        for _, row in top_patterns.iterrows():
            st.markdown(
                f"- `{row['Feature']}` gap = `{row['Absolute Gap']:.3f}` "
                f"(FN mean `{row['False Negatives Mean']:.3f}` vs TP mean `{row['True Positives Mean']:.3f}`)."
            )
    else:
        st.info("Not enough false negatives/true positives to compute pattern comparison for this model.")

    st.subheader("Error Distribution Visuals (Notebook Style)")
    fn_idx = error_sets["fn_idx"]
    tp_idx = error_sets["tp_idx"]
    source = artifacts["X_test"]

    if len(fn_idx) > 0 and len(tp_idx) > 0:
        for feature in ["oldpeak", "thalachh", "chol", "trestbps"]:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            sns.kdeplot(source.loc[fn_idx, feature], label="False Negatives", fill=True, ax=ax)
            sns.kdeplot(source.loc[tp_idx, feature], label="True Positives", fill=True, ax=ax)
            ax.set_title(f"{feature} Distribution: FN vs TP")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("KDE error-distribution plots skipped because one group is empty.")

    st.markdown(
        """
**How to interpret patterns:**  
If a feature has a large FN-vs-TP gap, those values may confuse the classifier and lead to missed disease cases.
Use this to guide threshold tuning, feature redesign, or class-weight strategies in future iterations.
"""
    )
    st.markdown(
        """
What I can do next from this analysis:
1. Lower decision threshold slightly if recall needs to increase further.
2. Add cost-sensitive learning or class weights to penalize false negatives more.
3. Add richer clinical features around patterns seen in FN rows.
4. Validate with clinician feedback before deployment decisions.
"""
    )


def show_prediction(artifacts: dict[str, Any], models: dict[str, Any]) -> None:
    st.header("Patient-Level Prediction")
    st.caption("Uses my tuned Random Forest pipeline.")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=28, max_value=90, value=54, step=1)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=1)
            trestbps = st.number_input("Resting BP", min_value=80.0, max_value=240.0, value=130.0, step=1.0)
            chol = st.number_input("Cholesterol", min_value=100.0, max_value=650.0, value=240.0, step=1.0)

        with c2:
            fbs = st.selectbox("Fasting Blood Sugar (fbs)", [0, 1], index=0)
            restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=1)
            thalachh = st.number_input("Max Heart Rate (thalachh)", min_value=60.0, max_value=230.0, value=150.0, step=1.0)
            exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1], index=0)
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

        with c3:
            slope = st.selectbox("Slope", [0, 1, 2, 3], index=2)
            ca = st.number_input("Number of Vessels (ca)", min_value=0.0, max_value=4.0, value=0.0, step=1.0)
            thal = st.selectbox("Thal", [1, 2, 3, 6, 7], index=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_input = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalachh": thalachh,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal,
                }
            ]
        )

        ready = preprocess_single_patient(raw_input, artifacts)
        model = models["Tuned Random Forest"]
        pred = int(model.predict(ready)[0])
        prob = float(model.predict_proba(ready)[0][1])

        if pred == 1:
            st.error(f"Predicted class: Heart Disease Risk (1) | Probability: {prob:.2%}")
        else:
            st.success(f"Predicted class: Lower Risk (0) | Disease Probability: {prob:.2%}")

        st.warning(
            "This tool is for decision support only. It is not a medical diagnosis and should not replace clinician judgment."
        )


def main() -> None:
    apply_theme()

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        st.stop()

    artifacts = build_artifacts(str(DATA_PATH))
    models = train_models(str(DATA_PATH))
    outputs = build_model_outputs(models, artifacts)

    section = st.sidebar.radio(
        "Navigation",
        [
            "1) Project Story",
            "2) Columns and Meanings",
            "3) All EDA Visuals",
            "4) Modeling Results",
            "5) Error Analysis",
            "6) Patient Prediction",
        ],
    )

    if section == "1) Project Story":
        show_project_story(artifacts)
    elif section == "2) Columns and Meanings":
        show_columns_meanings()
    elif section == "3) All EDA Visuals":
        show_all_eda_visuals(artifacts)
    elif section == "4) Modeling Results":
        show_modeling(artifacts, models, outputs)
    elif section == "5) Error Analysis":
        show_error_analysis(artifacts, outputs)
    else:
        show_prediction(artifacts, models)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
