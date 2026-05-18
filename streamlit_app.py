from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Heart Disease Prediction - Capstone Dashboard",
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


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TO_NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def compute_outlier_bounds(df: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
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


def apply_categorical_levels(
    df: pd.DataFrame, category_levels: dict[str, list[float]]
) -> pd.DataFrame:
    out = df.copy()
    for col, levels in category_levels.items():
        out[col] = pd.Categorical(out[col], categories=levels)
    return out


def encode_impute_and_cast(
    df: pd.DataFrame,
    category_levels: dict[str, list[float]],
    num_medians: dict[str, float],
    cat_modes: dict[str, float],
) -> pd.DataFrame:
    out = apply_categorical_levels(df, category_levels)
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


@st.cache_data
def build_artifacts(data_path: str) -> dict[str, object]:
    raw_df = pd.read_csv(data_path)
    numeric_df = convert_to_numeric(raw_df)

    missing_summary = pd.DataFrame(
        {
            "Missing Count": numeric_df.isna().sum(),
            "Missing %": (numeric_df.isna().sum() / len(numeric_df) * 100).round(3),
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
        category_levels=category_levels,
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
        "model_df": model_df,
        "missing_summary": missing_summary,
        "outliers_before": outliers_before,
        "outliers_after": outliers_after,
        "category_levels": category_levels,
        "bounds": bounds,
        "num_medians": num_medians,
        "cat_modes": cat_modes,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": X.columns.tolist(),
    }


@st.cache_resource
def train_best_model(artifacts: dict[str, object]) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        random_state=RANDOM_STATE,
    )
    model.fit(artifacts["X_train"], artifacts["y_train"])
    return model


def preprocess_single_patient(
    patient_df: pd.DataFrame,
    artifacts: dict[str, object],
) -> pd.DataFrame:
    df = convert_to_numeric(patient_df)
    df = apply_outlier_capping(df, artifacts["bounds"])
    df = add_engineered_features(df)
    df = encode_impute_and_cast(
        df,
        category_levels=artifacts["category_levels"],
        num_medians=artifacts["num_medians"],
        cat_modes=artifacts["cat_modes"],
    )
    df = df.reindex(columns=artifacts["feature_columns"], fill_value=0)
    return df


def show_overview(artifacts: dict[str, object]) -> None:
    st.title("Heart Disease Prediction - Capstone Dashboard")
    st.caption("Built from your `project.ipynb` workflow and findings.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{artifacts['raw_df'].shape[0]:,}")
    col2.metric("Original Features", artifacts["raw_df"].shape[1] - 1)
    col3.metric("Engineered + Encoded Features", artifacts["model_df"].shape[1] - 1)

    st.subheader("Step-by-Step Workflow")
    st.markdown(
        """
1. Loaded and profiled the merged heart dataset (2,181 records).
2. Converted string-based clinical fields into numeric values safely (`errors='coerce'`).
3. Audited missingness and retained records (no row dropping) due medical data value.
4. Capped extreme outliers in `chol`, `trestbps`, and `oldpeak` using IQR winsorization.
5. Engineered medical risk features: `high_bp`, `high_chol`, `stress_risk`, `low_hr`, `vessel_risk`, `risk_score`.
6. One-hot encoded key categorical variables (`cp`, `thal`, `slope`, `restecg`).
7. Imputed remaining missing values with median/mode and cast boolean dummies to integers.
8. Split train/test with stratification and compared multiple models.
9. Selected Tuned Random Forest as the best clinical model, mainly due higher recall.
"""
    )

    st.subheader("Class Balance")
    target_counts = artifacts["numeric_df"][TARGET_COL].value_counts().sort_index()
    target_pct = (
        artifacts["numeric_df"][TARGET_COL].value_counts(normalize=True).sort_index() * 100
    )
    balance_df = pd.DataFrame(
        {
            "Target": target_counts.index.map({0: "No disease (0)", 1: "Disease (1)"}),
            "Count": target_counts.values,
            "Percent": target_pct.values.round(2),
        }
    )
    st.dataframe(balance_df, use_container_width=True)
    st.info(
        "Your classes are almost balanced (~50/50), so accuracy is meaningful, but recall is still critical for patient safety."
    )


def show_eda(artifacts: dict[str, object]) -> None:
    st.header("EDA Findings and Chart Explanations")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Missingness", "Numerical Charts", "Categorical Charts", "Outliers"]
    )

    with tab1:
        st.subheader("Missing Values Before Imputation")
        miss = artifacts["missing_summary"].sort_values("Missing Count", ascending=False)
        st.dataframe(miss, use_container_width=True)
        st.markdown(
            """
- `ca` (~13.34%) and `thal` (~12.20%) had the largest missingness.
- You correctly avoided row dropping to preserve medical signal and sample size.
- Imputation was appropriate for this dataset size and clinical context.
"""
        )

    with tab2:
        st.subheader("KDE Distributions by Target")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(["chol", "thalachh", "oldpeak", "trestbps"]):
            sns.kdeplot(
                data=artifacts["capped_df"],
                x=col,
                hue=TARGET_COL,
                fill=True,
                ax=axes[i],
                common_norm=False,
            )
            axes[i].set_title(f"{col} by target")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            """
- `chol`: heavy overlap, so cholesterol alone is a moderate predictor.
- `thalachh`: clearer shift between classes, making it a stronger signal.
- `oldpeak`: visible class difference; one of the strongest standalone numerical indicators.
- `trestbps`: overlap is high, so weak-to-moderate alone but useful in combination.
"""
        )

    with tab3:
        st.subheader("Categorical Feature Countplots")
        cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            sns.countplot(data=artifacts["numeric_df"], x=col, hue=TARGET_COL, ax=axes[i])
            axes[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            """
- Very strong separation appeared in `cp`, `slope`, `ca`, and `thal`.
- `fbs` was weak alone (similar counts between classes).
- `restecg` and `exang` showed useful class separation.
- These findings align with why tree-based models performed strongly.
"""
        )

    with tab4:
        st.subheader("Outlier Check (IQR)")
        before = pd.Series(artifacts["outliers_before"], name="Before Capping")
        after = pd.Series(artifacts["outliers_after"], name="After Capping")
        outlier_df = pd.concat([before, after], axis=1).reset_index()
        outlier_df.columns = ["Feature", "Before Capping", "After Capping"]
        st.dataframe(outlier_df, use_container_width=True)
        st.success("Outliers in `chol`, `trestbps`, and `oldpeak` were reduced to 0 after winsorization.")


def show_modeling(artifacts: dict[str, object], best_model: RandomForestClassifier) -> None:
    st.header("Model Performance")
    st.subheader("Notebook Results Summary")
    st.dataframe(NOTEBOOK_RESULTS, use_container_width=True)

    recall_rank = NOTEBOOK_RESULTS.sort_values("Recall", ascending=False).reset_index(drop=True)
    st.subheader("Recall Ranking (Most Important for Medical Screening)")
    st.dataframe(recall_rank[["Model", "Recall", "Precision", "F1-Score", "ROC-AUC"]], use_container_width=True)

    best_row = NOTEBOOK_RESULTS.loc[NOTEBOOK_RESULTS["Model"] == "Tuned Random Forest"].iloc[0]
    fn_rate = 1 - float(best_row["Recall"])

    st.subheader("Best Model Explanation")
    st.markdown(
        f"""
- **Best model:** Tuned Random Forest
- **Recall:** {best_row['Recall']:.3f} (highest in your notebook table)
- **Why recall matters here:** it minimizes missed true heart disease cases (false negatives).
- **Estimated false-negative rate:** {fn_rate:.3%}
- **Precision:** {best_row['Precision']:.3f} (few unnecessary alarms)
- **F1-score:** {best_row['F1-Score']:.3f} (good precision/recall balance)
- **ROC-AUC:** {best_row['ROC-AUC']:.3f} (excellent class-separation ability)
"""
    )

    y_test = artifacts["y_test"]
    y_pred_best = best_model.predict(artifacts["X_test"])
    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()

    st.subheader("Confusion Matrix - Tuned Random Forest")
    fig, ax = plt.subplots(figsize=(5, 4))
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
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        f"""
- True Negatives: **{tn}**
- False Positives: **{fp}**
- False Negatives: **{fn}**
- True Positives: **{tp}**
"""
    )

    st.subheader("Notebook Logistic Error Analysis (Your Original Findings)")
    st.markdown(
        """
- TN: 180, FP: 40, FN: 52, TP: 165  
- Misclassification rate: 0.211  
- False Positive Rate: 0.182  
- False Negative Rate: 0.240  
"""
    )
    st.info(
        "This comparison reinforces why you moved toward stronger tree-based models to improve recall and reduce missed disease."
    )


def show_prediction(artifacts: dict[str, object], best_model: RandomForestClassifier) -> None:
    st.header("Patient-Level Prediction (Tuned Random Forest)")
    st.caption(
        "This uses your engineered-feature pipeline and tuned Random Forest settings from the notebook."
    )

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=28, max_value=90, value=54, step=1)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], index=1)
            trestbps = st.number_input("Resting BP (trestbps)", min_value=80.0, max_value=240.0, value=130.0, step=1.0)
            chol = st.number_input("Cholesterol (chol)", min_value=100.0, max_value=650.0, value=240.0, step=1.0)

        with c2:
            fbs = st.selectbox("Fasting Blood Sugar (fbs)", options=[0, 1], index=0)
            restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], index=1)
            thalachh = st.number_input("Max Heart Rate (thalachh)", min_value=60.0, max_value=230.0, value=150.0, step=1.0)
            exang = st.selectbox("Exercise-Induced Angina (exang)", options=[0, 1], index=0)
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

        with c3:
            slope = st.selectbox("Slope", options=[0, 1, 2, 3], index=2)
            ca = st.number_input("Number of Vessels (ca)", min_value=0.0, max_value=4.0, value=0.0, step=1.0)
            thal = st.selectbox("Thal", options=[1, 2, 3, 6, 7], index=1)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        patient_raw = pd.DataFrame(
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

        patient_ready = preprocess_single_patient(patient_raw, artifacts)
        pred = int(best_model.predict(patient_ready)[0])
        prob = float(best_model.predict_proba(patient_ready)[0][1])

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"Predicted class: **Heart Disease Risk (1)** | Probability: **{prob:.2%}**")
        else:
            st.success(f"Predicted class: **Lower Risk (0)** | Probability of disease: **{prob:.2%}**")

        st.warning(
            "Clinical note: This is a decision-support prototype, not a medical diagnosis. Final decisions must remain with healthcare professionals."
        )


def main() -> None:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        st.stop()

    artifacts = build_artifacts(str(DATA_PATH))
    best_model = train_best_model(artifacts)

    section = st.sidebar.radio(
        "Navigation",
        [
            "1) Project Workflow",
            "2) EDA Findings",
            "3) Model Performance",
            "4) Patient Prediction",
        ],
    )

    if section == "1) Project Workflow":
        show_overview(artifacts)
    elif section == "2) EDA Findings":
        show_eda(artifacts)
    elif section == "3) Model Performance":
        show_modeling(artifacts, best_model)
    else:
        show_prediction(artifacts, best_model)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
