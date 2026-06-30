"""
train_model.py
---------------
End-to-end training pipeline for Smart Lender, mirroring the architecture:

    Dataset Collection
        -> Data Preprocessing (missing values, encoding, outliers, SMOTE, scaling)
        -> Train-Test Split
        -> Model Training (Decision Tree, Random Forest, KNN, XGBoost)
        -> Model Evaluation (accuracy, confusion matrix, classification report, CV)
        -> Best Model Selection
        -> Model Storage (rdf.pkl)

Run from the project root:
    python train_model.py
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend - safe for VS Code / terminal / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERICAL_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "TotalIncome",
    "Credit_History",
]
TARGET_COL = "Loan_Status"
RANDOM_STATE = 42


def log(msg: str) -> None:
    print(f"[train_model] {msg}")


# ---------------------------------------------------------------------------
# 1. Dataset Collection
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python data/generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    log(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["LoanAmount", "Loan_Amount_Term"]:
        df[col] = df[col].fillna(df[col].median())
    # Credit_History missing is informative-ish in real data; impute with mode (most common = has good history)
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    log(f"Missing values handled. Remaining NaNs: {int(df.isna().sum().sum())}")
    return df


def handle_outliers(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Cap outliers using the IQR rule (winsorizing) instead of dropping rows,
    so we don't lose otherwise-valid applicants."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped:
            log(f"  Capped {n_capped} outliers in {col} -> [{lower:.0f}, {upper:.0f}]")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    return df


def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    target_encoder = LabelEncoder()
    df[TARGET_COL] = target_encoder.fit_transform(df[TARGET_COL])  # N=0, Y=1
    encoders[TARGET_COL] = target_encoder
    log(f"Encoded categorical columns: {CATEGORICAL_COLS} + target")
    return df, encoders


def preprocess(df: pd.DataFrame):
    df = df.drop(columns=["Loan_ID"])
    df = handle_missing_values(df)
    df = handle_outliers(df, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)
    return df, encoders


# ---------------------------------------------------------------------------
# 3. Train-Test Split + SMOTE balancing + Feature scaling
# ---------------------------------------------------------------------------
def split_balance_scale(df: pd.DataFrame):
    feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS
    X = df[feature_cols]
    y = df[TARGET_COL]

    log(f"Class balance before split -> {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # SMOTE is fit on the TRAINING split only, to avoid leaking test information
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    log(f"Class balance after SMOTE (train only) -> {pd.Series(y_train_bal).value_counts().to_dict()}")

    scaler = StandardScaler()
    X_train_scaled = X_train_bal.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train_bal[NUMERICAL_COLS])
    X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    return X_train_scaled, X_test_scaled, y_train_bal, y_test, scaler, feature_cols


# ---------------------------------------------------------------------------
# 4 & 5. Model Training + Evaluation
# ---------------------------------------------------------------------------
def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "XGBoost": XGBClassifier(
            n_estimators=210,
            max_depth=4,
            learning_rate=0.075,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=1.2,
            reg_alpha=0.2,
            min_child_weight=1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        log(f"Training {name} ...")
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cm = confusion_matrix(y_test, test_pred)
        report = classification_report(y_test, test_pred, target_names=["Rejected", "Approved"])

        results[name] = {
            "model": model,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm,
            "classification_report": report,
        }

        log(
            f"  {name}: train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
            f"cv_acc={cv_scores.mean():.3f}(+/-{cv_scores.std():.3f})"
        )
    return results


# ---------------------------------------------------------------------------
# 6. Best Model Selection
# ---------------------------------------------------------------------------
def select_best_model(results):
    # Model selection uses 5-fold CV accuracy (computed on the training data),
    # NOT the held-out test accuracy - this avoids leaking test-set information
    # into the model-choice decision. The test set is reserved purely for the
    # final, unbiased accuracy report.
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    log(
        f"Best model selected: {best_name} "
        f"(cv_acc={results[best_name]['cv_mean']:.3f}, test_acc={results[best_name]['test_accuracy']:.3f})"
    )
    return best_name, results[best_name]


# ---------------------------------------------------------------------------
# Reporting (plots + text report) - not in the original diagram boxes but
# useful evidence that evaluation actually happened.
# ---------------------------------------------------------------------------
def save_reports(results, best_name, feature_cols):
    # Model comparison bar chart
    names = list(results.keys())
    train_accs = [results[n]["train_accuracy"] for n in names]
    test_accs = [results[n]["test_accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, train_accs, width, label="Train Accuracy", color="#1F3B5C")
    ax.bar(x + width / 2, test_accs, width, label="Test Accuracy", color="#0FA3B1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison - Train vs Test Accuracy")
    ax.legend()
    for i, v in enumerate(train_accs):
        ax.text(i - width / 2, v + 0.015, f"{v:.2%}", ha="center", fontsize=8)
    for i, v in enumerate(test_accs):
        ax.text(i + width / 2, v + 0.015, f"{v:.2%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Confusion matrix heatmap for the best model
    cm = results[best_name]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_best_model.png", dpi=150)
    plt.close(fig)

    # Feature importance (tree-based models only)
    best_model = results[best_name]["model"]
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        importances.plot(kind="barh", ax=ax, color="#0FA3B1")
        ax.set_title(f"Feature Importance - {best_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)

    # Text report
    lines = ["SMART LENDER - MODEL EVALUATION REPORT", "=" * 45, ""]
    for name in names:
        r = results[name]
        lines.append(f"## {name}")
        lines.append(f"Train Accuracy : {r['train_accuracy']:.4f}")
        lines.append(f"Test Accuracy  : {r['test_accuracy']:.4f}")
        lines.append(f"5-Fold CV Acc  : {r['cv_mean']:.4f} (+/- {r['cv_std']:.4f})")
        lines.append("Classification Report:")
        lines.append(r["classification_report"])
        lines.append("-" * 45)
    lines.append(f"\nBEST MODEL SELECTED: {best_name}")
    report_text = "\n".join(lines)
    (REPORTS_DIR / "model_evaluation_report.txt").write_text(report_text)

    summary = {
        name: {
            "train_accuracy": round(results[name]["train_accuracy"], 4),
            "test_accuracy": round(results[name]["test_accuracy"], 4),
            "cv_mean": round(results[name]["cv_mean"], 4),
        }
        for name in names
    }
    summary["best_model"] = best_name
    (REPORTS_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    log("Saved plots + report to reports/")


# ---------------------------------------------------------------------------
# 7. Model Storage
# ---------------------------------------------------------------------------
def save_model_bundle(best_name, best_result, encoders, scaler, feature_cols):
    bundle = {
        "model": best_result["model"],
        "model_name": best_name,
        "scaler": scaler,
        "encoders": encoders,  # one LabelEncoder per categorical column + target
        "feature_columns": feature_cols,  # exact column order expected by the model
        "categorical_columns": CATEGORICAL_COLS,
        "numerical_columns": NUMERICAL_COLS,
        "test_accuracy": best_result["test_accuracy"],
        "train_accuracy": best_result["train_accuracy"],
    }
    out_path = MODEL_DIR / "rdf.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    log(f"Saved best model bundle -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_dataset()
    df, encoders = preprocess(df)
    X_train, X_test, y_train, y_test, scaler, feature_cols = split_balance_scale(df)

    models = build_models()
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    best_name, best_result = select_best_model(results)

    save_reports(results, best_name, feature_cols)
    save_model_bundle(best_name, best_result, encoders, scaler, feature_cols)

    log("Training pipeline complete.")


if __name__ == "__main__":
    main()
