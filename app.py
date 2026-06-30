"""
app.py
------
Flask application for Smart Lender.

Routes
------
GET  /          -> home.html        (landing page)
GET  /predict   -> predict.html     (application form)
POST /submit    -> submit.html      (validates input, runs the model, shows result)

The trained model bundle (model/rdf.pkl) contains everything needed to turn a
raw form submission into a prediction: the fitted model itself, the
LabelEncoders used during training, the StandardScaler, and the exact
feature-column order the model expects. This guarantees that prediction-time
preprocessing exactly mirrors training-time preprocessing.
"""

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "rdf.pkl"

app = Flask(__name__)
app.secret_key = "smart-lender-dev-secret-key"  # fine for local/demo use only

# ---------------------------------------------------------------------------
# Load the trained model bundle once, at startup (not per-request)
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH}.\n"
        "Train the model first by running:  python train_model.py"
    )

with open(MODEL_PATH, "rb") as f:
    BUNDLE = pickle.load(f)

MODEL = BUNDLE["model"]
SCALER = BUNDLE["scaler"]
ENCODERS = BUNDLE["encoders"]
FEATURE_COLUMNS = BUNDLE["feature_columns"]
CATEGORICAL_COLUMNS = BUNDLE["categorical_columns"]
NUMERICAL_COLUMNS = BUNDLE["numerical_columns"]
MODEL_NAME = BUNDLE["model_name"]
TRAIN_ACCURACY = BUNDLE["train_accuracy"]
TEST_ACCURACY = BUNDLE["test_accuracy"]

# Dropdown choices shown in the form - identical strings to what the
# LabelEncoders were fitted on during training.
FORM_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "Property_Area": ["Urban", "Semiurban", "Rural"],
}

REQUIRED_FIELDS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

FIELD_LABELS = {
    "Gender": "Gender",
    "Married": "Marital status",
    "Dependents": "Number of dependents",
    "Education": "Education",
    "Self_Employed": "Self-employment status",
    "Property_Area": "Property area",
    "ApplicantIncome": "Applicant income",
    "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount",
    "Loan_Amount_Term": "Loan term",
    "Credit_History": "Credit history",
}


@app.route("/")
def home():
    return render_template(
        "home.html",
        model_name=MODEL_NAME,
        train_acc=round(TRAIN_ACCURACY * 100, 1),
        test_acc=round(TEST_ACCURACY * 100, 1),
    )


@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html", options=FORM_OPTIONS, form_data={})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_and_parse(form):
    """Validates the raw form submission and returns (clean_data, errors)."""
    errors = []

    # 1. Presence check
    for field in REQUIRED_FIELDS:
        if not form.get(field, "").strip():
            errors.append(f"{FIELD_LABELS[field]} is required.")
    if errors:
        return None, errors

    data = {col: form.get(col) for col in CATEGORICAL_COLUMNS}

    # 2. Categorical values must be one the model was trained on
    for col in CATEGORICAL_COLUMNS:
        if data[col] not in ENCODERS[col].classes_:
            errors.append(f"Invalid value selected for {FIELD_LABELS[col]}.")
    if errors:
        return None, errors

    # 3. Numeric parsing
    numeric_fields = {
        "ApplicantIncome": float,
        "CoapplicantIncome": float,
        "LoanAmount": float,
        "Loan_Amount_Term": float,
        "Credit_History": float,
    }
    for field, caster in numeric_fields.items():
        try:
            data[field] = caster(form.get(field))
        except (TypeError, ValueError):
            errors.append(f"{FIELD_LABELS[field]} must be a valid number.")
    if errors:
        return None, errors

    # 4. Range / business-rule checks
    if data["ApplicantIncome"] < 0 or data["CoapplicantIncome"] < 0:
        errors.append("Income values cannot be negative.")
    if data["LoanAmount"] <= 0:
        errors.append("Loan amount must be greater than zero.")
    if data["Loan_Amount_Term"] <= 0:
        errors.append("Loan term must be greater than zero.")
    if data["Credit_History"] not in (0.0, 1.0):
        errors.append("Credit history must be either 0 (none) or 1 (has credit history).")
    if errors:
        return None, errors

    data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
    return data, []


def build_feature_vector(data: dict) -> pd.DataFrame:
    """Turns validated form data into a single-row DataFrame, encoded and
    scaled exactly as during training, in the exact column order the model
    expects."""
    row = {}
    for col in CATEGORICAL_COLUMNS:
        row[col] = ENCODERS[col].transform([data[col]])[0]
    for col in NUMERICAL_COLUMNS:
        row[col] = data[col]

    df = pd.DataFrame([row])[FEATURE_COLUMNS]
    df[NUMERICAL_COLUMNS] = SCALER.transform(df[NUMERICAL_COLUMNS])
    return df


@app.route("/submit", methods=["POST"])
def submit():
    data, errors = validate_and_parse(request.form)

    if errors:
        for err in errors:
            flash(err, "error")
        return render_template("predict.html", options=FORM_OPTIONS, form_data=request.form), 400

    try:
        features = build_feature_vector(data)
        prediction = int(MODEL.predict(features)[0])
        probabilities = MODEL.predict_proba(features)[0]
        approved = prediction == 1
        confidence = round(float(probabilities[1] if approved else probabilities[0]) * 100, 1)
    except Exception as exc:  # noqa: BLE001 - surface a friendly error, log the real one
        app.logger.exception("Prediction failed")
        flash(f"We couldn't generate a prediction ({exc}). Please check your inputs and try again.", "error")
        return render_template("predict.html", options=FORM_OPTIONS, form_data=request.form), 500

    # Confidence ring geometry (SVG circle, r=80 -> circumference ~= 502.65)
    circumference = 2 * 3.14159265 * 80
    ring_offset = round(circumference * (1 - confidence / 100), 2)

    summary_rows = [
        ("Gender", data["Gender"]),
        ("Marital status", data["Married"]),
        ("Dependents", data["Dependents"]),
        ("Education", data["Education"]),
        ("Self-employed", data["Self_Employed"]),
        ("Property area", data["Property_Area"]),
        ("Applicant income", f"{data['ApplicantIncome']:,.0f} / month"),
        ("Co-applicant income", f"{data['CoapplicantIncome']:,.0f} / month"),
        ("Total income", f"{data['TotalIncome']:,.0f} / month"),
        ("Loan amount", f"{data['LoanAmount']:,.0f} (thousands)"),
        ("Loan term", f"{data['Loan_Amount_Term']:,.0f} months"),
        ("Credit history", "Has credit history" if data["Credit_History"] == 1.0 else "No credit history"),
    ]

    return render_template(
        "submit.html",
        approved=approved,
        confidence=confidence,
        ring_circumference=round(circumference, 2),
        ring_offset=ring_offset,
        summary_rows=summary_rows,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    # debug=True is fine for local development in VS Code;
    # set to False (or use a WSGI server like gunicorn/waitress) in production.
    app.run(debug=True, host="127.0.0.1", port=5000)
