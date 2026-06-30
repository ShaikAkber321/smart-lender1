"""
generate_dataset.py
--------------------
Generates a realistic synthetic loan-applicant dataset that mirrors the
classic bank loan-eligibility schema (Gender, Married, Dependents, Education,
Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status).

The data is generated with deliberate real-world messiness so that the
preprocessing stage in train_model.py (missing-value handling, outlier
handling, encoding, SMOTE balancing, scaling) has real work to do:

  * Missing values scattered across several columns (MCAR)
  * A small fraction of extreme outliers in the income / loan columns
  * Class imbalance in the target (more approvals than rejections)
  * A genuine, learnable signal (credit history, income-to-loan ratio,
    education, employment type) so the trained models are meaningful
    rather than memorizing noise.

Run directly to (re)create data/loan_data.csv:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 900


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.80, 0.20])
    married = rng.choice(["Yes", "No"], size=n_rows, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_rows, p=[0.50, 0.15, 0.20, 0.15])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_rows, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], size=n_rows, p=[0.85, 0.15])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_rows, p=[0.35, 0.38, 0.27])

    # Income: lognormal so most applicants cluster low-mid, with a long right tail
    applicant_income = rng.lognormal(mean=8.35, sigma=0.45, size=n_rows).round(0)
    applicant_income = np.clip(applicant_income, 1500, None)

    # ~40% of applicants have no co-applicant (single income)
    has_coapplicant = rng.random(n_rows) > 0.40
    coapplicant_income = np.where(
        has_coapplicant,
        rng.lognormal(mean=7.6, sigma=0.55, size=n_rows).round(0),
        0,
    )

    # Loan amount roughly scales with combined income, plus noise
    combined_income = applicant_income + coapplicant_income
    loan_amount = (combined_income * rng.uniform(0.05, 0.18, size=n_rows) / 1000 * 100).round(0)
    loan_amount = np.clip(loan_amount, 9, None)

    loan_term = rng.choice(
        [12, 36, 60, 84, 120, 180, 240, 300, 360],
        size=n_rows,
        p=[0.02, 0.03, 0.03, 0.02, 0.05, 0.07, 0.08, 0.10, 0.60],
    )

    credit_history = rng.choice([1.0, 0.0], size=n_rows, p=[0.84, 0.16])

    # ---- Target generation: a real, learnable signal + noise ----
    # Credit history is, by far, the dominant predictor of loan approval in
    # real-world bank data (this mirrors the well-known public loan-eligibility
    # dataset, where credit_history alone predicts ~80% of outcomes correctly).
    income_to_loan = combined_income / (loan_amount * 1000 + 1)
    z = (
        2.2 * (credit_history - 0.5) * 2  # strong but not perfectly deterministic
        + 0.40 * (education == "Graduate")
        - 0.30 * (self_employed == "Yes")
        + 0.25 * (property_area == "Semiurban")
        - 0.10 * (property_area == "Rural")
        + 0.70 * np.tanh(income_to_loan * 3 - 1)
        - 0.12 * (dependents == "3+")
        + rng.normal(0, 0.85, size=n_rows)  # irreducible noise -> realistic (not perfect) accuracy
        - 0.15
    )
    prob_approved = 1 / (1 + np.exp(-z))
    loan_status = np.where(rng.random(n_rows) < prob_approved, "Y", "N")

    df = pd.DataFrame(
        {
            "Loan_ID": [f"LP{1000 + i}" for i in range(n_rows)],
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income.astype(int),
            "CoapplicantIncome": coapplicant_income.astype(int),
            "LoanAmount": loan_amount.astype(int),
            "Loan_Amount_Term": loan_term.astype(int),
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )

    # ---- Inject realistic outliers (≈2% of rows) ----
    outlier_idx = rng.choice(n_rows, size=max(1, int(0.02 * n_rows)), replace=False)
    df.loc[outlier_idx, "ApplicantIncome"] = rng.integers(45000, 81000, size=len(outlier_idx))
    outlier_idx2 = rng.choice(n_rows, size=max(1, int(0.015 * n_rows)), replace=False)
    df.loc[outlier_idx2, "LoanAmount"] = rng.integers(600, 900, size=len(outlier_idx2))

    # ---- Inject missing values (MCAR) on several columns ----
    def inject_missing(col, frac):
        idx = rng.choice(n_rows, size=int(frac * n_rows), replace=False)
        df.loc[idx, col] = np.nan

    inject_missing("Gender", 0.03)
    inject_missing("Married", 0.02)
    inject_missing("Dependents", 0.025)
    inject_missing("Self_Employed", 0.05)
    inject_missing("LoanAmount", 0.035)
    inject_missing("Loan_Amount_Term", 0.02)
    inject_missing("Credit_History", 0.06)

    return df


if __name__ == "__main__":
    from pathlib import Path

    out_path = Path(__file__).resolve().parent / "loan_data.csv"
    data = generate()
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data)} rows -> {out_path}")
    print(data.isna().sum())
    print("\nClass balance:\n", data["Loan_Status"].value_counts())
