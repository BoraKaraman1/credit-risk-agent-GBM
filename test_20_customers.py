"""
Standalone test: score 20 synthetic customers using the champion model.
Bypasses the API/Supabase layer — loads the model directly.
Prints a results table with score, decision, and top adverse action reasons.
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = DATA_DIR / "models"

APPROVE_THRESHOLD = 0.15
REVIEW_THRESHOLD  = 0.30
NUM_ADVERSE       = 3

# Human-readable names (matches scoring_service.py)
FEATURE_DISPLAY = {
    "loan_amnt": "Loan Amount", "term": "Loan Term", "int_rate": "Interest Rate",
    "installment": "Monthly Installment", "emp_length": "Employment Length",
    "home_ownership": "Home Ownership", "annual_inc": "Annual Income",
    "verification_status": "Income Verification", "purpose": "Loan Purpose",
    "dti": "Debt-to-Income Ratio", "delinq_2yrs": "Delinquencies (2yr)",
    "inq_last_6mths": "Credit Inquiries (6mo)", "mths_since_last_delinq": "Months Since Delinquency",
    "open_acc": "Open Accounts", "pub_rec": "Public Records",
    "revol_bal": "Revolving Balance", "revol_util": "Revolving Utilization",
    "total_acc": "Total Accounts", "mort_acc": "Mortgage Accounts",
    "pub_rec_bankruptcies": "Bankruptcies", "credit_history_months": "Credit History (mo)",
    "fico_score": "FICO Score", "emp_length_missing": "Emp Length Not Reported",
    "log_annual_inc": "Log Annual Income", "loan_to_income": "Loan-to-Income",
    "installment_to_income": "Installment-to-Income", "dti_x_income": "Abs Debt Burden ($k)",
    "grade_numeric": "Credit Grade", "delinq_ever": "Ever Delinquent",
    "high_utilization": "High Utilization Flag", "has_mortgage": "Has Mortgage",
    "has_bankruptcy": "Has Bankruptcy", "sub_grade_numeric": "Sub-Grade",
}

# Categorical encodings used during Gold feature engineering
# home_ownership: MORTGAGE=0, OTHER=1, OWN=2, RENT=3
# verification_status: Not Verified=0, Source Verified=1, Verified=2
# purpose (alphabetical label encoding from Lending Club categories):
#   car=0, credit_card=1, debt_consolidation=2, educational=3, home_improvement=4,
#   house=5, major_purchase=6, medical=7, moving=8, other=9,
#   renewable_energy=10, small_business=11, vacation=12, wedding=13
# grade: A=1 B=2 C=3 D=4 E=5 F=6 G=7
# sub_grade: A1=1 ... G5=35

def make_features(loan_amnt, term, int_rate, installment, grade, sub_grade,
                  emp_length, home_ownership_code, annual_inc,
                  verification_status_code, purpose_code,
                  dti, delinq_2yrs, fico_score, inq_last_6mths,
                  mths_since_last_delinq, open_acc, pub_rec, revol_bal,
                  revol_util, total_acc, mort_acc, pub_rec_bankruptcies,
                  credit_history_months, emp_length_missing=0):
    """Build the full 33-feature Gold vector from human-friendly inputs."""
    grade_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
    g = grade_map.get(grade, grade) if isinstance(grade, str) else grade

    monthly_inc = annual_inc / 12
    log_annual_inc   = np.log1p(annual_inc)
    loan_to_income   = loan_amnt / annual_inc if annual_inc > 0 else 0
    inst_to_income   = installment / monthly_inc if monthly_inc > 0 else 0
    dti_x_income     = dti * annual_inc / 1000
    delinq_ever      = int(delinq_2yrs > 0)
    high_util        = int(revol_util > 75)
    has_mortgage     = int(mort_acc > 0)
    has_bankruptcy   = int(pub_rec_bankruptcies > 0)

    return {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "emp_length": emp_length,
        "home_ownership": home_ownership_code, "annual_inc": annual_inc,
        "verification_status": verification_status_code, "purpose": purpose_code,
        "dti": dti, "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths,
        "mths_since_last_delinq": mths_since_last_delinq, "open_acc": open_acc,
        "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util,
        "total_acc": total_acc, "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "credit_history_months": credit_history_months,
        "fico_score": fico_score, "emp_length_missing": emp_length_missing,
        "log_annual_inc": log_annual_inc, "loan_to_income": loan_to_income,
        "installment_to_income": inst_to_income, "dti_x_income": dti_x_income,
        "grade_numeric": g, "delinq_ever": delinq_ever,
        "high_utilization": high_util, "has_mortgage": has_mortgage,
        "has_bankruptcy": has_bankruptcy,
        "sub_grade_numeric": sub_grade,
    }


# ---------- 20 synthetic customers ----------
# Profiles range from very safe (prime) to very risky (subprime).
# home_ownership: 0=MORTGAGE, 2=OWN, 3=RENT, 1=OTHER
# verification_status: 2=Verified, 1=Source Verified, 0=Not Verified
# purpose: 2=debt_consolidation, 1=credit_card, 4=home_improvement, 11=small_business, 9=other

CUSTOMERS = [
    # --- PRIME (very safe) ---
    ("C01 - Prime mortgage holder",
     make_features(15000,36,6.5,460,"A",3,10,0,95000,2,4,8.0,0,780,0,999,8,0,2000,12.0,22,2,0,180)),
    ("C02 - Senior professional, low DTI",
     make_features(10000,36,7.2,310,"A",5,10,0,120000,2,1,5.5,0,795,0,999,10,0,5000,8.0,28,3,0,240)),
    ("C03 - Homeowner, verified, long history",
     make_features(20000,60,8.9,415,"B",8,8,2,85000,2,2,11.0,0,760,1,999,9,0,8000,18.0,19,1,0,200)),
    ("C04 - Mid-income, clean record",
     make_features(12000,36,9.5,385,"B",10,5,3,65000,1,2,13.0,0,745,0,999,7,0,3500,22.0,14,0,0,150)),
    ("C05 - Young professional, low utilization",
     make_features(8000,36,10.2,260,"B",11,3,3,58000,2,1,10.5,0,735,1,999,6,0,1200,15.0,11,0,0,96)),

    # --- NEAR-PRIME (likely approve / edge cases) ---
    ("C06 - Moderate DTI, some inquiries",
     make_features(18000,60,13.5,410,"C",13,7,3,72000,1,2,18.0,0,710,2,24,8,0,9000,35.0,16,1,0,132)),
    ("C07 - Rentor, verified income",
     make_features(14000,36,12.0,467,"C",14,6,3,60000,2,1,16.0,0,715,1,36,7,0,4000,28.0,13,0,0,120)),
    ("C08 - Self-employed, higher rate",
     make_features(22000,60,14.8,512,"C",15,4,3,78000,0,11,20.0,1,700,2,18,9,0,15000,42.0,18,0,0,108,1)),
    ("C09 - Home improvement loan, good FICO",
     make_features(25000,60,11.5,540,"B",9,9,0,90000,1,4,14.0,0,742,0,999,11,0,12000,25.0,21,2,0,168)),
    ("C10 - Minor delinquency 3yr ago",
     make_features(16000,60,15.2,380,"C",16,5,3,55000,1,2,19.5,1,695,1,36,8,0,7000,48.0,15,0,0,140)),

    # --- SUBPRIME (likely review or decline) ---
    ("C11 - High utilization, frequent inquiries",
     make_features(20000,60,19.5,510,"D",18,3,3,48000,0,1,26.0,2,665,4,12,10,1,22000,82.0,20,0,0,90)),
    ("C12 - High DTI, unverified income",
     make_features(30000,60,21.0,710,"D",20,2,3,52000,0,2,30.5,1,650,3,24,7,0,18000,78.0,14,0,0,84)),
    ("C13 - Recent delinquency, thin file",
     make_features(10000,36,22.5,330,"E",22,1,3,38000,0,9,24.0,2,635,5,6,5,1,8000,88.0,10,0,0,60)),
    ("C14 - Small business, high risk",
     make_features(35000,60,20.2,810,"D",19,4,3,61000,0,11,28.0,1,640,2,18,8,1,28000,72.0,16,0,0,96)),
    ("C15 - Public record, high rate",
     make_features(15000,36,24.0,500,"E",24,2,3,42000,0,2,22.0,3,625,4,9,6,2,5000,90.0,12,1,0,72)),

    # --- HIGH RISK (likely decline) ---
    ("C16 - Near-maximum DTI, bankrupty history",
     make_features(25000,60,26.0,590,"F",27,1,3,45000,0,2,38.0,3,595,5,3,7,2,12000,93.0,15,0,1,60)),
    ("C17 - Employment unknown, many delinquencies",
     make_features(18000,60,25.5,430,"E",23,0,3,39000,0,9,32.0,4,610,6,3,8,1,9000,85.0,13,0,0,48,1)),
    ("C18 - Very high utilization, thin file",
     make_features(12000,36,28.5,410,"F",28,0,3,35000,0,1,35.0,5,580,7,1,4,3,4000,97.0,9,0,0,36,1)),
    ("C19 - Multiple public records, low FICO",
     make_features(20000,60,29.0,480,"G",31,0,1,40000,0,9,40.0,4,570,8,2,5,4,6000,95.0,11,0,1,48)),
    ("C20 - Worst profile: all risk factors present",
     make_features(30000,60,30.5,720,"G",35,0,3,32000,0,2,45.0,6,540,9,1,6,5,3000,99.0,10,0,2,36,1)),
]


def load_model_and_explainer():
    champion_dir = MODELS_DIR / "champion"
    model_path = champion_dir / "model.joblib"
    if not model_path.exists():
        model_path = champion_dir / "model.pkl"

    with open(champion_dir / "model_metadata.json") as f:
        meta = json.load(f)

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    return model, explainer, meta["features"], meta["version"]


def score_all(customers, model, explainer, feature_cols):
    rows = []
    for name, feat_dict in customers:
        X = np.array([[feat_dict.get(c, 0.0) or 0.0 for c in feature_cols]])

        score = float(model.predict_proba(X)[:, 1][0])
        if score < APPROVE_THRESHOLD:
            decision = "APPROVE"
        elif score < REVIEW_THRESHOLD:
            decision = "REVIEW"
        else:
            decision = "DECLINE"

        # Adverse actions (SHAP) for non-approvals
        reasons = []
        if decision != "APPROVE":
            shap_vals = explainer.shap_values(X)[0]
            top_idx = np.argsort(-shap_vals)
            for idx in top_idx:
                if shap_vals[idx] <= 0 or len(reasons) >= NUM_ADVERSE:
                    break
                reasons.append(
                    f"{FEATURE_DISPLAY.get(feature_cols[idx], feature_cols[idx])} "
                    f"({feat_dict.get(feature_cols[idx], 0):.1f}, SHAP={shap_vals[idx]:+.3f})"
                )

        rows.append({
            "Customer": name,
            "Score": score,
            "Decision": decision,
            "FICO": feat_dict["fico_score"],
            "DTI": feat_dict["dti"],
            "RevUtil%": feat_dict["revol_util"],
            "AnnInc": feat_dict["annual_inc"],
            "Reasons": " | ".join(reasons) if reasons else "—",
        })
    return rows


def print_results(rows, model_version):
    DECISION_COLOR = {"APPROVE": "\033[92m", "REVIEW": "\033[93m", "DECLINE": "\033[91m"}
    RESET = "\033[0m"

    print(f"\n{'='*110}")
    print(f"  CREDIT RISK SCORING — 20 Customers   [Model: {model_version}]")
    print(f"  Thresholds: APPROVE < {APPROVE_THRESHOLD} | REVIEW {APPROVE_THRESHOLD}–{REVIEW_THRESHOLD} | DECLINE ≥ {REVIEW_THRESHOLD}")
    print(f"{'='*110}")
    print(f"{'#':<3} {'Customer':<40} {'Score':>6} {'Decision':>8}  {'FICO':>4} {'DTI%':>5} {'Util%':>5} {'Income':>8}")
    print(f"{'-'*110}")

    summary = {"APPROVE": 0, "REVIEW": 0, "DECLINE": 0}
    for i, r in enumerate(rows, 1):
        color = DECISION_COLOR.get(r["Decision"], "")
        print(f"{i:<3} {r['Customer']:<40} {r['Score']:>6.3f} "
              f"{color}{r['Decision']:>8}{RESET}  "
              f"{r['FICO']:>4} {r['DTI']:>5.1f} {r['RevUtil%']:>5.1f} {r['AnnInc']:>8,.0f}")
        if r["Reasons"] != "—":
            print(f"    └─ Reasons: {r['Reasons']}")
        summary[r["Decision"]] += 1

    print(f"{'='*110}")
    print(f"  Summary: {summary['APPROVE']} APPROVED  |  {summary['REVIEW']} REVIEW  |  {summary['DECLINE']} DECLINED")
    print(f"{'='*110}\n")


if __name__ == "__main__":
    print("Loading champion model and SHAP explainer...")
    model, explainer, feature_cols, version = load_model_and_explainer()
    print(f"Model {version} loaded ({len(feature_cols)} features)\n")

    print("Scoring 20 customers...")
    results = score_all(CUSTOMERS, model, explainer, feature_cols)
    print_results(results, version)
