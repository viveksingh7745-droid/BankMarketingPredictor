"""
predict.py
Predicts whether a bank customer will subscribe to a term deposit.
Usage:
    python predict.py               ← demo with 6 sample customers
    python predict.py --interactive ← enter customer details manually
"""

import pickle, json, sys
import pandas as pd
import numpy as np

MODEL_FILE    = "models/rf_model.pkl"
FEATURES_FILE = "models/feature_names.pkl"
METRICS_FILE  = "models/metrics.json"

EDU_ORDER = {
    "illiterate":0,"basic.4y":1,"basic.6y":2,"basic.9y":3,
    "high.school":4,"professional.course":5,"university.degree":6,"unknown":3,
}
MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
             "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
DOW_MAP   = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5}

ALL_JOBS = ["admin.","blue-collar","technician","services","management",
            "retired","entrepreneur","self-employed","housemaid","student","unemployed"]
ALL_MARITAL = ["married","single","divorced"]


def load_model():
    with open(MODEL_FILE,    "rb") as f: model    = pickle.load(f)
    with open(FEATURES_FILE, "rb") as f: features = pickle.load(f)
    with open(METRICS_FILE)        as f: metrics  = json.load(f)
    return model, features, metrics["best_threshold"]


def build_feature_row(c: dict, feature_names: list) -> pd.DataFrame:
    """Convert a raw customer dict into the encoded feature vector."""
    row = {}

    # Numeric
    row["age"]             = c["age"]
    row["duration"]        = c["duration"]
    row["campaign"]        = c["campaign"]
    row["previous"]        = c.get("previous", 0)
    row["pdays_clean"]     = c.get("pdays", 999) if c.get("pdays", 999) != 999 else -1
    row["emp.var.rate"]    = c.get("emp_var_rate", -1.8)
    row["cons.price.idx"]  = c.get("cons_price_idx", 93.2)
    row["cons.conf.idx"]   = c.get("cons_conf_idx", -42.0)
    row["euribor3m"]       = c.get("euribor3m", 1.3)
    row["nr.employed"]     = c.get("nr_employed", 5099.1)

    # Encoded
    row["education_enc"]    = EDU_ORDER.get(c.get("education","high.school"), 4)
    row["month_enc"]        = MONTH_MAP.get(c.get("month","may"), 5)
    row["dow_enc"]          = DOW_MAP.get(c.get("day_of_week","mon"), 1)
    row["default_flag"]     = 1 if c.get("default","no") == "yes" else 0
    row["housing_flag"]     = 1 if c.get("housing","yes") == "yes" else 0
    row["loan_flag"]        = 1 if c.get("loan","no")    == "yes" else 0
    row["contact_cell"]     = 1 if c.get("contact","cellular") == "cellular" else 0
    row["poutcome_success"] = 1 if c.get("poutcome","nonexistent") == "success" else 0
    row["prev_contacted"]   = 1 if c.get("pdays", 999) != 999 else 0

    # Job one-hot (drop_first → drop "admin.")
    for job in ALL_JOBS[1:]:
        row[f"job_{job}"] = 1 if c.get("job","admin.") == job else 0

    # Marital one-hot (drop_first → drop "divorced")
    for mar in ALL_MARITAL[1:]:
        row[f"marital_{mar}"] = 1 if c.get("marital","married") == mar else 0

    # Build df aligned to training feature order
    df_row = pd.DataFrame([row])
    for col in feature_names:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_names]
    return df_row


def predict_customer(model, feature_names, threshold, customer: dict):
    df_row = build_feature_row(customer, feature_names)
    prob   = model.predict_proba(df_row)[0][1]
    pred   = int(prob >= threshold)
    return pred, prob


def print_result(customer, pred, prob):
    label   = "✅ YES — likely to subscribe" if pred == 1 else "❌ NO  — unlikely to subscribe"
    bar_len = int(prob * 30)
    bar     = "█" * bar_len + "░" * (30 - bar_len)
    risk    = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.35 else "LOW"

    print(f"\n  Customer : {customer.get('name','—')}")
    print(f"  Age      : {customer['age']}  |  Job: {customer.get('job','—')}  |  Duration: {customer['duration']}s")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Prediction  : {label}")
    print(f"  Probability : {prob*100:.1f}%  [{bar}]")
    print(f"  Lead score  : {risk}")


# ── Demo customers ─────────────────────────────────────────────────────────────
DEMO_CUSTOMERS = [
    {"name": "Amit Sharma",   "age":55, "job":"retired",      "marital":"married",
     "education":"university.degree", "default":"no", "housing":"no",  "loan":"no",
     "contact":"cellular", "month":"oct", "day_of_week":"thu",
     "duration":520, "campaign":1, "pdays":10, "previous":2, "poutcome":"success",
     "emp_var_rate":-3.4, "cons_price_idx":92.4, "cons_conf_idx":-26.9, "euribor3m":0.7},

    {"name": "Priya Nair",    "age":32, "job":"admin.",        "marital":"single",
     "education":"high.school", "default":"no", "housing":"yes", "loan":"no",
     "contact":"cellular", "month":"may", "day_of_week":"mon",
     "duration":120, "campaign":5, "pdays":999, "previous":0, "poutcome":"nonexistent",
     "emp_var_rate":1.1, "cons_price_idx":93.9, "cons_conf_idx":-46.2, "euribor3m":4.9},

    {"name": "Rahul Verma",   "age":45, "job":"management",   "marital":"married",
     "education":"university.degree", "default":"no", "housing":"yes", "loan":"no",
     "contact":"cellular", "month":"apr", "day_of_week":"tue",
     "duration":380, "campaign":2, "pdays":999, "previous":0, "poutcome":"nonexistent",
     "emp_var_rate":-1.8, "cons_price_idx":93.0, "cons_conf_idx":-40.0, "euribor3m":1.3},

    {"name": "Sneha Iyer",    "age":24, "job":"student",      "marital":"single",
     "education":"university.degree", "default":"no", "housing":"no",  "loan":"no",
     "contact":"cellular", "month":"oct", "day_of_week":"wed",
     "duration":640, "campaign":1, "pdays":5, "previous":1, "poutcome":"success",
     "emp_var_rate":-3.4, "cons_price_idx":92.1, "cons_conf_idx":-30.0, "euribor3m":0.8},

    {"name": "Kiran Reddy",   "age":61, "job":"blue-collar",  "marital":"divorced",
     "education":"basic.9y", "default":"unknown", "housing":"yes", "loan":"yes",
     "contact":"telephone", "month":"may", "day_of_week":"fri",
     "duration":80,  "campaign":9, "pdays":999, "previous":0, "poutcome":"nonexistent",
     "emp_var_rate":1.4, "cons_price_idx":94.4, "cons_conf_idx":-50.8, "euribor3m":5.0},

    {"name": "Deepa Menon",   "age":38, "job":"technician",   "marital":"married",
     "education":"professional.course", "default":"no", "housing":"no", "loan":"no",
     "contact":"cellular", "month":"sep", "day_of_week":"thu",
     "duration":290, "campaign":2, "pdays":15, "previous":1, "poutcome":"failure",
     "emp_var_rate":-1.7, "cons_price_idx":92.6, "cons_conf_idx":-38.3, "euribor3m":1.1},
]


def run_demo(model, features, threshold):
    print("\n" + "="*58)
    print("  BANK MARKETING PREDICTOR — Demo")
    print(f"  Decision threshold: {threshold:.2f}")
    print("="*58)
    for c in DEMO_CUSTOMERS:
        pred, prob = predict_customer(model, features, threshold, c)
        print_result(c, pred, prob)
    print("\n" + "="*58)


def run_interactive(model, features, threshold):
    print("\n" + "="*58)
    print("  BANK MARKETING PREDICTOR — Interactive Mode")
    print("  Type 'quit' to exit")
    print("="*58)
    while True:
        print()
        try:
            name     = input("  Customer name        : ").strip()
            if name.lower() in ("quit","exit","q"): break
            age      = int(input("  Age                  : "))
            job      = input(f"  Job {ALL_JOBS} : ").strip()
            duration = int(input("  Last call duration (s): "))
            campaign = int(input("  Number of contacts   : "))
            housing  = input("  Housing loan? (yes/no): ").strip().lower()
            loan     = input("  Personal loan? (yes/no): ").strip().lower()
            contact  = input("  Contact type (cellular/telephone): ").strip().lower()

            c = {"name": name, "age": age, "job": job, "duration": duration,
                 "campaign": campaign, "housing": housing, "loan": loan,
                 "contact": contact, "pdays": 999, "previous": 0,
                 "poutcome": "nonexistent"}
            pred, prob = predict_customer(model, features, threshold, c)
            print_result(c, pred, prob)
        except (ValueError, KeyboardInterrupt):
            print("  Invalid input, try again.")


def main():
    model, features, threshold = load_model()
    if "--interactive" in sys.argv:
        run_interactive(model, features, threshold)
    else:
        run_demo(model, features, threshold)
        print("\nTip: Run  python predict.py --interactive  for manual entry!")


if __name__ == "__main__":
    main()
