"""
run_pipeline.py
Runs the complete Bank Marketing Predictor pipeline:
  1. Generate data  →  2. Preprocess + EDA  →  3. Train  →  4. Predict demo
"""

import subprocess, sys, os

STEPS = [
    ("generate_data.py", "Step 1/4 — Generating bank marketing dataset"),
    ("preprocess.py",    "Step 2/4 — Preprocessing & EDA"),
    ("train_model.py",   "Step 3/4 — Training Random Forest model"),
    ("predict.py",       "Step 4/4 — Demo predictions"),
]

def banner(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)

def run(script, label):
    banner(label)
    r = subprocess.run([sys.executable, script], capture_output=False)
    if r.returncode != 0:
        print(f"\n❌  {script} failed.")
        sys.exit(1)

def main():
    banner("Bank Marketing Predictor — Pipeline Starting")
    for d in ["data","models","outputs"]:
        os.makedirs(d, exist_ok=True)
    for script, label in STEPS:
        run(script, label)

    banner("Pipeline complete! 🎉")
    print("\nKey outputs:")
    print("  data/bank.csv                  ← raw dataset (1000 customers)")
    print("  data/bank_processed.csv        ← encoded features")
    print("  data/test_predictions.csv      ← model predictions on test set")
    print("  models/rf_model.pkl            ← trained Random Forest")
    print("  models/metrics.json            ← accuracy, AUC, F1, threshold")
    print("  outputs/1_eda_overview.png     ← 6-panel EDA chart")
    print("  outputs/2_correlations.png     ← feature correlations")
    print("  outputs/3_roc_curve.png        ← ROC-AUC curve")
    print("  outputs/4_precision_recall.png ← PR curve + threshold tuning")
    print("  outputs/5_confusion_matrix.png ← TP/TN/FP/FN")
    print("  outputs/6_feature_importance.png ← top 20 features")
    print("  outputs/7_cv_scores.png        ← cross-validation results")
    print("\nInteractive mode:")
    print("  python predict.py --interactive")

if __name__ == "__main__":
    main()
