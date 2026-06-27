"""
train_model.py
Trains a Random Forest classifier to predict bank term deposit subscriptions.
Handles class imbalance, tunes threshold, saves model + full evaluation charts.
"""

import pandas as pd
import numpy as np
import pickle, json, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

PROC_FILE    = "data/bank_processed.csv"
MODEL_FILE   = "models/rf_model.pkl"
FEATURES_FILE = "models/feature_names.pkl"
METRICS_FILE = "models/metrics.json"


def load_data(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "y_bin"]
    X = df[feature_cols]
    y = df["y_bin"]
    print(f"Loaded {len(df)} rows | Features: {len(feature_cols)} | Positive rate: {y.mean()*100:.1f}%")
    return X, y, feature_cols


def plot_roc_curve(y_test, y_prob, path="outputs/3_roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#F8F8F8")
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1565C0")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — Random Forest", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")
    return auc


def plot_precision_recall(y_test, y_prob, path="outputs/4_precision_recall.png"):
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#F8F8F8")

    # PR curve
    ax = axes[0]
    ax.plot(rec, prec, color="#E53935", lw=2, label=f"AP = {ap:.4f}")
    ax.fill_between(rec, prec, alpha=0.1, color="#E53935")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend()
    ax.set_facecolor("white")

    # Precision & Recall vs Threshold
    ax = axes[1]
    ax.plot(thresholds, prec[:-1], color="#1565C0", lw=2, label="Precision")
    ax.plot(thresholds, rec[:-1],  color="#E53935", lw=2, label="Recall")
    # Find best F1 threshold
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = f1s.argmax()
    best_thresh = thresholds[best_idx]
    ax.axvline(best_thresh, color="green", linestyle="--", lw=1.5,
               label=f"Best threshold = {best_thresh:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Threshold", fontweight="bold")
    ax.legend()
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")
    return best_thresh


def plot_confusion_matrix(cm_arr, threshold, path="outputs/5_confusion_matrix.png"):
    labels = ["No (not subscribed)", "Yes (subscribed)"]
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#F8F8F8")
    im = ax.imshow(cm_arr, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    thresh = cm_arr.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i,j]}",
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm_arr[i,j] > thresh else "black")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_feature_importance(model, feature_names, top_n=20, path="outputs/6_feature_importance.png"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#F8F8F8")
    colors = cm.Blues_r(np.linspace(0.2, 0.8, len(top)))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], alpha=0.9)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Gini Importance")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_cv_scores(cv_results, path="outputs/7_cv_scores.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#F8F8F8")
    metrics = [
        ("test_accuracy", "Accuracy",  "#1565C0"),
        ("test_roc_auc",  "ROC-AUC",   "#E53935"),
        ("test_f1",       "F1 Score",  "#2E7D32"),
    ]
    for ax, (key, label, color) in zip(axes, metrics):
        scores = cv_results[key]
        folds  = [f"Fold {i+1}" for i in range(len(scores))]
        bars   = ax.bar(folds, scores, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(scores.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean={scores.mean():.3f}")
        ax.set_ylim(0, 1.1)
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor("white")
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("5-Fold Stratified Cross-Validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def main():
    X, y, feature_names = load_data(PROC_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Train positives: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    # ── Random Forest with class_weight to handle imbalance ──────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",   # handles imbalanced classes
        random_state=42,
        n_jobs=-1,
    )

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\nRunning 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "roc_auc", "f1"],
        return_train_score=False,
    )
    print(f"  CV Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  CV ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"  CV F1       : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    plot_cv_scores(cv_results)

    # ── Train final model ─────────────────────────────────────────────────────
    print("\nTraining final Random Forest ...")
    model.fit(X_train, y_train)

    # ── Probability predictions ───────────────────────────────────────────────
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    # ── ROC & best threshold ──────────────────────────────────────────────────
    auc        = plot_roc_curve(y_test, y_prob_test)
    best_thresh = plot_precision_recall(y_test, y_prob_test)

    # ── Predictions at best threshold ─────────────────────────────────────────
    y_pred = (y_prob_test >= best_thresh).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    print(f"\n{'='*55}")
    print(f"  Test Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  Best Threshold  : {best_thresh:.4f}")
    print(f"{'='*55}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No','Yes'])}")

    cm_arr = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm_arr, best_thresh)
    plot_feature_importance(model, feature_names)

    # ── Save model & metrics ──────────────────────────────────────────────────
    with open(MODEL_FILE,    "wb") as f: pickle.dump(model, f)
    with open(FEATURES_FILE, "wb") as f: pickle.dump(feature_names, f)

    tn, fp, fn, tp = cm_arr.ravel()
    metrics = {
        "test_accuracy":   round(acc, 4),
        "roc_auc":         round(auc, 4),
        "f1_score":        round(f1, 4),
        "best_threshold":  round(float(best_thresh), 4),
        "cv_accuracy_mean": round(float(cv_results["test_accuracy"].mean()), 4),
        "cv_roc_auc_mean":  round(float(cv_results["test_roc_auc"].mean()),  4),
        "cv_f1_mean":       round(float(cv_results["test_f1"].mean()),       4),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "n_features": len(feature_names),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved   → {MODEL_FILE}")
    print(f"Metrics saved → {METRICS_FILE}")

    # Save test predictions
    test_df = X_test.copy()
    test_df["actual"]    = y_test.values
    test_df["predicted"] = y_pred
    test_df["probability"] = y_prob_test.round(4)
    test_df.to_csv("data/test_predictions.csv", index=False)
    print(f"Test predictions → data/test_predictions.csv")


if __name__ == "__main__":
    main()
