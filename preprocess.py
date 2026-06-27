"""
preprocess.py
Cleans the bank marketing dataset, runs EDA, and prepares features for ML.
Outputs: data/bank_processed.csv  +  outputs/1_eda_*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

os.makedirs("outputs", exist_ok=True)
RAW_FILE  = "data/bank.csv"
OUT_FILE  = "data/bank_processed.csv"

# Ordinal encoding for education
EDU_ORDER = {
    "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
    "high.school": 4, "professional.course": 5, "university.degree": 6, "unknown": 3,
}
MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
             "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
DOW_MAP   = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5}


def load_and_inspect(path):
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print(f"\nTarget distribution:\n{df['y'].value_counts()}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary target
    df["y_bin"] = (df["y"] == "yes").astype(int)

    # Ordinal
    df["education_enc"] = df["education"].map(EDU_ORDER).fillna(3).astype(int)
    df["month_enc"]     = df["month"].map(MONTH_MAP)
    df["dow_enc"]       = df["day_of_week"].map(DOW_MAP)

    # Binary flags
    for col in ["default", "housing", "loan"]:
        df[f"{col}_flag"] = (df[col] == "yes").astype(int)

    df["contact_cell"]     = (df["contact"]  == "cellular").astype(int)
    df["poutcome_success"] = (df["poutcome"] == "success").astype(int)
    df["prev_contacted"]   = (df["pdays"]    != 999).astype(int)
    df["pdays_clean"]      = df["pdays"].replace(999, -1)   # -1 = never contacted

    # One-hot encode job & marital
    job_dummies = pd.get_dummies(df["job"],     prefix="job",     drop_first=True)
    mar_dummies = pd.get_dummies(df["marital"], prefix="marital", drop_first=True)
    df = pd.concat([df, job_dummies, mar_dummies], axis=1)

    # Drop original categorical columns
    drop_cols = ["y","job","marital","education","default","housing","loan",
                 "contact","month","day_of_week","poutcome","pdays"]
    df = df.drop(columns=drop_cols)
    return df


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_overview(df_raw):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#F8F8F8")
    fig.suptitle("Bank Marketing EDA — Customer Overview", fontsize=15, fontweight="bold")

    colors = ["#1565C0", "#E53935"]

    # 1. Target distribution
    ax = axes[0, 0]
    counts = df_raw["y"].value_counts()
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Subscription Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_facecolor("white")
    for i, (label, val) in enumerate(counts.items()):
        ax.text(i, val + 5, f"{val}\n({val/len(df_raw)*100:.1f}%)",
                ha="center", fontsize=10, fontweight="bold")

    # 2. Age distribution by outcome
    ax = axes[0, 1]
    for label, color in zip(["no","yes"], colors):
        sub = df_raw[df_raw["y"] == label]["age"]
        ax.hist(sub, bins=20, alpha=0.6, label=label, color=color)
    ax.set_title("Age Distribution by Outcome", fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_facecolor("white")

    # 3. Job vs subscription rate
    ax = axes[0, 2]
    job_rate = df_raw.groupby("job")["y"].apply(lambda x: (x=="yes").mean() * 100).sort_values()
    ax.barh(job_rate.index, job_rate.values, color="#1565C0", alpha=0.8)
    ax.set_title("Subscription Rate by Job (%)", fontweight="bold")
    ax.set_xlabel("Subscription Rate (%)")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8)

    # 4. Call duration by outcome
    ax = axes[1, 0]
    for label, color in zip(["no","yes"], colors):
        sub = df_raw[df_raw["y"] == label]["duration"]
        ax.hist(sub, bins=30, alpha=0.6, label=label, color=color)
    ax.set_title("Call Duration by Outcome", fontweight="bold")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_facecolor("white")

    # 5. Month vs subscription rate
    ax = axes[1, 1]
    month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    month_rate  = df_raw.groupby("month")["y"].apply(lambda x: (x=="yes").mean()*100)
    month_rate  = month_rate.reindex([m for m in month_order if m in month_rate.index])
    ax.bar(month_rate.index, month_rate.values, color="#1565C0", alpha=0.8)
    ax.set_title("Subscription Rate by Month (%)", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rate (%)")
    ax.set_facecolor("white")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # 6. Education vs subscription rate
    ax = axes[1, 2]
    edu_order = ["basic.4y","basic.6y","basic.9y","high.school","professional.course","university.degree"]
    edu_rate  = df_raw.groupby("education")["y"].apply(lambda x: (x=="yes").mean()*100)
    edu_rate  = edu_rate.reindex([e for e in edu_order if e in edu_rate.index])
    ax.barh(edu_rate.index, edu_rate.values, color="#E53935", alpha=0.8)
    ax.set_title("Subscription Rate by Education (%)", fontweight="bold")
    ax.set_xlabel("Rate (%)")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig("outputs/1_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/1_eda_overview.png")


def plot_correlations(df_proc):
    num_cols = df_proc.select_dtypes(include=np.number).columns.tolist()
    corr = df_proc[num_cols].corr()["y_bin"].drop("y_bin").sort_values()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#F8F8F8")
    colors = ["#E53935" if v < 0 else "#1565C0" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Correlation with Subscription (y)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig("outputs/2_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/2_correlations.png")


def main():
    df_raw  = load_and_inspect(RAW_FILE)
    plot_overview(df_raw)

    df_proc = encode_features(df_raw)
    plot_correlations(df_proc)

    df_proc.to_csv(OUT_FILE, index=False)
    print(f"\nProcessed data saved → {OUT_FILE}")
    print(f"Shape after encoding: {df_proc.shape}")
    print(f"Features: {df_proc.shape[1]-1}  |  Target: y_bin")


if __name__ == "__main__":
    main()
