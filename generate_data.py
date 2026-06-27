"""
generate_data.py
Generates a realistic synthetic Bank Marketing dataset and saves to data/bank.csv.
Mimics the UCI Bank Marketing dataset structure.
Run this first.
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)
N = 1000   # number of customers

# ── Helper samplers ───────────────────────────────────────────────────────────

def weighted_choice(choices, weights, n):
    return np.random.choice(choices, size=n, p=np.array(weights) / sum(weights))

# ── Feature generation ────────────────────────────────────────────────────────

age        = np.clip(np.random.normal(40, 12, N).astype(int), 18, 95)
job        = weighted_choice(
    ["admin.", "blue-collar", "technician", "services", "management",
     "retired", "entrepreneur", "self-employed", "housemaid", "student", "unemployed"],
    [18, 17, 15, 8, 9, 8, 5, 5, 4, 5, 6], N)
marital    = weighted_choice(["married", "single", "divorced"], [60, 30, 10], N)
education  = weighted_choice(
    ["basic.4y", "basic.6y", "basic.9y", "high.school", "university.degree", "professional.course"],
    [8, 6, 15, 25, 35, 11], N)
default    = weighted_choice(["no", "yes", "unknown"], [79, 2, 19], N)
housing    = weighted_choice(["yes", "no", "unknown"],  [52, 44, 4], N)
loan       = weighted_choice(["no", "yes", "unknown"],  [82, 15, 3], N)

# Contact info
contact    = weighted_choice(["cellular", "telephone"], [63, 37], N)
month      = weighted_choice(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    [1,2,4,5,25,5,15,13,5,5,10,10], N)
day_of_week = weighted_choice(["mon","tue","wed","thu","fri"], [20,20,20,20,20], N)
duration   = np.clip(np.random.exponential(250, N).astype(int), 0, 3600)

# Campaign
campaign   = np.clip(np.random.poisson(2.5, N), 1, 30)
pdays      = np.where(np.random.rand(N) < 0.15,
                      np.random.randint(1, 27, N), 999)
previous   = np.where(pdays < 999, np.random.randint(1, 5, N), 0)
poutcome   = np.where(pdays < 999,
                      weighted_choice(["success","failure","nonexistent"], [35,35,30], N),
                      "nonexistent")

# Socio-economic context
emp_var_rate  = np.round(np.random.choice([-3.4,-1.8,-1.7,-0.1,1.1,1.4], N,
                         p=[0.1,0.15,0.15,0.2,0.2,0.2]), 1)
cons_price_idx = np.round(92 + np.random.normal(1, 1.5, N), 3)
cons_conf_idx  = np.round(-42 + np.random.normal(0, 5, N), 1)
euribor3m      = np.round(np.clip(np.random.normal(3, 1.8, N), 0.6, 5.1), 3)
nr_employed    = np.round(np.random.choice([4963.6,5008.7,5017.5,5076.2,5099.1,5191.0], N), 1)

# ── Target: subscription (y) — realistic ~11% positive rate ──────────────────
# Build a score that increases with: longer duration, prior success, low euribor,
# student/retired jobs, higher education, cellular contact
score = (
      0.004  * duration
    + 0.5    * (poutcome == "success").astype(float)
    - 0.1    * euribor3m
    + 0.3    * np.isin(job, ["student","retired"]).astype(float)
    + 0.2    * np.isin(education, ["university.degree","professional.course"]).astype(float)
    + 0.15   * (contact == "cellular").astype(float)
    - 0.05   * campaign
    + np.random.normal(0, 0.5, N)
)
prob = 1 / (1 + np.exp(-score + 1.5))   # sigmoid, shifted for ~11% base rate
y    = (np.random.rand(N) < prob).astype(int)
y_label = np.where(y == 1, "yes", "no")

print(f"Subscription rate: {y.mean()*100:.1f}%  ({y.sum()} yes / {N-y.sum()} no)")

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "age": age, "job": job, "marital": marital, "education": education,
    "default": default, "housing": housing, "loan": loan,
    "contact": contact, "month": month, "day_of_week": day_of_week,
    "duration": duration, "campaign": campaign, "pdays": pdays,
    "previous": previous, "poutcome": poutcome,
    "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
    "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
    "nr.employed": nr_employed,
    "y": y_label,
})

df.to_csv("data/bank.csv", index=False)
print(f"Dataset saved → data/bank.csv  ({len(df)} rows, {df.shape[1]} columns)")
print("\nColumn overview:")
print(df.dtypes.to_string())
