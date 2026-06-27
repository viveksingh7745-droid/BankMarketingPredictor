# Bank Marketing Predictor — CodTech Internship Project. Internship ID-CITS3857

Predicts whether a bank customer will subscribe to a **term deposit**
using a **Random Forest** classifier trained on telemarketing campaign data.
Mimics the real UCI Bank Marketing dataset — no download required.

## Problem Statement
Banks run telemarketing campaigns to sell term deposits. Calling every customer
is expensive. This model predicts which customers are likely to say **yes**,
letting the bank focus resources on high-probability leads.

## Project Structure
```
bank_marketing/
├── generate_data.py       ← creates 1000-customer synthetic dataset
├── preprocess.py          ← cleans, encodes features, EDA charts
├── train_model.py         ← Random Forest + threshold tuning + evaluation
├── predict.py             ← predict for new customers (demo + interactive)
├── run_pipeline.py        ← runs all 4 steps in one command
├── requirements.txt
├── data/                  ← generated CSVs
├── models/                ← saved model + features + metrics
└── outputs/               ← 7 PNG charts
```

## Setup & Run
```bash
pip install -r requirements.txt
python run_pipeline.py

# Interactive mode — enter your own customer details
python predict.py --interactive
```

## Features Used (20 input variables)
| Category | Features |
|---|---|
| Customer info | age, job, marital status, education, default, housing loan, personal loan |
| Campaign | contact type, month, day of week, call duration, no. of contacts, previous outcome |
| Socio-economic | employment variation rate, consumer price index, consumer confidence, euribor 3m, no. employed |

## Model — Random Forest
- 200 trees, max depth 12, balanced class weights (handles imbalance)
- Threshold tuned on Precision-Recall curve for best F1
- 5-fold stratified cross-validation

## Output Charts
| File | Description |
|---|---|
| `1_eda_overview.png` | Subscription rate, age, job, month, education, call duration |
| `2_correlations.png` | Feature correlation with target variable |
| `3_roc_curve.png` | ROC-AUC curve |
| `4_precision_recall.png` | PR curve + optimal threshold selection |
| `5_confusion_matrix.png` | TP, TN, FP, FN at best threshold |
| `6_feature_importance.png` | Top 20 most influential features |
| `7_cv_scores.png` | Cross-validation accuracy, AUC, F1 per fold |

## Key Metrics (typical results)
- ROC-AUC: ~0.80+
- CV Accuracy: ~85%+
- The model correctly identifies high-potential leads saving campaign costs
