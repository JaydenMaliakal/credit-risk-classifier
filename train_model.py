# ============================================================
# Credit Risk Classifier — Phase 2: Train XGBoost Model
# ============================================================
# Run this in PyCharm after installing dependencies:
#   pip install pandas scikit-learn xgboost matplotlib seaborn
# Make sure lending_club_sample.csv is in the same folder.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

print("=" * 55)
print("  Credit Risk Classifier — Phase 2: Model Training")
print("=" * 55)

# ── 1. LOAD DATA ────────────────────────────────────────────
print("\n[1/5] Loading data...")
df = pd.read_csv("lending_club_sample.csv", low_memory=False)

# Keep only closed loans (fully paid or charged off)
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
print(f"    Closed loans: {len(df):,}")

# Target: 1 = default, 0 = paid
df["default"] = (df["loan_status"] == "Charged Off").astype(int)
print(f"    Default rate: {df['default'].mean():.1%}")

# ── 2. FEATURE ENGINEERING ──────────────────────────────────
print("\n[2/5] Engineering features...")

# Select the 15 most predictive features
FEATURES = [
    "loan_amnt",       # How much they borrowed
    "int_rate",        # Interest rate — higher = riskier
    "grade",           # LendingClub's own risk grade (A-G)
    "emp_length",      # Employment length
    "annual_inc",      # Annual income
    "dti",             # Debt-to-income ratio
    "fico_range_low",  # FICO credit score
    "revol_util",      # Credit utilization %
    "delinq_2yrs",     # Delinquencies in last 2 years
    "inq_last_6mths",  # Credit inquiries last 6 months
    "open_acc",        # Number of open credit accounts
    "pub_rec",         # Public derogatory records
    "purpose",         # Loan purpose (debt, credit card, etc)
    "home_ownership",  # Rent / Own / Mortgage
    "term",            # 36 or 60 month loan
]

X = df[FEATURES].copy()
y = df["default"].copy()

# Clean emp_length — convert "10+ years" → 10, "< 1 year" → 0
def clean_emp_length(val):
    if pd.isnull(val):
        return 0
    if "10+" in str(val):
        return 10
    if "< 1" in str(val):
        return 0
    return int("".join(filter(str.isdigit, str(val))))

X["emp_length"] = X["emp_length"].apply(clean_emp_length)

# Clean term — "36 months" → 36
X["term"] = X["term"].str.extract(r"(\d+)").astype(int)

# Fill remaining nulls with median
X["revol_util"] = X["revol_util"].fillna(X["revol_util"].median())

# Encode categorical columns (grade, purpose, home_ownership)
for col in ["grade", "purpose", "home_ownership"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print(f"    Features ready: {X.shape[1]} columns, {len(X):,} rows")
print(f"    No nulls: {X.isnull().sum().sum() == 0}")

# ── 3. TRAIN / TEST SPLIT ───────────────────────────────────
print("\n[3/5] Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train):,} rows")
print(f"    Test:  {len(X_test):,} rows")

# ── 4. TRAIN XGBOOST ────────────────────────────────────────
print("\n[4/5] Training XGBoost model...")

# scale_pos_weight handles class imbalance (80% paid vs 20% default)
scale = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale,   # Handles class imbalance
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

model.fit(X_train, y_train)
print("    Training complete!")

# ── 5. EVALUATE ─────────────────────────────────────────────
print("\n[5/5] Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("\n" + "─" * 55)
print("  RESULTS")
print("─" * 55)
print(f"\n  AUC Score:     {auc:.4f}  (1.0 = perfect, 0.5 = random)")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Fully Paid", "Charged Off"]))

# ── PLOTS ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Credit Risk Classifier — Model Results", fontsize=14)

# Plot 1: Feature importance
importances = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=True)

axes[0].barh(importances.index, importances.values, color="#378ADD")
axes[0].set_title("Feature Importance")
axes[0].set_xlabel("Importance score")

# Plot 2: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Paid", "Default"],
            yticklabels=["Paid", "Default"])
axes[1].set_title("Confusion Matrix")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# Plot 3: ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[2])
axes[2].set_title(f"ROC Curve (AUC = {auc:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Chart saved as model_results.png")

# ── SAVE MODEL ──────────────────────────────────────────────
with open("credit_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("  Model saved as credit_risk_model.pkl")
print("\n✅ Phase 2 complete! Ready for Phase 3 (SHAP explainability).")