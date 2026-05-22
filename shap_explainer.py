# ============================================================
# Credit Risk Classifier — Phase 3: SHAP Explainability
# ============================================================
# Run this after Phase 2 (train_model.py).
# Requires: pip install shap
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("=" * 55)
print("  Credit Risk Classifier — Phase 3: SHAP Explainability")
print("=" * 55)

# ── 1. RECREATE FEATURES (same as Phase 2) ──────────────────
print("\n[1/4] Preparing data...")

df = pd.read_csv("lending_club_sample.csv", low_memory=False)
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
df["default"] = (df["loan_status"] == "Charged Off").astype(int)

FEATURES = [
    "loan_amnt", "int_rate", "grade", "emp_length", "annual_inc",
    "dti", "fico_range_low", "revol_util", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec",
    "purpose", "home_ownership", "term",
]

FEATURE_LABELS = {
    "loan_amnt":       "Loan Amount ($)",
    "int_rate":        "Interest Rate (%)",
    "grade":           "Loan Grade (A–G)",
    "emp_length":      "Employment Length (yrs)",
    "annual_inc":      "Annual Income ($)",
    "dti":             "Debt-to-Income Ratio",
    "fico_range_low":  "FICO Credit Score",
    "revol_util":      "Credit Utilization (%)",
    "delinq_2yrs":     "Delinquencies (2 yrs)",
    "inq_last_6mths":  "Credit Inquiries (6 mo)",
    "open_acc":        "Open Credit Accounts",
    "pub_rec":         "Public Records",
    "purpose":         "Loan Purpose",
    "home_ownership":  "Home Ownership",
    "term":            "Loan Term (months)",
}

X = df[FEATURES].copy()
y = df["default"].copy()

def clean_emp_length(val):
    if pd.isnull(val): return 0
    if "10+" in str(val): return 10
    if "< 1" in str(val): return 0
    return int("".join(filter(str.isdigit, str(val))))

X["emp_length"] = X["emp_length"].apply(clean_emp_length)
X["term"] = X["term"].str.extract(r"(\d+)").astype(int)
X["revol_util"] = X["revol_util"].fillna(X["revol_util"].median())
X = X.fillna(X.median(numeric_only=True))

# Store original values before encoding (for readable explanations)
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
grade_map_inv = {v: k for k, v in grade_map.items()}
purpose_map_inv = {0:"car", 1:"credit card", 2:"debt consolidation",
                   3:"home improvement", 4:"house", 5:"major purchase",
                   6:"medical", 7:"moving", 8:"other",
                   9:"renewable energy", 10:"small business", 11:"vacation"}
home_map_inv = {0:"ANY", 1:"MORTGAGE", 2:"OWN", 3:"RENT"}

for col in ["grade", "purpose", "home_ownership"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Test set: {len(X_test):,} loans")

# ── 2. LOAD MODEL + COMPUTE SHAP VALUES ─────────────────────
print("\n[2/4] Loading model and computing SHAP values...")
print("    (This takes ~30 seconds for 8,800 loans...)")

with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("    SHAP values computed!")

# ── 3. PLOT 1 — GLOBAL FEATURE IMPORTANCE ───────────────────
print("\n[3/4] Generating charts...")

readable_cols = [FEATURE_LABELS[f] for f in FEATURES]
X_test_readable = X_test.copy()
X_test_readable.columns = readable_cols

fig1, ax1 = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_test_readable,
    plot_type="bar",
    show=False,
    color="#378ADD"
)
plt.title("Global Feature Importance (SHAP)\nWhich factors drive default risk most?", fontsize=13)
plt.tight_layout()
plt.savefig("shap_global_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: shap_global_importance.png")

# ── PLOT 2 — BEESWARM (impact direction) ────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test_readable,
    show=False
)
plt.title("SHAP Beeswarm — How Each Feature Pushes Risk Up or Down", fontsize=12)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: shap_beeswarm.png")

# ── 4. INDIVIDUAL LOAN EXPLANATIONS ─────────────────────────
print("\n[4/4] Explaining individual loans...")

def explain_loan(X_test, shap_values, idx, y_test, model, label=""):
    """Print a human-readable explanation for one loan."""
    row = X_test.iloc[idx]
    prob = model.predict_proba(X_test.iloc[[idx]])[0][1]
    actual = y_test.iloc[idx]
    sv = shap_values[idx]

    grade_val = grade_map_inv.get(int(X_test.iloc[idx]["grade"]), "?")
    purpose_val = purpose_map_inv.get(int(X_test.iloc[idx]["purpose"]), "?")

    print(f"\n  {'─'*50}")
    print(f"  LOAN #{idx} — {label}")
    print(f"  {'─'*50}")
    print(f"  Default probability : {prob:.1%}")
    print(f"  Risk tier           : {'🔴 HIGH' if prob > 0.5 else '🟡 MEDIUM' if prob > 0.3 else '🟢 LOW'}")
    print(f"  Actual outcome      : {'Charged Off ❌' if actual == 1 else 'Fully Paid ✅'}")
    print(f"\n  Loan details:")
    print(f"    Amount: ${row['loan_amnt']:,.0f}  |  Rate: {row['int_rate']:.1f}%  |  Grade: {grade_val}")
    print(f"    Income: ${row['annual_inc']:,.0f}  |  DTI: {row['dti']:.1f}%  |  FICO: {row['fico_range_low']:.0f}")
    print(f"    Purpose: {purpose_val}  |  Term: {row['term']:.0f} months")

    print(f"\n  Why the model flagged this loan (top factors):")
    shap_pairs = list(zip(FEATURES, sv))
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, val in shap_pairs[:6]:
        direction = "↑ INCREASES risk" if val > 0 else "↓ DECREASES risk"
        raw = row[feat]
        label_str = FEATURE_LABELS[feat]
        if feat == "grade":
            raw_str = grade_map_inv.get(int(raw), str(raw))
        elif feat == "purpose":
            raw_str = purpose_map_inv.get(int(raw), str(raw))
        elif feat == "home_ownership":
            raw_str = home_map_inv.get(int(raw), str(raw))
        elif feat in ["loan_amnt", "annual_inc"]:
            raw_str = f"${raw:,.0f}"
        elif feat in ["int_rate", "dti", "revol_util"]:
            raw_str = f"{raw:.1f}%"
        else:
            raw_str = f"{raw:.0f}"
        print(f"    {direction:22s} | {label_str}: {raw_str}  (SHAP: {val:+.3f})")

    # Waterfall chart for this loan
    fig, ax = plt.subplots(figsize=(10, 5))
    top_features = [FEATURE_LABELS[f] for f, _ in shap_pairs[:8]]
    top_shap = [v for _, v in shap_pairs[:8]]
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in top_shap]
    bars = ax.barh(top_features[::-1], top_shap[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on default probability)")
    ax.set_title(f"Loan #{idx} — Why risk is {prob:.1%}\n{label}", fontsize=11)
    red_patch = mpatches.Patch(color="#E24B4A", label="Increases default risk")
    green_patch = mpatches.Patch(color="#1D9E75", label="Decreases default risk")
    ax.legend(handles=[red_patch, green_patch], loc="lower right")
    plt.tight_layout()
    fname = f"shap_loan_{idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Chart saved: {fname}")

# Find a high-risk defaulted loan
high_risk_idx = (
    pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    * (y_test == 1)
).nlargest(1).index[0]
high_risk_pos = X_test.index.get_loc(high_risk_idx)

# Find a low-risk paid loan
low_risk_idx = (
    pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    * (y_test == 0)
).nsmallest(1).index[0]
low_risk_pos = X_test.index.get_loc(low_risk_idx)

explain_loan(X_test, shap_values, high_risk_pos, y_test, model,
             label="HIGH RISK — Model correctly flagged this default")
explain_loan(X_test, shap_values, low_risk_pos, y_test, model,
             label="LOW RISK — Model correctly cleared this loan")

print("\n" + "=" * 55)
print("  ✅ Phase 3 complete!")
print("=" * 55)
print("\n  Files saved:")
print("    shap_global_importance.png  — overall feature ranking")
print("    shap_beeswarm.png           — how features push risk")
print("    shap_loan_*.png             — individual explanations")
print("\n  Next: Phase 4 — Full Streamlit app anyone can use!")