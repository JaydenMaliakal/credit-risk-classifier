# ============================================================
# Credit Risk Classifier - Phase 4: Streamlit Web App
# ============================================================
# Install dependencies:
#   pip install streamlit shap matplotlib
# Run with:
#   streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import shap

st.set_page_config(
    page_title="Credit Risk Classifier",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("credit_risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_model()

GRADE_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
PURPOSE_MAP = {
    "Car": 0, "Credit Card": 1, "Debt Consolidation": 2,
    "Home Improvement": 3, "House": 4, "Major Purchase": 5,
    "Medical": 6, "Moving": 7, "Other": 8,
    "Renewable Energy": 9, "Small Business": 10, "Vacation": 11,
}
HOME_MAP = {"ANY": 0, "Mortgage": 1, "Own": 2, "Rent": 3}

FEATURE_LABELS = {
    "loan_amnt":      "Loan Amount ($)",
    "int_rate":       "Interest Rate (%)",
    "grade":          "Loan Grade (A-G)",
    "emp_length":     "Employment Length (yrs)",
    "annual_inc":     "Annual Income ($)",
    "dti":            "Debt-to-Income Ratio",
    "fico_range_low": "FICO Credit Score",
    "revol_util":     "Credit Utilization (%)",
    "delinq_2yrs":    "Delinquencies (last 2 yrs)",
    "inq_last_6mths": "Credit Inquiries (last 6 mo)",
    "open_acc":       "Open Credit Accounts",
    "pub_rec":        "Public Records",
    "purpose":        "Loan Purpose",
    "home_ownership": "Home Ownership",
    "term":           "Loan Term (months)",
}

FEATURES = list(FEATURE_LABELS.keys())

st.title("Credit Risk Classifier")
st.markdown(
    "Enter loan details below to predict the probability of default. "
    "Built with **XGBoost** and **SHAP** on 44,000 real LendingClub loans."
)
st.divider()

st.sidebar.header("Loan Application Details")
st.sidebar.markdown("Adjust the values to match the applicant.")

loan_amnt   = st.sidebar.number_input("Loan Amount ($)",              1000, 35000, 10000, step=500)
int_rate    = st.sidebar.slider("Interest Rate (%)",                  5.0,  30.0,  12.0,  step=0.1)
grade       = st.sidebar.selectbox("Loan Grade",                      list(GRADE_MAP.keys()))
emp_length  = st.sidebar.slider("Employment Length (years)",          0,    10,    3)
annual_inc  = st.sidebar.number_input("Annual Income ($)",            10000, 300000, 60000, step=1000)
dti         = st.sidebar.slider("Debt-to-Income Ratio (%)",           0.0,  50.0,  15.0,  step=0.1)
fico        = st.sidebar.slider("FICO Credit Score",                  580,  850,   690,   step=5)
revol_util  = st.sidebar.slider("Credit Utilization (%)",             0.0,  100.0, 50.0,  step=1.0)
delinq      = st.sidebar.number_input("Delinquencies (last 2 yrs)",   0, 20, 0)
inq         = st.sidebar.number_input("Credit Inquiries (last 6 mo)", 0, 10, 1)
open_acc    = st.sidebar.number_input("Open Credit Accounts",         1,    40,    8)
pub_rec     = st.sidebar.number_input("Public Records",               0,    10,    0)
purpose     = st.sidebar.selectbox("Loan Purpose",                    list(PURPOSE_MAP.keys()))
home        = st.sidebar.selectbox("Home Ownership",                  list(HOME_MAP.keys()))
term        = st.sidebar.selectbox("Loan Term (months)",              [36, 60])

predict_btn = st.sidebar.button("Analyze Risk", use_container_width=True, type="primary")

if predict_btn:
    input_data = pd.DataFrame([{
        "loan_amnt":      loan_amnt,
        "int_rate":       int_rate,
        "grade":          GRADE_MAP[grade],
        "emp_length":     emp_length,
        "annual_inc":     annual_inc,
        "dti":            dti,
        "fico_range_low": fico,
        "revol_util":     revol_util,
        "delinq_2yrs":    delinq,
        "inq_last_6mths": inq,
        "open_acc":       open_acc,
        "pub_rec":        pub_rec,
        "purpose":        PURPOSE_MAP[purpose],
        "home_ownership": HOME_MAP[home],
        "term":           term,
    }])

    prob = model.predict_proba(input_data)[0][1]
    sv   = explainer.shap_values(input_data)[0]

    if prob >= 0.5:
        tier, color = "HIGH RISK", "#E24B4A"
    elif prob >= 0.3:
        tier, color = "MEDIUM RISK", "#BA7517"
    else:
        tier, color = "LOW RISK", "#1D9E75"

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### Risk Assessment")
        st.markdown(
            f"<div style='background:{color}18; border:1.5px solid {color}; "
            f"border-radius:12px; padding:20px; text-align:center;'>"
            f"<div style='font-size:36px; font-weight:600; color:{color};'>{prob:.1%}</div>"
            f"<div style='font-size:18px; color:{color}; font-weight:500;'>{tier}</div>"
            f"<div style='font-size:13px; color:gray; margin-top:6px;'>Default probability</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Loan Summary")
        st.markdown(f"""
| Field | Value |
|-------|-------|
| Amount | ${loan_amnt:,} |
| Interest Rate | {int_rate}% |
| Grade | {grade} |
| Term | {term} months |
| FICO Score | {fico} |
| DTI | {dti}% |
| Purpose | {purpose} |
| Home | {home} |
""")

    with col3:
        st.markdown("### Risk Gauge")
        fig_gauge, ax = plt.subplots(figsize=(4, 2.5))
        ax.barh(["Default Risk"], [prob], color=color, height=0.4)
        ax.barh(["Default Risk"], [1 - prob], left=[prob], color="#E8E8E8", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.axvline(0.3, color="#BA7517", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(0.5, color="#E24B4A", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(f"Predicted default probability: {prob:.1%}", fontsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_gauge)
        plt.close()

    st.divider()

    st.markdown("### Why did the model give this score?")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows exactly which factors "
        "pushed the risk **up** or **down** for this specific applicant."
    )

    col4, col5 = st.columns([1.2, 1])

    with col4:
        shap_pairs = sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)[:8]
        labels = [FEATURE_LABELS[f] for f, _ in shap_pairs]
        values = [v for _, v in shap_pairs]
        colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in values]

        fig_shap, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("SHAP value - impact on default probability")
        ax2.set_title("Feature contributions for this applicant", fontsize=12)
        red_p = mpatches.Patch(color="#E24B4A", label="Increases default risk")
        grn_p = mpatches.Patch(color="#1D9E75", label="Decreases default risk")
        ax2.legend(handles=[red_p, grn_p])
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

    with col5:
        st.markdown("#### Top factors explained")
        for feat, val in shap_pairs[:6]:
            direction = "Increases risk" if val > 0 else "Decreases risk"
            raw = input_data.iloc[0][feat]
            label_str = FEATURE_LABELS[feat]

            if feat == "grade":
                raw_str = grade
            elif feat == "purpose":
                raw_str = purpose
            elif feat == "home_ownership":
                raw_str = home
            elif feat in ["loan_amnt", "annual_inc"]:
                raw_str = f"${raw:,.0f}"
            elif feat in ["int_rate", "dti", "revol_util"]:
                raw_str = f"{raw:.1f}%"
            else:
                raw_str = str(int(raw))

            st.markdown(
                f"<div style='padding:10px; margin-bottom:8px; border-radius:8px; "
                f"background:{'#FCEBEB' if val > 0 else '#EAF3DE'}; "
                f"border-left:3px solid {'#E24B4A' if val > 0 else '#1D9E75'};'>"
                f"<b>{label_str}</b>: {raw_str}<br>"
                f"<span style='font-size:13px; color:{'#A32D2D' if val > 0 else '#3B6D11'};'>"
                f"{direction} (SHAP: {val:+.3f})</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown("### Recommendation")
    if prob >= 0.5:
        st.error(
            f"**Decline recommended.** This applicant has a {prob:.1%} predicted default probability. "
            f"Key risk factors include a high interest rate, elevated DTI, or poor credit history. "
            f"Consider requesting additional collateral or a co-signer."
        )
    elif prob >= 0.3:
        st.warning(
            f"**Review carefully.** Default probability is {prob:.1%}, which represents moderate risk. "
            f"Consider a lower loan amount or shorter term to reduce exposure."
        )
    else:
        st.success(
            f"**Approve recommended.** Default probability is only {prob:.1%}. "
            f"Strong credit profile with manageable debt load."
        )

else:
    st.markdown("### Enter loan details in the sidebar and click Analyze Risk")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### How it works
        1. Enter applicant details in the sidebar
        2. Click **Analyze Risk**
        3. Get a default probability score
        4. See exactly why via SHAP
        """)
    with col2:
        st.markdown("""
        #### What is SHAP?
        SHAP (SHapley Additive exPlanations) is a
        technique that explains why a model made a
        specific prediction, not just what the
        prediction was.
        """)
    with col3:
        st.markdown("""
        #### About the model
        - Trained on 44,000 real LendingClub loans
        - XGBoost classifier (AUC: 0.72)
        - 15 financial features
        """)



st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "Built with Python, XGBoost, SHAP, and Streamlit. "
    "Data: LendingClub 2007-2018"
    "</div>",
    unsafe_allow_html=True
)