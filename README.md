# Credit Risk Classifier

I've always had a passion for computer science, and over time my curiosity grew upon learning how the same technology that can make games is also able to analyze market patterns and predict real world outcomes. This project sits at the intersection of two things I find genuinely interesting, which are computer science and finance, and my goal was to build something that could actually be useful to people regardless of their financial background.

The Credit Risk Classifier predicts the probability of a loan defaulting using real LendingClub data, and pairs that prediction with SHAP explainability to show the thought process behind each result. Rather than giving a score with no context, the tool breaks down which specific financial factors pushed the risk up or down for each individual applicant, making the output transparent and accessible.

---

## Overview

- Predicts the probability a borrower will default on their loan
- Uses SHAP values to explain which factors drove each individual prediction
- Deployed as an interactive web app where anyone can enter loan details and get a result

---

## Dataset

- **Source:** LendingClub loan data (2007–2018), via UCI / Kaggle
- **Size:** 44,005 closed loans (fully paid or charged off)
- **Default rate:** 20.5%
- **Features used:** 15 financial features including FICO score, debt-to-income ratio, loan grade, interest rate, employment length, and more

---

## Model

| Detail | Value |
|--------|-------|
| Algorithm | XGBoost Classifier |
| AUC Score | 0.72 |
| Training set | 35,204 loans |
| Test set | 8,801 loans |
| Class imbalance handling | scale_pos_weight |

XGBoost was selected as the model of choice given that it is the industry standard for tabular financial data and is widely used in production environments at banks and fintech companies.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) breaks down each prediction by quantifying how much each feature pushed the risk score up or down for a specific applicant. This was the most interesting part of the project to build because it transforms the model from a black box into something that can be reasoned about and explained to another person.

**Example — High risk loan (95.2% default probability):**
- Interest rate 26.0% — SHAP: +1.626
- Debt-to-income 35.0% — SHAP: +0.507
- 60-month term — SHAP: +0.426
- FICO score 670 — SHAP: +0.206

**Example — Low risk loan (7.9% default probability):**
- Interest rate 8.4% — SHAP: -0.967
- Debt-to-income 3.9% — SHAP: -0.715
- Annual income $25,000 — SHAP: -0.301

---

## Web App

Built with Streamlit. Enter any loan applicant's details and get back a default probability, a risk tier, a SHAP chart explaining the prediction, and a plain-English recommendation.

### Run locally

```bash
git clone https://github.com/JaydenMaliakal/credit-risk-classifier.git
cd credit-risk-classifier

pip install streamlit xgboost shap pandas scikit-learn matplotlib seaborn

streamlit run app.py
```

---

## Project Structure

```
credit-risk-classifier/
│
├── app.py                  # Streamlit web app
├── train_model.py          # XGBoost training script
├── shap_explainer.py       # SHAP analysis script
├── trim_data.py            # Data trimming utility
├── credit_risk_model.pkl   # Trained model
└── README.md
```

---

## Key Findings

- Interest rate was the strongest single predictor of default, outweighing even FICO score, which was an unexpected result going into the analysis
- The model produced false negatives in cases where the interest rate appeared healthy but the debt-to-income ratio was elevated, reinforcing the idea that machine learning models still require human judgment layered on top of them
- Properly accounting for the class imbalance between paid and defaulted loans made a meaningful difference to the model's ability to correctly identify actual defaults

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **XGBoost** — gradient boosted decision trees
- **SHAP** — model explainability
- **Streamlit** — web app framework
- **Matplotlib / Seaborn** — data visualization

---

## License

MIT License — free to use, modify, and distribute.
