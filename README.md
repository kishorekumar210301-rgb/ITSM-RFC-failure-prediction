# ITSM-RFC-failure-prediction

# Machine Learning for IT Service Management (ITSM): Predictive Analytics & RFC Failure Prediction

> **Company Project** | Project ID: PRCL-0012

## Overview

Mid-sized IT service providers handle 20,000–25,000 incidents a year. Even with mature ITIL processes, incident management is often reactive and customer satisfaction stays low. This project applies machine learning to historical ITSM data to move operations from **reactive to proactive**, with the primary focus on **predicting RFC (Request for Change) failures**.

## Business Objectives

- Predict high-priority tickets (P1/P2) for faster intervention
- Forecast incident volumes for workforce and capacity planning
- Automate ticket classification to reduce manual reassignment delays
- **Predict RFC failures so change-management teams can act preventively**

## Use Cases & Models

| # | Use Case | Task | Models Compared |
|---|----------|------|-----------------|
| 1 | High-priority ticket prediction | Classification | Logistic Regression, Random Forest, XGBoost |
| 2 | Incident volume forecasting | Regression (lag features) | Linear Regression, Random Forest, XGBoost |
| 3 | Automated ticket tagging | Multi-class classification | Decision Tree, Random Forest, Naive Bayes, SVM |
| 4 | **RFC failure prediction** | Binary classification | Random Forest, XGBoost, Neural Network |

## Key Results

- **High-priority prediction:** all three models reached about 99.5%+ accuracy (Logistic Regression best at 99.62%).
- **Volume forecasting:** XGBoost and Random Forest (R² ≈ 0.96) clearly beat Linear Regression (R² ≈ 0.84).
- **Auto-tagging:** Random Forest was best at about 91.3% accuracy.
- **RFC failure prediction:** a harder, imbalanced problem. Models were built on a **leakage-free** feature set, which gives modest but realistic performance. See the notebook for the full comparison.

## Methodology

1. **Exploratory Data Analysis:** distributions, cross-analysis (Impact/Urgency/Priority/Category), skewness and kurtosis, correlation heatmap
2. **Preprocessing:** missing-value imputation, one-hot and label encoding, scaling (MinMax and Standard), outlier treatment (log transform)
3. **Feature engineering:** severity score, ratios, binning, interaction and polynomial features
4. **Target creation and leakage prevention:** removed post-resolution and outcome-related columns before training the RFC failure model
5. **Modeling and evaluation:** train/test split, accuracy, precision, recall, F1, RMSE and R², plus feature importance analysis

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · TensorFlow/Keras

## Repository Structure

```
├── RFC_Failure_Prediction_ITSM.ipynb   # Full analysis and modeling notebook
├── README.md
├── LICENSE
└── .gitignore
```

## How to Run

```bash
git clone https://github.com/kishorekumar210301-rgb/ITSM-RFC-failure-prediction.git
cd ITSM-RFC-failure-prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow jupyter
jupyter notebook RFC_Failure_Prediction_ITSM.ipynb
```

> **Note:** The original dataset came from a private company database and is **not included** in this repository. To run the notebook, load your own ITSM dataset into a DataFrame named `data` in place of the database-loading cell.

## Limitations

- Class imbalance (successful changes far outnumber failed ones) limits recall on rare failures
- Training data comes from a single organization, so results may not generalize
- RFC failure prediction accuracy is moderate and should support, not replace, human judgement

## Future Scope

- Use LLMs/Transformers on unstructured work notes and implementation plans
- CMDB dependency graphing to estimate change blast radius
- Real-time integration with the ITSM platform for automated risk scoring

## Author

**Kishore Kumar G**
