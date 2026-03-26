# Credit Wise Loan Intelligence System

AI-powered loan approval system built using machine learning with explainability, dynamic tracking, and an interactive dashboard.

---

## Overview

This project is an end-to-end machine learning system designed to predict loan approval based on applicant financial and personal data. It goes beyond traditional models by integrating explainability, real-time-like tracking, and a self-improving retraining mechanism.

The system is deployed using Streamlit and provides an interactive interface for both predictions and insights.

---

## Key Features

* Loan approval prediction (Approved / Rejected)
* Multiple machine learning models:

  * Logistic Regression (with hyperparameter tuning)
  * Random Forest
  * Gradient Boosting
* Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
* SHAP-based explainability:

  * Feature impact visualization
  * Human-readable decision reasoning
* Interactive dashboard:

  * Approval distribution
  * Income vs credit score analysis
  * Model performance comparison
  * Correlation heatmap
* Prediction tracking system:

  * Stores user predictions in history
  * Enables trend monitoring
* Self-improving system:

  * Automatically retrains model using historical prediction data
  * Adapts to changing patterns over time
* Clean and user-friendly interface built with Streamlit

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SHAP

---

## System Workflow

1. Data preprocessing (handling missing values, encoding, scaling)
2. Model training and evaluation
3. Deployment using Streamlit
4. User input and real-time prediction
5. Prediction logging in history
6. Periodic retraining using new data

---

## Run Locally

```bash
git clone https://github.com/Rajneel-Chavan/credit-wise-loan-intelligence-system.git
cd credit-wise-loan-intelligence-system
pip install -r requirements.txt
streamlit run app.py
```

---

## Live Demo

https://credit-wise-loan-intelligence-system-ejiucqzy2xjwakxm9mmewu.streamlit.app/

---

## Note

This project demonstrates not only machine learning modeling but also system-level thinking, including explainability, monitoring, and model lifecycle handling.

---
