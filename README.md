# Credit Wise Loan Intelligence System

AI-powered loan approval system using machine learning, enhanced with explainable AI and an interactive dashboard for real-time decision support.

---

## Overview

This project is an end-to-end machine learning application that predicts loan approval based on applicant financial and personal details.

Unlike traditional prediction systems, this project focuses on **interpretability and decision intelligence** by integrating SHAP-based explanations and clear reasoning behind each prediction.

It simulates a real-world financial system where not only predictions matter, but also understanding *why* a decision was made.

---

## Features

* Real-time loan approval prediction (Approved / Rejected)
* Approval probability with risk categorization
* Explainable AI using SHAP values
* Visualization of feature impact on predictions
* Decision-based storytelling (why approved or rejected)
* Actionable feedback for improving approval chances
* Interactive dashboard including:

  * Loan distribution visualization
  * Income analysis
  * Confusion matrix
  * ROC curve comparison
* Clean and structured UI using Streamlit

---

## Machine Learning Models

The system uses multiple models for performance comparison:

* Logistic Regression (with hyperparameter tuning)
* Random Forest
* Gradient Boosting

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SHAP (Explainability)

---

## Machine Learning Workflow

1. Data loading and preprocessing
2. Handling missing values using SimpleImputer
3. Encoding categorical variables using OneHotEncoder
4. Feature scaling using StandardScaler
5. Model training and hyperparameter tuning using GridSearchCV
6. Model evaluation and comparison
7. SHAP-based explainability integration
8. Deployment using Streamlit

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

## Use Case

This project demonstrates how machine learning can be used in financial decision-making systems where transparency and explainability are critical. It highlights how predictions can be combined with reasoning to support better and more trustworthy decisions.

---

## Author

Rajneel Deepak Chavan
