# Credit Wise Loan Intelligence System

AI-powered loan approval system using machine learning with an interactive dashboard and real-time predictions.

---

## Overview

This project is an end-to-end machine learning application designed to predict loan approval based on applicant financial and personal details. It integrates multiple models, performs data preprocessing, and provides a user-friendly interface using Streamlit.

---

## Features

* Predicts loan approval (Approved / Rejected)
* Uses multiple machine learning models:

  * Logistic Regression (with hyperparameter tuning)
  * Random Forest
  * Gradient Boosting
* Model comparison using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
* Interactive dashboard with:

  * Loan distribution visualization
  * Income analysis
  * Confusion matrix
  * ROC curve comparison
* Real-time prediction with simple reasoning
* Clean and structured user interface using Streamlit

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Machine Learning Workflow

1. Data loading and preprocessing
2. Handling missing values using SimpleImputer
3. Encoding categorical variables using OneHotEncoder
4. Feature scaling using StandardScaler
5. Model training and hyperparameter tuning using GridSearchCV
6. Model evaluation and comparison
7. Deployment using Streamlit

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



