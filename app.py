import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import shap

# =================================================================
# 1. SYSTEM SETTINGS & LOGGING
# =================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Credit Wise Loan Intelligence", layout="wide", page_icon="🏦")

# Global File Paths
DATA_PATH = "loan_approval_data.csv"
HISTORY_PATH = "history.csv"
MODEL_PATH = "model_assets.joblib"

st.title("🏦 Credit Wise: Advanced Loan Intelligence System")
st.markdown("---")

# =================================================================
# 2. AUTO-UPDATE LOGIC (1000 PREDICTIONS THRESHOLD)
# =================================================================
# Every time the app loads, it checks if history has reached 1000 rows.
# If it has, it deletes the model assets to trigger a full retrain.
RETRAIN_THRESHOLD = 1000

if os.path.exists(HISTORY_PATH):
    try:
        current_history = pd.read_csv(HISTORY_PATH)
        if len(current_history) >= RETRAIN_THRESHOLD:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
                logger.info(f"Retrain threshold {RETRAIN_THRESHOLD} reached. Model purged for update.")
    except Exception as e:
        logger.warning(f"Auto-update check bypassed: {e}")

# =================================================================
# 3. DATA LOADING & PREPROCESSING (DETAILED)
# =================================================================
if not os.path.exists(DATA_PATH):
    st.error("Error: 'loan_approval_data.csv' not found!")
    st.stop()

# Load raw data
raw_df = pd.read_csv(DATA_PATH)

# Handling Missing Values Manually
categorical_features = raw_df.select_dtypes(include=["object"]).columns
numerical_features = raw_df.select_dtypes(include=["float64", "int64"]).columns

imputer_num = SimpleImputer(strategy="mean")
raw_df[numerical_features] = imputer_num.fit_transform(raw_df[numerical_features])

imputer_cat = SimpleImputer(strategy="most_frequent")
raw_df[categorical_features] = imputer_cat.fit_transform(raw_df[categorical_features])

# Drop non-essential ID columns
if "Applicant_ID" in raw_df.columns:
    raw_df = raw_df.drop("Applicant_ID", axis=1)

# Manual Feature Engineering: Education Level
raw_df["Education_Level"] = raw_df["Education_Level"].map({"Graduate": 1, "Not Graduate": 0})

# Encoding the Target Variable
label_encoder = LabelEncoder()
raw_df["Loan_Approved"] = label_encoder.fit_transform(raw_df["Loan_Approved"])

# One-Hot Encoding for Categorical Variables
ohe_columns = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe_processor = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded_array = ohe_processor.fit_transform(raw_df[ohe_columns])
encoded_feature_names = ohe_processor.get_feature_names_out(ohe_columns)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=raw_df.index)

# Combine and Create Final Training Set
final_training_df = pd.concat([raw_df.drop(columns=ohe_columns), encoded_df], axis=1)

# =================================================================
# 4. MACHINE LEARNING PIPELINE (GRIDSEARCH & ENSEMBLES)
# =================================================================
X = final_training_df.drop(columns=["Loan_Approved"])
y = final_training_df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling Features
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Check if model already exists, otherwise Train
if os.path.exists(MODEL_PATH):
    system_assets = joblib.load(MODEL_PATH)
    logger.info("Loaded pre-trained model assets.")
else:
    with st.spinner("Auto-Updating AI Models with latest data..."):
        # Model A: Logistic Regression with Hyperparameter Tuning
        log_reg = LogisticRegression(max_iter=2000)
        log_params = {'C': [0.1, 1, 10]}
        grid_log = GridSearchCV(log_reg, log_params, cv=3).fit(X_train_scaled, y_train)
        best_log = grid_log.best_estimator_

        # Model B: Random Forest Ensemble
        rf_ensemble = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf_ensemble.fit(X_train_scaled, y_train)

        # Model C: Gradient Boosting Machine
        gb_machine = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        gb_machine.fit(X_train_scaled, y_train)

        # Evaluate and Prepare Results Table
        def get_model_stats(m, name):
            p = m.predict(X_test_scaled)
            return [name, accuracy_score(y_test, p), f1_score(y_test, p), roc_auc_score(y_test, m.predict_proba(X_test_scaled)[:,1])]

        results_list = [
            get_model_stats(best_log, "Logistic Regression"),
            get_model_stats(rf_ensemble, "Random Forest"),
            get_model_stats(gb_machine, "Gradient Boosting")
        ]
        comparison_df = pd.DataFrame(results_list, columns=["Model", "Accuracy", "F1 Score", "ROC-AUC"])

        # Save all assets into Joblib
        system_assets = {
            "log_model": best_log,
            "rf_model": rf_ensemble,
            "gb_model": gb_machine,
            "scaler": feature_scaler,
            "ohe": ohe_processor,
            "ohe_cols": ohe_columns,
            "features": X.columns.tolist(),
            "stats": comparison_df,
            "raw_data": raw_df,
            "X_test": X_test_scaled,
            "y_test": y_test
        }
        joblib.dump(system_assets, MODEL_PATH)

# =================================================================
# 5. SIDEBAR NAVIGATION
# =================================================================
st.sidebar.title("Navigation Menu")
app_mode = st.sidebar.radio("Go To Page:", ["Executive Dashboard", "Loan Prediction Engine"])

# =================================================================
# 6. DASHBOARD PAGE
# =================================================================
if app_mode == "Executive Dashboard":
    st.header("Executive Portfolio Analysis")
    
    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    avg_approval = system_assets["raw_data"]["Loan_Approved"].mean() * 100
    avg_income = system_assets["raw_data"]["Applicant_Income"].mean()
    total_apps = len(system_assets["raw_data"])
    avg_credit = system_assets["raw_data"]["Credit_Score"].mean()

    m1.metric("Approval Rate", f"{avg_approval:.1f}%")
    m2.metric("Total Records", f"{total_apps}")
    m3.metric("Avg Applicant Income", f"${avg_income:,.0f}")
    m4.metric("Mean Credit Score", f"{avg_credit:.0f}")

    st.markdown("---")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Approval Distribution")
        fig_pie, ax_pie = plt.subplots()
        system_assets["raw_data"]["Loan_Approved"].value_counts().plot.pie(
            autopct="%1.1f%%", colors=["#ff4b4b", "#00cc96"], labels=["Rejected", "Approved"], ax=ax_pie
        )
        st.pyplot(fig_pie)
    with col_v2:
        st.subheader("Income vs Credit Score Correlation")
        fig_scat, ax_scat = plt.subplots()
        sns.scatterplot(data=system_assets["raw_data"], x="Applicant_Income", y="Credit_Score", 
                        hue="Loan_Approved", palette={0: "#ff4b4b", 1: "#00cc96"}, ax=ax_scat)
        st.pyplot(fig_scat)

    st.markdown("---")
    st.subheader("🤖 Model Performance Audit")
    tab_metrics, tab_roc = st.tabs(["Metric Benchmarks", "ROC Curve Analysis"])
    with tab_metrics:
        st.table(system_assets["stats"].style.highlight_max(axis=0, color="#D4EDDA"))
    with tab_roc:
        fig_roc, ax_roc = plt.subplots(figsize=(10, 5))
        fpr1, tpr1, _ = roc_curve(system_assets["y_test"], system_assets["log_model"].predict_proba(system_assets["X_test"])[:, 1])
        ax_roc.plot(fpr1, tpr1, label="Logistic Regression")
        fpr2, tpr2, _ = roc_curve(system_assets["y_test"], system_assets["rf_model"].predict_proba(system_assets["X_test"])[:, 1])
        ax_roc.plot(fpr2, tpr2, label="Random Forest")
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.legend()
        st.pyplot(fig_roc)

# =================================================================
# 7. PREDICTION ENGINE (STORY + GRAPH)
# =================================================================
if app_mode == "Loan Prediction Engine":
    st.header("Individual Applicant Assessment")
    
    with st.form("loan_application_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Single", "Married"])
        with c2:
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed"])
        with c3:
            prop_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Business", "Education", "Personal"])

        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            income = st.number_input("Monthly Income (USD)", 0, 100000, 5000)
            age = st.number_input("Applicant Age", 18, 100, 30)
        with c5:
            credit_score = st.number_input("Credit Score", 300, 850, 700)
            loan_amt = st.number_input("Loan Amount (USD)", 0, 500000, 25000)
        with c6:
            dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
            savings = st.number_input("Current Savings", 0, 100000, 10000)

        run_prediction = st.form_submit_button("🚀 Execute Risk Assessment")

    if run_prediction:
        with st.spinner("Analyzing your financial story..."):
            # 1. Transform User Input
            user_input = pd.DataFrame({
                "Education_Level": [1 if education == "Graduate" else 0],
                "Employment_Status": [employment], "Marital_Status": [marital],
                "Loan_Purpose": [loan_purpose], "Property_Area": [prop_area],
                "Gender": [gender], "Employer_Category": ["Private"],
                "Applicant_Income": [income], "Coapplicant_Income": [0],
                "Credit_Score": [credit_score], "DTI_Ratio": [dti],
                "Savings": [savings], "Age": [age], "Loan_Amount": [loan_amt]
            })

            ohe_encoded = system_assets["ohe"].transform(user_input[system_assets["ohe_cols"]])
            ohe_df = pd.DataFrame(ohe_encoded, columns=system_assets["ohe"].get_feature_names_out(system_assets["ohe_cols"]))
            final_user_input = pd.concat([user_input.drop(columns=system_assets["ohe_cols"]), ohe_df], axis=1)
            final_user_input = final_user_input.reindex(columns=system_assets["features"], fill_value=0)
            scaled_user_input = system_assets["scaler"].transform(final_user_input)

            # 2. Decision Logic
            decision = system_assets["rf_model"].predict(scaled_user_input)[0]
            confidence = system_assets["rf_model"].predict_proba(scaled_user_input)[0, 1]

            # 3. Result Display
            st.markdown("---")
            res_col, conf_col = st.columns([2, 1])
            with res_col:
                if decision == 1:
                    st.success("### STATUS: APPLICATION APPROVED")
                    st.balloons()
                else:
                    st.error("### STATUS: APPLICATION DECLINED")
            with conf_col:
                st.metric("Approval Confidence", f"{confidence*100:.1f}%")

            # 4. SHAP Logic (1D Fix)
            explainer = shap.TreeExplainer(system_assets["rf_model"])
            raw_shap_vals = explainer.shap_values(scaled_user_input)
            
            if isinstance(raw_shap_vals, list):
                sv_final = np.array(raw_shap_vals[1]).flatten()
            elif len(raw_shap_vals.shape) == 3:
                sv_final = raw_shap_vals[0, :, 1].flatten()
            else:
                sv_final = raw_shap_vals[:, 1].flatten() if raw_shap_vals.shape[1] == 2 else raw_shap_vals.flatten()

            impact_map = pd.Series(sv_final, index=system_assets["features"]).sort_values(ascending=False)
            top_pos = impact_map.index[0].replace('_', ' ')
            top_neg = impact_map.index[-1].replace('_', ' ')

            # 5. Narrative 
            st.subheader("Decision Narrative")
            if decision == 1:
                st.info(f"**Why you were approved:** The primary factor was your **{top_pos}**. This indicated low risk to the system.")
            else:
                st.warning(f"**Why you were declined:** The system flagged your **{top_neg}** as the primary risk concern.")

            # 6. Graph
            st.subheader("Feature Impact Analysis")
            plot_data = pd.concat([impact_map.head(5), impact_map.tail(5)]).sort_values()
            fig_f, ax_f = plt.subplots(figsize=(10, 6))
            colors = ['#ff4b4b' if x < 0 else '#00cc96' for x in plot_data]
            plot_data.plot(kind='barh', color=colors, ax=ax_f)
            st.pyplot(fig_f)

            # 7. Persistence
            log_entry = user_input.copy()
            log_entry["Loan_Approved"] = "Approved" if decision == 1 else "Rejected"
            log_entry["Timestamp"] = datetime.now()
            log_entry.to_csv(HISTORY_PATH, mode='a', header=not os.path.exists(HISTORY_PATH), index=False)

st.markdown("---")
st.caption(f"System Instance: {id(system_assets)} | Threshold: {RETRAIN_THRESHOLD} | Status: Online")
