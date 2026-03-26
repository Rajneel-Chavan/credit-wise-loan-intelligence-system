import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import shap
import numpy as np

# --- SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Credit Wise", layout="wide")

st.title("Credit Wise Loan Intelligence System")

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("loan_approval_data.csv")
    logger.info("Data loaded successfully")

    categorical_col = df.select_dtypes(include=["object"]).columns
    numerical_col = df.select_dtypes(include=["float64", "int64"]).columns

    df[numerical_col] = SimpleImputer(strategy="mean").fit_transform(df[numerical_col])
    df[categorical_col] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_col])

    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)

    df["Education_Level"] = df["Education_Level"].map({"Graduate": 1, "Not Graduate": 0})

    le = LabelEncoder()
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

    cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)

    df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

    return df, le, ohe, cols

df, le, ohe, cols = load_and_preprocess_data()

# --- ML PIPELINE ---
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

low_income_threshold = df["Applicant_Income"].quantile(0.25)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training Logistic Regression
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)
log_model = grid_search.best_estimator_

# Training Ensemble Models
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
gb_model = GradientBoostingClassifier()

log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)

# --- CRITICAL: MODEL DICTIONARY ---
model_dict = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

def evaluate(model, X_t):
    pred = model.predict(X_t)
    pred_proba = model.predict_proba(X_t)[:, 1] if hasattr(model, 'predict_proba') else None
    roc_auc = round(roc_auc_score(y_test, pred_proba), 3) if pred_proba is not None else 'N/A'
    return [
        round(accuracy_score(y_test, pred), 3),
        round(precision_score(y_test, pred), 3),
        round(recall_score(y_test, pred), 3),
        round(f1_score(y_test, pred), 3),
        roc_auc
    ]

results = pd.DataFrame([
    ["Logistic Regression"] + evaluate(log_model, X_test_scaled),
    ["Random Forest"] + evaluate(rf_model, X_test_scaled),
    ["Gradient Boosting"] + evaluate(gb_model, X_test_scaled)
], columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])

best_model_name = results.loc[results["F1"].idxmax(), "Model"]

# --- SIDEBAR NAVIGATION ---
menu = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"])

# --- DASHBOARD PAGE ---
if menu == "Dashboard":
    st.header("📊 Executive Portfolio Overview")
    st.markdown("Detailed analytics of the loan application landscape and model efficacy.")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    approval_rate = df["Loan_Approved"].mean() * 100
    avg_dti = df["DTI_Ratio"].mean()
    high_value_applicants = len(df[df["Applicant_Income"] > df["Applicant_Income"].quantile(0.75)])
    avg_credit = df["Credit_Score"].mean()

    mcol1.metric("Approval Rate", f"{approval_rate:.1f}%", delta=f"{approval_rate - 50:.1f}% vs Target")
    mcol2.metric("Portfolio Risk (Avg DTI)", f"{avg_dti:.2f}")
    mcol3.metric("High-Value Leads", high_value_applicants)
    mcol4.metric("Avg. Credit Score", f"{avg_credit:.0f}")

    st.markdown("---")
    st.subheader("📍 Applicant Profile Insights")
    lcol1, lcol2 = st.columns([1, 1.2])

    with lcol1:
        st.write("**Approval Status Split**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        colors = ["#ff4b4b", "#00cc96"]
        pie_data = df["Loan_Approved"].value_counts().sort_index()
        ax1.pie(pie_data, labels=["Rejected", "Approved"], autopct="%1.1f%%", startangle=140, colors=colors)
        st.pyplot(fig1)

    with lcol2:
        st.write("**Income vs. Credit Score Correlation**")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x="Applicant_Income", y="Credit_Score", hue="Loan_Approved", palette={0: "#ff4b4b", 1: "#00cc96"}, alpha=0.6, ax=ax2)
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🤖 Model Intelligence & Technical Audit")
    tab_metrics, tab_correlations, tab_curves = st.tabs(["Metric Table", "Feature Correlation", "ROC Curves"])

    with tab_metrics:
        st.dataframe(results.style.highlight_max(axis=0, subset=['Accuracy', 'F1', 'ROC-AUC'], color='#D4EDDA'), use_container_width=True)

    with tab_correlations:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.iloc[:, :10].corr(), annot=True, cmap="RdYlGn", fmt=".2f", ax=ax3)
        st.pyplot(fig3)

    with tab_curves:
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        for model_name, model_obj in model_dict.items():
            fpr, tpr, _ = roc_curve(y_test, model_obj.predict_proba(X_test_scaled)[:, 1])
            ax4.plot(fpr, tpr, label=f"{model_name} (AUC: {roc_auc_score(y_test, model_obj.predict_proba(X_test_scaled)[:, 1]):.2f})")
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.legend()
        st.pyplot(fig4)

# --- PREDICTION PAGE ---
if menu == "Prediction":
    st.subheader("Applicant Details")
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        employment = st.selectbox("Employment", ["Employed", "Unemployed", "Self-employed"])
        marital = st.selectbox("Marital Status", ["Single", "Married"])
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business"])

    with col2:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        employer_cat = st.selectbox("Employer Category", ["Private", "Government", "Self-employed"])

    st.markdown("---")
    st.subheader("Financial Details")
    col3, col4 = st.columns(2)

    with col3:
        income = st.number_input("Applicant Income (USD)", 0)
        credit_score = st.number_input("Credit Score", 0)
        savings = st.number_input("Savings (USD)", 0)

    with col4:
        co_income = st.number_input("Coapplicant Income (USD)", 0)
        dti = st.number_input("DTI Ratio", 0.0)
        loan_amount = st.number_input("Loan Amount (USD)", 0)

    age = st.number_input("Age", 18)

    st.markdown("---")
    show_debug = st.checkbox("Show debug details (input + model predictions)")
    st.info("Fill applicant and financial details (all income amounts in USD) to predict loan approval")
    st.markdown("---")

    if st.button("Predict Loan Status"):
        errors = []
        if income <= 0: errors.append("Applicant Income must be greater than 0")
        if credit_score < 300 or credit_score > 850: errors.append("Credit Score must be between 300 and 850")
        if age < 18: errors.append("Age must be at least 18")
        if dti < 0 or dti > 1: errors.append("DTI Ratio must be between 0 and 1")

        if errors:
            for error in errors: st.error(error)
        else:
            logger.info(f"Prediction requested with best model: {best_model_name}")
            with st.spinner("Analyzing applicant profile..."):
                input_df = pd.DataFrame({
                    "Education_Level": [1 if education == "Graduate" else 0],
                    "Employment_Status": [employment], "Marital_Status": [marital],
                    "Loan_Purpose": [loan_purpose], "Property_Area": [property_area],
                    "Gender": [gender], "Employer_Category": [employer_cat],
                    "Applicant_Income": [income], "Coapplicant_Income": [co_income],
                    "Credit_Score": [credit_score], "DTI_Ratio": [dti],
                    "Savings": [savings], "Age": [age], "Loan_Amount": [loan_amount]
                })

                encoded = ohe.transform(input_df[cols])
                encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols))
                input_df = pd.concat([input_df.drop(columns=cols), encoded_df], axis=1)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)
                input_scaled = scaler.transform(input_df)

                # Prediction Logic
                selected_model = model_dict.get(best_model_name, log_model)
                pred = selected_model.predict(input_scaled)[0]
                prob = selected_model.predict_proba(input_scaled)[0, 1] if hasattr(selected_model, 'predict_proba') else 0.5

                if show_debug:
                    st.subheader("Debug details")
                    st.dataframe(input_df)
                    for n, m in model_dict.items():
                        st.write(f"{n} prediction:", int(m.predict(input_scaled)[0]))

                # --- NEW NARRATIVE OUTPUT SECTION ---
                st.markdown("---")
                st.header("Personalized Credit Assessment")

                res_col1, res_col2 = st.columns([2, 1])
                with res_col1:
                    if pred == 1:
                        st.success("### APPLICATION APPROVED")
                        st.balloons()
                    else:
                        st.error("### APPLICATION DECLINED")
                with res_col2:
                    st.metric("Approval Confidence", f"{prob*100:.1f}%")

                st.markdown("---")
                with st.spinner("Analyzing decision factors..."):
                    if "Logistic" in best_model_name:
                        explainer = shap.LinearExplainer(log_model, X_train_scaled)
                        shap_values = explainer.shap_values(input_scaled)[0]
                    else:
                        explainer = shap.TreeExplainer(selected_model)
                        sv = explainer.shap_values(input_scaled)
                        shap_values = sv[1][0] if isinstance(sv, list) else sv[0]

                    impact_series = pd.Series(shap_values, index=X.columns).sort_values(ascending=False)
                    top_pos = impact_series.index[0].replace('_', ' ')
                    top_neg = impact_series.index[-1].replace('_', ' ')

                    st.subheader("Decision Narrative")
                    if pred == 1:
                        st.write(f"Approval was primarily driven by your strong **{top_pos}**. This significantly outweighed other risks.")
                    else:
                        st.write(f"The decision to decline was heavily influenced by your **{top_neg}**.")
                        st.warning(f"💡 **Actionable Advice:** Improving your {top_neg} could shift your probability significantly.")

                with st.expander("View Detailed Mathematical Impact"):
                    plot_data = pd.concat([impact_series.head(5), impact_series.tail(5)]).sort_values()
                    fig_s, ax_s = plt.subplots(figsize=(10, 6))
                    colors = ['#ff4b4b' if x < 0 else '#00cc96' for x in plot_data]
                    plot_data.plot(kind='barh', color=colors, ax=ax_s)
                    ax_s.set_yticklabels([l.get_text().replace('_', ' ') for l in ax_s.get_yticklabels()])
                    st.pyplot(fig_s)

                st.caption("Disclaimer: AI-generated analysis based on historical patterns.")
