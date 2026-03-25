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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Credit Wise", layout="wide")

st.title("Credit Wise Loan Intelligence System")

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("loan_approval_data.csv")
    logger.info("Data loaded successfully")

    categorical_col = df.select_dtypes(include=["object"]).columns
    numerical_col = df.select_dtypes(include=["float64"]).columns

    df[numerical_col] = SimpleImputer(strategy="mean").fit_transform(df[numerical_col])
    df[categorical_col] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_col])

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

X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

low_income_threshold = df["Applicant_Income"].quantile(0.25)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)
log_model = grid_search.best_estimator_
logger.info(f"Best Logistic Regression params: {grid_search.best_params_}")

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
gb_model = GradientBoostingClassifier()

log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)

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

menu = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.subheader("Model Comparison")
    st.dataframe(results.style.highlight_max(axis=0), use_container_width=True)
    st.markdown("---")

    best_f1_model = results.loc[results["F1"].idxmax(), "Model"]
    st.success(f"Best Model: {best_f1_model}")
    st.markdown("---")

    approval_rate = df["Loan_Approved"].mean() * 100
    avg_income = df["Applicant_Income"].mean()
    avg_credit = df["Credit_Score"].mean()
    low_income_quantile = df["Applicant_Income"].quantile(0.25)

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Approval Rate", f"{approval_rate:.2f}%")
    mcol2.metric("Average Income", f"${avg_income:,.0f}")
    mcol3.metric("Average Credit Score", f"{avg_credit:.0f}")

    st.caption(f"Low income alert threshold: ${low_income_quantile:,.0f} (25th percentile)")

    st.markdown("---")
    st.subheader("Loan Distribution")

    lcol1, lcol2 = st.columns(2)
    with lcol1:
        fig1, ax1 = plt.subplots(figsize=(4, 3.5))
        pie_data = df["Loan_Approved"].map({1: "Approved", 0: "Rejected"}).value_counts()
        pie_data.plot.pie(autopct="%1.1f%%", ax=ax1, colors=["blue", "orange"])
        ax1.set_ylabel("")
        ax1.legend(title="Loan Status")
        st.pyplot(fig1)

    with lcol2:
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        sns.histplot(df["Applicant_Income"], bins=20, ax=ax2)
        ax2.set_title("Applicant Income")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Model Performance")

    pmcol1, pmcol2 = st.columns(2)
    with pmcol1:
        fig5, ax5 = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_test, log_model.predict(X_test_scaled))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_xlabel("Predicted")
        ax5.set_ylabel("Actual")
        ax5.set_title("Confusion Matrix (Logistic)")
        st.pyplot(fig5)

    with pmcol2:
        fig4, ax4 = plt.subplots(figsize=(4, 3.5))
        for model, name in [(log_model, "Logistic Regression"), (rf_model, "Random Forest"), (gb_model, "Gradient Boosting")]:
            if hasattr(model, 'predict_proba'):
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
                ax4.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.3f})")
        ax4.plot([0, 1], [0, 1], 'k--')
        ax4.set_xlabel("False Positive Rate")
        ax4.set_ylabel("True Positive Rate")
        ax4.legend(fontsize='small')
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("Feature Relationships")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ---------------- PREDICTION ----------------
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
        income = st.number_input("Applicant Income", 0)
        credit_score = st.number_input("Credit Score", 0)
        savings = st.number_input("Savings", 0)

    with col4:
        co_income = st.number_input("Coapplicant Income", 0)
        dti = st.number_input("DTI Ratio", 0.0)
        loan_amount = st.number_input("Loan Amount", 0)

    age = st.number_input("Age", 18)

    st.markdown("---")

    show_debug = st.checkbox("Show debug details (input + model predictions)")
    st.info("Fill applicant and financial details to predict loan approval")
    st.markdown("---")

    if st.button("Predict Loan Status"):
        errors = []
        if income <= 0:
            errors.append("Applicant Income must be greater than 0")
        if credit_score < 300 or credit_score > 850:
            errors.append("Credit Score must be between 300 and 850")
        if age < 18:
            errors.append("Age must be at least 18")
        if dti < 0 or dti > 1:
            errors.append("DTI Ratio must be between 0 and 1")

        if errors:
            for error in errors:
                st.error(error)
        else:
            logger.info(f"Prediction requested with best model: {best_model_name}")
            with st.spinner("Analyzing applicant profile..."):
                input_df = pd.DataFrame({
                    "Education_Level": [1 if education == "Graduate" else 0],
                    "Employment_Status": [employment],
                    "Marital_Status": [marital],
                    "Loan_Purpose": [loan_purpose],
                    "Property_Area": [property_area],
                    "Gender": [gender],
                    "Employer_Category": [employer_cat],
                    "Applicant_Income": [income],
                    "Coapplicant_Income": [co_income],
                    "Credit_Score": [credit_score],
                    "DTI_Ratio": [dti],
                    "Savings": [savings],
                    "Age": [age],
                    "Loan_Amount": [loan_amount]
                })

                encoded = ohe.transform(input_df[cols])
                encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols))

                input_df = pd.concat([input_df.drop(columns=cols), encoded_df], axis=1)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)

                input_scaled = scaler.transform(input_df)

                model_map = {
                    "Logistic Regression": log_model,
                    "Random Forest": rf_model,
                    "Gradient Boosting": gb_model,
                }

                selected_model = model_map.get(best_model_name, log_model)
                pred = selected_model.predict(input_scaled)[0]

                if hasattr(selected_model, 'predict_proba'):
                    prob = selected_model.predict_proba(input_scaled)[0, 1]
                else:
                    prob = None

                if show_debug:
                    st.subheader("Debug details")
                    st.write("Model used:", best_model_name)
                    st.write("Encoded input:")
                    st.dataframe(input_df)
                    for n, m in model_map.items():
                        if hasattr(m, 'predict_proba'):
                            st.write(f"{n} prediction:", int(m.predict(input_scaled)[0]), "proba", float(m.predict_proba(input_scaled)[0, 1]))
                        else:
                            st.write(f"{n} prediction:", int(m.predict(input_scaled)[0]), "proba: N/A")

                st.markdown("## Final Decision")
                st.markdown("---")

                center1, center2, center3 = st.columns([1, 2, 1])
                with center2:
                    if pred == 1:
                        st.success("Loan Approved")
                    else:
                        st.error("Loan Rejected")

                    reasons = []
                    if credit_score < 600:
                        reasons.append("Low credit score")
                    if income < low_income_threshold:
                        reasons.append("Low income for this portfolio")
                    if dti > 0.4:
                        reasons.append("High DTI ratio")

                    if len(reasons) == 0:
                        reasons_text = "Strong financial profile with manageable risk."
                    else:
                        reasons_text = ", ".join(reasons)

                    st.markdown(f"**Reason:** {reasons_text}")

                st.markdown("---")