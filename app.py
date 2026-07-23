import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Set page layout
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")

st.title("💳 Credit Risk & Default Prediction Engine")
st.write("Enter applicant details below to evaluate credit default probability using a trained XGBoost ensemble model.")

# Load saved model and features
@st.cache_resource
def load_model():
    model = joblib.load('credit_risk_xgboost.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, feature_names = load_model()

# User Input Form
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    month_duration = st.slider("Loan Duration (Months)", 4, 72, 24)
    credit_amount = st.number_input("Credit Amount ($)", min_value=250, max_value=20000, value=3000)
    age = st.slider("Applicant Age", 18, 75, 35)

with col2:
    payment_ratio = st.slider("Payment to Income Ratio (1-4)", 1, 4, 2)
    existing_credits = st.selectbox("Number of Existing Credits", [1, 2, 3, 4])
    checking_status = st.selectbox("Checking Account Status", ["No Account", "< 0 DM", "0 to 200 DM", ">= 200 DM"])

# Prediction Button
if st.button("Evaluate Credit Risk", type="primary"):
    # Build feature dictionary matching trained columns
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Assign numerical inputs
    input_data['month_duration'] = month_duration
    input_data['credit_amount'] = credit_amount
    input_data['age'] = age
    input_data['payment_to_income_ratio'] = payment_ratio
    input_data['n_credits'] = existing_credits
    
    # Handle checking status dummy mapping
    if checking_status == "No Account" and "status_account_no checking account" in input_data.columns:
        input_data["status_account_no checking account"] = 1
    elif checking_status == "< 0 DM" and "status_account__ 0 DM" in input_data.columns:
        input_data["status_account__ 0 DM"] = 1

    # Get Prediction
    prob = model.predict_proba(input_data)[0][1]
    
    st.subheader("Risk Assessment Result")
    
    # Visual Output
    if prob >= 0.45:
        st.error(f"⚠️ **HIGH RISK / DEFAULT LIKELY**\n\nDefault Probability: **{prob * 100:.1f}%**")
    else:
        st.success(f"✅ **LOW RISK / APPROVED**\n\nDefault Probability: **{prob * 100:.1f}%**")

    st.progress(float(prob))
