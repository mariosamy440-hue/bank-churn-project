
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model_local.pkl")

model = load_model()

st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a bank customer is likely to leave the bank.")

st.header("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score", 300, 900, 650
    )

    geography = st.selectbox(
        "Geography", ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    age = st.number_input(
        "Age", 18, 100, 35
    )

    tenure = st.number_input(
        "Tenure (Years)", 0, 10, 5
    )

with col2:
    balance = st.number_input(
        "Balance", min_value=0.0, value=50000.0
    )

    num_products = st.selectbox(
        "Number of Products", [1, 2, 3, 4]
    )

    has_card = st.selectbox(
        "Has Credit Card?", ["Yes", "No"]
    )

    active_member = st.selectbox(
        "Is Active Member?", ["Yes", "No"]
    )

    salary = st.number_input(
        "Estimated Salary", min_value=0.0, value=50000.0
    )

st.divider()

if st.button("Predict Churn", type="primary"):

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": salary
    }])

    prediction = model.predict(customer)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ The model predicts that the customer may leave the bank.")
    else:
        st.success("✅ The model predicts that the customer may stay with the bank.")