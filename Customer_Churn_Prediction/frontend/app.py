import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 1, 72)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")

if st.button("Predict"):
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "InternetService": InternetService,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.write("Churn Probability:", result["churn_probability"])

        if result["prediction"] == 1:
            st.error("Customer Likely to Churn ❌")
        else:
            st.success("Customer Likely to Stay ✅")