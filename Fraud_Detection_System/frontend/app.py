import streamlit as st
import requests
import numpy as np

API_URL = "http://localhost:8000/predict"

st.title("💳 Fraud Detection System")

st.write("Enter transaction feature values:")

features = []
for i in range(30):  # creditcard dataset has 30 features
    value = st.number_input(f"Feature {i}", value=0.0)
    features.append(value)

if st.button("Predict Fraud"):
    response = requests.post(API_URL, json={"features": features})

    if response.status_code == 200:
        result = response.json()

        st.write("Fraud Probability:", result["fraud_probability"])

        if result["prediction"] == 1:
            st.error("Fraudulent Transaction 🚨")
        else:
            st.success("Legitimate Transaction ✅")