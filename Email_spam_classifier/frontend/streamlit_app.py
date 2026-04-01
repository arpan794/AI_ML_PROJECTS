import streamlit as st
import requests

st.title("Email/SMS Spam Classifier")

text = st.text_area("Enter message")

if st.button("Predict"):
    response = requests.post(
        "http://backend:8000/predict",
        json={"text": text}
    )
    result = response.json()
    st.success(result["prediction"])