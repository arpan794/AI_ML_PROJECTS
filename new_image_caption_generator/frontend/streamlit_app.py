import streamlit as st
import requests

API_URL = "http://localhost:8000/generate-caption"

st.title("Image Caption Generator")

file = st.file_uploader("Upload Image")

if file:

    st.image(file)

    if st.button("Generate Caption"):

        files = {"file": file.getvalue()}

        response = requests.post(API_URL, files=files)

        caption = response.json()["caption"]

        st.success(caption)