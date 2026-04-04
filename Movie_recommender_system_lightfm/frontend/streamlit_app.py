import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL")
# API_URL = "http://localhost:8000/translate"

st.title("Movie Recommender")

option = st.radio(
    "Choose user type",
    ["Existing User", "New User"]
)

if option == "Existing User":

    user_id = st.number_input("User ID", min_value=1)

    if st.button("Recommend"):
        response = requests.get(
            f"{API_URL}/recommend/{user_id}"
        )
        st.write(response.json())


else:

    genres = st.multiselect(
        "Select Genres",
        ["Action","Comedy","Drama","Sci-Fi","Romance"]
    )

    if st.button("Recommend Movies"):

        response = requests.post(
            f"{API_URL}/recommend-new-user",
            json=genres
        )

        st.write(response.json())