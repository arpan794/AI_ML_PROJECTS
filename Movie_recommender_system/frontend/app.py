import streamlit as st
import requests

API_URL = "http://localhost:8000/recommend"

st.title("🎬 Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Get Recommendations"):

    response = requests.post(API_URL, json={"user_id": user_id})

    if response.status_code == 200:
        result = response.json()
        st.write("Top Recommendations:")

        for movie in result["recommendations"]:
            st.write(f"Movie ID: {movie[0]} | Predicted Rating: {round(movie[1], 2)}")