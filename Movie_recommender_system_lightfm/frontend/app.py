from streamlit import st
import requests

st.title("Movie Recommender")

option = st.radio(
    "Choose user type",
    ["Existing User", "New User"]
)

if option == "Existing User":

    user_id = st.number_input("User ID", min_value=1)

    if st.button("Recommend"):
        response = requests.get(
            f"http://localhost:8000/recommend/{user_id}"
        )
        st.write(response.json())


else:

    genres = st.multiselect(
        "Select Genres",
        ["Action","Comedy","Drama","Sci-Fi","Romance"]
    )

    if st.button("Recommend Movies"):

        response = requests.post(
            "http://localhost:8000/recommend-new-user",
            json=genres
        )

        st.write(response.json())