from fastapi import FastAPI
from utils.recommender import recommend_by_genre, recommend_movies

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Movie Recommendation API"}


@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):

    recs = recommend_movies(user_id)

    return {
        "user_id": user_id,
        "recommendations": recs
    }


@app.post("/recommend-new-user")
def recommend_new_user(genres: list[str]):

    movies = recommend_by_genre(genres)

    return {"recommendations": movies}