from fastapi import FastAPI
from app.schemas import UserRequest
from app.recommender import Recommender

app = FastAPI(title="Recommendation API")

recommender = Recommender()

@app.get("/")
def home():
    return {"message": "Recommendation System Running"}

@app.post("/recommend")
def recommend(data: UserRequest):

    recommendations = recommender.recommend_movies(data.user_id)

    return {
        "user_id": data.user_id,
        "recommendations": recommendations
    }