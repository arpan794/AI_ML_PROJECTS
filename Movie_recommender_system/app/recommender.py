import joblib
import pandas as pd

class Recommender:

    def __init__(self):
        self.model = joblib.load("model/svd_model.pkl")
        self.ratings = pd.read_csv("ratings.csv")

    def recommend_movies(self, user_id, n=5):

        all_movies = self.ratings["movieId"].unique()
        rated_movies = self.ratings[self.ratings["userId"] == user_id]["movieId"]

        movies_to_predict = [m for m in all_movies if m not in rated_movies.values]

        predictions = []
        for movie in movies_to_predict:
            pred = self.model.predict(user_id, movie)
            predictions.append((movie, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]