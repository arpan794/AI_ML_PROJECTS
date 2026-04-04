import numpy as np
import joblib
import pandas as pd

model = joblib.load("models/lightfm_model.pkl")

movies = pd.read_csv("data/movies.csv")


def recommend_movies(user_id, n=10):

    n_items = len(movies)

    scores = model.predict(user_id, np.arange(n_items))

    top_items = np.argsort(-scores)[:n]

    return movies.iloc[top_items]["title"].tolist()


def recommend_by_genre(selected_genres, n=10):

    filtered_movies = movies[
        movies["genres"].apply(
            lambda x: any(g in x for g in selected_genres)
        )
    ]
	
    n = min(n, len(filtered_movies))

    return filtered_movies.sample(n)["title"].tolist()