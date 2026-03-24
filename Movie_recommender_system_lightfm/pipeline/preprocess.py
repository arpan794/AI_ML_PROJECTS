import pandas as pd

def load_data():

    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    df = ratings.merge(movies, on="movieId")

    return df