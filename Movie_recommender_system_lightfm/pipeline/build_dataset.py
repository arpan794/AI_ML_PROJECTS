from lightfm.data import Dataset


def build_dataset(df):

    dataset = Dataset()

    users = df["userId"].unique()
    items = df["movieId"].unique()

    # collect all genres
    genres = set()

    for g in df["genres"].str.split("|"):
        genres.update(g)

    dataset.fit(
        users=users,
        items=items,
        item_features=list(genres)
    )


    # interactions
    interactions, weights = dataset.build_interactions(
        [(row.userId, row.movieId, row.rating)
         for row in df.itertuples()]
    )


    # item features
    item_features = dataset.build_item_features(
        [(row.movieId, row.genres.split("|"))
         for row in df.itertuples()]
    )


    return dataset, interactions, item_features