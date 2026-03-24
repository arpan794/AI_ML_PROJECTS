import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load ratings dataset
df = pd.read_csv("ratings.csv")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.2)

# Matrix Factorization (SVD)
model = SVD()
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

print("RMSE:", rmse)

joblib.dump(model, "../model/svd_model.pkl")
print("Model Saved")