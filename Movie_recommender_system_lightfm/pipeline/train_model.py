import joblib
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

from preprocess import load_data
from build_dataset import build_dataset


df = load_data()

dataset, interactions, item_features = build_dataset(df)


model = LightFM(
    loss="warp",
    no_components=100,
    learning_rate=0.05
)

model.fit(
    interactions,
    item_features=item_features,
    epochs=100,
    num_threads=4
)


precision = precision_at_k(model, interactions, item_features=item_features, k=10).mean()

print("Precision@10:", precision)


joblib.dump(model, "models/lightfm_model.pkl")

print("Model saved")