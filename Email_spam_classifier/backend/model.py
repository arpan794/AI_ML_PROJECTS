import joblib

model = joblib.load("backend/saved_models/model.pkl")

def predict(text):
    return model.predict([text])[0]