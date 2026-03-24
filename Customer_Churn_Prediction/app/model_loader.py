import joblib

def load_model():
    return joblib.load("model/churn_model.pkl")