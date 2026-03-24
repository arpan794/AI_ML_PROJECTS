import joblib

def load_model():
    return joblib.load("model/fraud_model.pkl")