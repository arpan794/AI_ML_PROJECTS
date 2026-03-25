import os
import joblib

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "churn_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    return joblib.load(model_path)