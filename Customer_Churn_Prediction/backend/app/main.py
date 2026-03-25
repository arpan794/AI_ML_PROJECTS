from fastapi import FastAPI
from .schemas import CustomerData
from .model_loader import load_model
import pandas as pd

app = FastAPI(title="Customer Churn API")

model = load_model()

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob > 0.5)

    return {
        "churn_probability": float(prob),
        "prediction": prediction
    }