from fastapi import FastAPI
from app.schemas import Transaction
from app.model_loader import load_model
import numpy as np

app = FastAPI(title="Fraud Detection API")

model = load_model()

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: Transaction):

    input_array = np.array(data.features).reshape(1, -1)
    prob = model.predict_proba(input_array)[0][1]
    prediction = int(prob > 0.5)

    return {
        "fraud_probability": float(prob),
        "prediction": prediction
    }