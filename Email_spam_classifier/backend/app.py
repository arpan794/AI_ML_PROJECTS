from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Classifier API"}

@app.post("/predict")
def predict_spam(data: InputText):
    result = predict(data.text)
    return {
        "prediction": "Spam" if result == 1 else "Not Spam"
    }