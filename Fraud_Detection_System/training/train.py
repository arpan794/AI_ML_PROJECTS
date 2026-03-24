import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# Load dataset
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "xgboost": xgb.XGBClassifier(eval_metric="logloss")
}

best_model = None
best_score = 0

for name, model in models.items():

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)

    print(f"{name} ROC-AUC:", score)

    if score > best_score:
        best_score = score
        best_model = pipeline

joblib.dump(best_model, "../model/fraud_model.pkl")
print("Best model saved.")