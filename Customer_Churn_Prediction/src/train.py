import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load dataset (example telecom dataset)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "xgboost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

best_model = None
best_score = 0

for name, model in models.items():

    pipeline = ImbPipeline([
        ("preprocessing", preprocessor),
        ("smote", SMOTE()),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)

    print(f"{name} ROC-AUC:", score)

    if score > best_score:
        best_score = score
        best_model = pipeline

MODEL_DIR = "/kaggle/working/model"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)