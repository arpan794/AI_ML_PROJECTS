import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

model = joblib.load("../model/fraud_model.pkl")
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

preds = model.predict(X)
cm = confusion_matrix(y, preds)

tn, fp, fn, tp = cm.ravel()

# Define business costs
cost_fn = 1000  # Missing fraud
cost_fp = 10    # Investigating normal transaction

total_cost = (fn * cost_fn) + (fp * cost_fp)

print("Confusion Matrix:\n", cm)
print("Total Business Cost:", total_cost)