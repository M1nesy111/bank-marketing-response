import os
import joblib
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

_model = joblib.load(os.path.join(model_dir, "model.pkl"))
_scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
_columns = joblib.load(os.path.join(model_dir, "columns.pkl"))

RAW_FIELDS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]

print(len(RAW_FIELDS))

def build_features(customer: dict) -> pd.DataFrame:
    row = pd.DataFrame([customer], columns=RAW_FIELDS)
    encoded = pd.get_dummies(row)
    aligned = encoded.reindex(columns=_columns, fill_value=0)
    return aligned

def predict_one(customer: dict) -> float:
    features = build_features(customer)
    scaled = _scaler.transform(features)
    probability = _model.predict_proba(scaled)[0, 1]
    return float(probability)

if __name__ == "__main__":
    demo = {
        "age": 35, "job": "admin.", "marital": "married",
        "education": "university.degree", "default": "no",
        "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon",
        "campaign": 2, "pdays": 999, "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1, "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4, "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }
    print("认购概率:", predict_one(demo))