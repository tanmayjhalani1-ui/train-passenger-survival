from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
FEATURES = ["Age", "Fare", "Gender", "SeatClass"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact is missing. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict_one(model, age: float, fare: float, gender: str, seat_class: str) -> dict:
    if age <= 0:
        raise ValueError("Age must be greater than zero.")
    if fare < 0:
        raise ValueError("Fare cannot be negative.")
    if gender not in {"Female", "Male"}:
        raise ValueError("Gender must be Female or Male.")
    if seat_class not in {"First", "Second", "Third"}:
        raise ValueError("Seat class must be First, Second, or Third.")
    row = pd.DataFrame([{
        "Age": age,
        "Fare": fare,
        "Gender": gender,
        "SeatClass": seat_class,
    }], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "Survived" if prediction else "Did not survive",
        "survival_probability": probability,
    }
