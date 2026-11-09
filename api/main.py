from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_model, predict_one

app = FastAPI(
    title="Passenger Survival Prediction API",
    description="A small demonstration API backed by the repository's trained scikit-learn pipeline.",
    version="1.0.0",
)
model = None


class PassengerInput(BaseModel):
    age: float = Field(gt=0, le=120)
    fare: float = Field(ge=0)
    gender: Literal["Female", "Male"]
    seat_class: Literal["First", "Second", "Third"]


@app.on_event("startup")
def load_artifact():
    global model
    try:
        model = load_model()
    except FileNotFoundError:
        model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: PassengerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained. Run python -m src.train first.")
    try:
        return predict_one(
            model, payload.age, payload.fare, payload.gender, payload.seat_class
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
