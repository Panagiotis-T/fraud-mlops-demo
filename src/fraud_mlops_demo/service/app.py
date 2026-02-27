from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("artifacts/model.joblib")

app = FastAPI(title="Fraud MLOps Demo Service")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class Features(BaseModel):
    features: List[float]


@app.post("/predict")
def predict(payload: Features):
    model = get_model()
    preds = model.predict_proba([payload.features])[0][1]
    return {"fraud_score": float(preds)}