from __future__ import annotations

import os
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="IMDB Spanish Sentiment API", version="1.0.0")

MODEL_URI = os.getenv("MODEL_URI", "models:/imdb-spanish-sentiment@champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
_model = None


class PredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Lista de reseñas en español.")


class PredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]


@app.on_event("startup")
def startup_event() -> None:
    global _model
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
    _model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado.")

    frame = pd.DataFrame({"text": payload.texts})
    predictions = _model.predict(frame)
    return PredictionResponse(predictions=predictions.to_dict(orient="records"))
