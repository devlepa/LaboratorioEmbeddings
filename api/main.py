from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Asegura que el paquete imdb_sentiment sea importable tanto en desarrollo
# (carga desde src/) como al servir desde un artifact MLflow que incluye
# el código empaquetado en artifacts/code/src/.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DEFAULT_MODEL_URI = str(Path(__file__).resolve().parents[1] / "models" / "champion")
MODEL_URI = os.getenv("MODEL_URI", _DEFAULT_MODEL_URI)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
_model = None


class PredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Lista de reseñas en español.")


class PredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    yield


app = FastAPI(title="IMDB Spanish Sentiment API", version="1.0.0", lifespan=lifespan)


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
