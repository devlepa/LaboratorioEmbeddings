from __future__ import annotations

import json
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
# el codigo empaquetado en artifacts/code/src/.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_URI = str(_PROJECT_ROOT / "models" / "champion")
MODEL_URI = os.getenv("MODEL_URI", _DEFAULT_MODEL_URI)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
_model = None
_model_info: dict[str, Any] = {}


def _load_model_info() -> dict[str, Any]:
    """Lee los artefactos del modelo campeon y construye el dict de informacion."""
    model_dir = Path(MODEL_URI)
    mlmodel_path = model_dir / "MLmodel"
    metadata_path = model_dir / "artifacts" / "metadata.json"

    # -- MLmodel (run_id, model_id, fechas) --
    mlmodel: dict[str, Any] = {}
    if mlmodel_path.exists():
        import yaml  # incluido en mlflow
        mlmodel = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8"))

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    run_id: str = mlmodel.get("run_id", "")

    # -- Artefactos del run (metricas, config, dataset) --
    run_analysis = _PROJECT_ROOT / "mlruns" / "1" / run_id / "artifacts" / "analysis"
    test_report: dict[str, Any] = {}
    val_report: dict[str, Any] = {}
    experiment_config: dict[str, Any] = {}
    dataset_meta: dict[str, Any] = {}

    if run_analysis.exists():
        _tr = run_analysis / "test_classification_report.json"
        _vr = run_analysis / "validation_classification_report.json"
        _ec = run_analysis / "experiment_config.json"
        _dm = run_analysis / "dataset_metadata.json"
        if _tr.exists():
            test_report = json.loads(_tr.read_text(encoding="utf-8"))
        if _vr.exists():
            val_report = json.loads(_vr.read_text(encoding="utf-8"))
        if _ec.exists():
            experiment_config = json.loads(_ec.read_text(encoding="utf-8"))
        if _dm.exists():
            dataset_meta = json.loads(_dm.read_text(encoding="utf-8"))

    def _macro(report: dict) -> dict:
        m = report.get("macro avg", {})
        return {
            "accuracy": round(report.get("accuracy", 0), 4),
            "f1": round(m.get("f1-score", 0), 4),
            "precision": round(m.get("precision", 0), 4),
            "recall": round(m.get("recall", 0), 4),
        }

    def _per_class(report: dict) -> dict:
        out = {}
        for label, human in (("0", "negative"), ("1", "positive")):
            if label in report:
                c = report[label]
                out[human] = {
                    "precision": round(c.get("precision", 0), 4),
                    "recall": round(c.get("recall", 0), 4),
                    "f1": round(c.get("f1-score", 0), 4),
                    "support": int(c.get("support", 0)),
                }
        return out

    comparison = [
        {"model": "baseline_tfidf_logreg",       "test_f1": 0.8996, "selected": True},
        {"model": "dense_trainable_embeddings",   "test_f1": 0.8806, "selected": False},
        {"model": "conceptnet_finetuned",          "test_f1": 0.8804, "embedding_coverage": "32.62%", "selected": False},
        {"model": "spacy_lg_finetuned",            "test_f1": 0.8768, "embedding_coverage": "84.72%", "selected": False},
        {"model": "spacy_md_finetuned",            "test_f1": 0.8695, "embedding_coverage": "84.72%", "selected": False},
        {"model": "conceptnet_frozen",             "test_f1": 0.8014, "embedding_coverage": "32.62%", "selected": False},
        {"model": "spacy_md_frozen",               "test_f1": 0.7920, "embedding_coverage": "84.72%", "selected": False},
        {"model": "spacy_lg_frozen",               "test_f1": 0.7905, "embedding_coverage": "84.72%", "selected": False},
    ]

    return {
        "model_id": mlmodel.get("model_uuid", ""),
        "model_name": metadata.get("model_name", experiment_config.get("name", "")),
        "backend": metadata.get("backend", ""),
        "threshold": metadata.get("threshold", 0.5),
        "mlflow_run_id": run_id,
        "created_utc": mlmodel.get("utc_time_created", ""),
        "description": experiment_config.get("description", ""),
        "dataset": {
            "source": dataset_meta.get("dataset_handle", ""),
            "total_rows": dataset_meta.get("rows", 0),
            "class_distribution": dataset_meta.get("class_distribution", {}),
        },
        "test_metrics": _macro(test_report),
        "test_metrics_per_class": _per_class(test_report),
        "validation_metrics": _macro(val_report),
        "selection_criteria": {
            "metric": "test_f1 (macro avg)",
            "rationale": [
                "TF-IDF cubre el 100% del vocabulario del corpus; los embeddings preentrenados "
                "solo cubren entre el 32% (ConceptNet) y el 85% (spaCy lg), dejando tokens clave sin representacion.",
                "Las resenas de cine expresan opinion con vocabulario polarizado explicito "
                "(excelente, horrible, magnifico) que los n-gramas capturan directamente.",
                "Los modelos con embeddings congelados pierden hasta 10 puntos de F1 respecto "
                "al baseline por el alto porcentaje de tokens sin vector.",
                "El afinar embeddings mejora los modelos neurales pero no logra superar al "
                "baseline, lo que indica que la complejidad adicional no aporta valor en este dominio.",
                "El modelo baseline es determinista, mas rapido en inferencia y no requiere GPU.",
            ],
        },
        "experiment_comparison": comparison,
    }


class PredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Lista de resenas en espanol.")


class PredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_info
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    _model_info = _load_model_info()
    yield


app = FastAPI(title="IMDB Spanish Sentiment API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Informacion completa del modelo campeon: metricas, criterios de seleccion y comparacion."""
    return _model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="El modelo todavia no esta cargado.")

    frame = pd.DataFrame({"text": payload.texts})
    predictions = _model.predict(frame)
    return PredictionResponse(predictions=predictions.to_dict(orient="records"))
