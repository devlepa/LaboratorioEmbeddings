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

# Importable tanto en desarrollo (src/) como desde artifact MLflow (code/src/)
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_URI = str(_PROJECT_ROOT / "models" / "champion")
MODEL_URI = os.getenv("MODEL_URI", _DEFAULT_MODEL_URI)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

_model = None
_model_info: dict[str, Any] = {}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _macro_metrics(report: dict) -> dict:
    m = report.get("macro avg", {})
    return {
        "accuracy": round(report.get("accuracy", 0), 4),
        "f1": round(m.get("f1-score", 0), 4),
        "precision": round(m.get("precision", 0), 4),
        "recall": round(m.get("recall", 0), 4),
    }


def _per_class_metrics(report: dict) -> dict:
    result = {}
    for label, name in (("0", "negative"), ("1", "positive")):
        if label in report:
            c = report[label]
            result[name] = {
                "precision": round(c.get("precision", 0), 4),
                "recall": round(c.get("recall", 0), 4),
                "f1": round(c.get("f1-score", 0), 4),
                "support": int(c.get("support", 0)),
            }
    return result


def _load_model_info() -> dict[str, Any]:
    model_dir = Path(MODEL_URI)

    # --- MLmodel (run_id, model_id, fecha de creacion) ---
    import yaml
    mlmodel = yaml.safe_load((model_dir / "MLmodel").read_text(encoding="utf-8"))
    run_id: str = mlmodel.get("run_id", "")

    # --- Artefactos empaquetados dentro del directorio del modelo ---
    analysis_dir = model_dir / "artifacts" / "analysis"
    metadata = _read_json(model_dir / "artifacts" / "metadata.json")
    test_report = _read_json(analysis_dir / "test_classification_report.json")
    val_report = _read_json(analysis_dir / "validation_classification_report.json")
    experiment_config = _read_json(analysis_dir / "experiment_config.json")
    dataset_meta = _read_json(analysis_dir / "dataset_metadata.json")
    metrics_summary = _read_json(analysis_dir / "metrics_summary.json")

    # --- Fallback: obtener metricas desde el servidor MLflow si no hay archivos locales ---
    if not metrics_summary and run_id and TRACKING_URI:
        try:
            client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
            run = client.get_run(run_id)
            metrics_summary = run.data.metrics
        except Exception:
            pass

    # --- Datos del test set ---
    test_metrics = _macro_metrics(test_report)
    if "test_roc_auc" in metrics_summary:
        test_metrics["roc_auc"] = round(metrics_summary["test_roc_auc"], 4)

    # --- Datos de validacion ---
    val_metrics = _macro_metrics(val_report)
    if "val_roc_auc" in metrics_summary:
        val_metrics["roc_auc"] = round(metrics_summary["val_roc_auc"], 4)

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
            "splits": {
                "train": dataset_meta.get("class_distribution", {}).get("train", {}),
                "validation": dataset_meta.get("class_distribution", {}).get("validation", {}),
                "test": dataset_meta.get("class_distribution", {}).get("test", {}),
            },
        },
        "test_metrics": test_metrics,
        "test_metrics_per_class": _per_class_metrics(test_report),
        "validation_metrics": val_metrics,
        "validation_metrics_per_class": _per_class_metrics(val_report),
        "selection_criteria": {
            "metric": "test_f1 (macro avg)",
            "rationale": [
                "TF-IDF cubre el 100% del vocabulario; los embeddings preentrenados solo cubren "
                "entre el 32% (ConceptNet) y el 85% (spaCy lg), dejando tokens clave sin representacion.",
                "Las resenas expresan opinion con vocabulario polarizado explicito "
                "(excelente, horrible, magnifico) que los n-gramas capturan directamente.",
                "Los modelos con embeddings congelados pierden hasta 10 puntos de F1 por el alto "
                "porcentaje de tokens sin vector.",
                "El afinar embeddings mejora los modelos neurales pero no supera al baseline, "
                "lo que indica que la complejidad adicional no aporta valor en este dominio.",
                "El modelo baseline es determinista, mas rapido en inferencia y no requiere GPU.",
            ],
        },
        "experiment_comparison": [
            {"model": "baseline_tfidf_logreg",     "test_f1": 0.8996, "test_roc_auc": 0.9626, "selected": True},
            {"model": "dense_trainable_embeddings", "test_f1": 0.8806, "test_roc_auc": None,   "selected": False},
            {"model": "conceptnet_finetuned",        "test_f1": 0.8804, "test_roc_auc": 0.9493, "embedding_coverage": "32.62%", "selected": False},
            {"model": "spacy_lg_finetuned",          "test_f1": 0.8768, "test_roc_auc": None,   "embedding_coverage": "84.72%", "selected": False},
            {"model": "spacy_md_finetuned",          "test_f1": 0.8695, "test_roc_auc": None,   "embedding_coverage": "84.72%", "selected": False},
            {"model": "conceptnet_frozen",           "test_f1": 0.8014, "test_roc_auc": 0.8873, "embedding_coverage": "32.62%", "selected": False},
            {"model": "spacy_md_frozen",             "test_f1": 0.7920, "test_roc_auc": None,   "embedding_coverage": "84.72%", "selected": False},
            {"model": "spacy_lg_frozen",             "test_f1": 0.7905, "test_roc_auc": None,   "embedding_coverage": "84.72%", "selected": False},
        ],
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
