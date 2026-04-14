#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


def _require_remote_tracking_uri(uri: str | None) -> str:
    """Aborta si MLFLOW_TRACKING_URI no apunta a un servidor HTTP/HTTPS remoto."""
    if not uri:
        sys.exit(
            "ERROR: MLFLOW_TRACKING_URI no está configurado.\n"
            "Exportá la variable antes de ejecutar:\n"
            "  export MLFLOW_TRACKING_URI=http://<host>:5000"
        )
    if not uri.startswith(("http://", "https://")):
        sys.exit(
            f"ERROR: MLFLOW_TRACKING_URI='{uri}' no es una URL HTTP/HTTPS remota.\n"
            "El entrenamiento solo puede ejecutarse apuntando al servidor MLflow remoto.\n"
            "Corregí la variable:\n"
            "  export MLFLOW_TRACKING_URI=http://<host>:5000"
        )
    return uri

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imdb_sentiment.config import ExperimentConfig, load_config
from imdb_sentiment.data import prepare_dataset_splits
from imdb_sentiment.embeddings import build_embedding_matrix
from imdb_sentiment.evaluate import binary_metrics, save_evaluation_artifacts, save_history_plot, save_model_summary
from imdb_sentiment.modeling import adapt_vectorizer, build_baseline_pipeline, build_neural_model, fit_neural_model
from imdb_sentiment.tracking import log_deployable_model


def _save_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def flatten_mapping(payload: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in payload.items():
        composite_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_mapping(value, composite_key))
        elif isinstance(value, (list, tuple)):
            flat[composite_key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[composite_key] = value
    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena todos los experimentos de sentimientos sobre IMDB en español.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Ruta del archivo YAML de configuración.")
    parser.add_argument("--tracking-uri", default=None, help="URI de MLflow. Si se omite, usa MLFLOW_TRACKING_URI.")
    parser.add_argument("--only", nargs="*", default=None, help="Lista opcional de experimentos a ejecutar.")
    return parser.parse_args()


def _selected_experiments(experiments: list[ExperimentConfig], only: list[str] | None) -> list[ExperimentConfig]:
    if not only:
        return experiments
    requested = set(only)
    return [experiment for experiment in experiments if experiment.name in requested]


def _save_text_lines(lines: list[str], path: Path) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _save_baseline_coefficients(model, path: Path, top_n: int = 25) -> Path:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    ordered = pd.concat([frame.nsmallest(top_n, "coefficient"), frame.nlargest(top_n, "coefficient")], ignore_index=True)
    ordered.to_csv(path, index=False)
    return path


def _common_run_payload(config, experiment: ExperimentConfig, split) -> dict[str, object]:
    payload = {
        "experiment": asdict(experiment),
        "dataset": {
            "handle": config.dataset.handle,
            "text_column": split.text_column,
            "label_column": split.label_column,
        },
        "split": asdict(config.splits),
    }
    if experiment.type == "baseline":
        payload["baseline"] = asdict(config.baseline)
    else:
        payload["neural"] = asdict(config.neural)
    return payload


def run_baseline(config, experiment: ExperimentConfig, split, summary_rows: list[dict[str, object]]) -> None:
    pipeline = build_baseline_pipeline(config.baseline)
    pipeline.fit(split.x_train, split.y_train)

    val_probabilities = pipeline.predict_proba(split.x_val)[:, 1]
    test_probabilities = pipeline.predict_proba(split.x_test)[:, 1]
    val_metrics, val_predictions = binary_metrics(split.y_val, val_probabilities, threshold=config.neural.threshold, prefix="val")
    test_metrics, test_predictions = binary_metrics(split.y_test, test_probabilities, threshold=config.neural.threshold, prefix="test")
    metrics = val_metrics | test_metrics

    mlflow.set_tags(
        {
            "task": "sentiment-analysis",
            "dataset": config.dataset.handle,
            "model_family": "baseline",
            "embedding_mode": "tfidf",
            "run_name": experiment.name,
        }
    )
    mlflow.log_params(flatten_mapping(_common_run_payload(config, experiment, split)))
    mlflow.log_metrics(metrics)

    with tempfile.TemporaryDirectory() as temp_dir:
        analysis_dir = Path(temp_dir) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        save_json(split.dataset_metadata, analysis_dir / "dataset_metadata.json")
        save_json(asdict(experiment), analysis_dir / "experiment_config.json")
        save_evaluation_artifacts(split.y_val, val_probabilities, val_predictions, analysis_dir, prefix="validation")
        save_evaluation_artifacts(split.y_test, test_probabilities, test_predictions, analysis_dir, prefix="test")
        _save_baseline_coefficients(pipeline, analysis_dir / "top_tfidf_coefficients.csv")
        mlflow.log_artifacts(str(analysis_dir), artifact_path="analysis")

    model_metadata = {
        "threshold": config.neural.threshold,
        "backend": "sklearn",
        "model_name": experiment.name,
    }
    try:
        log_deployable_model(pipeline, backend="sklearn", artifact_path="model", metadata=model_metadata)
    except Exception as exc:
        mlflow.set_tag("model_artifact_upload_error", str(exc)[:500])
        print(f"WARNING: No se pudo subir el artefacto del modelo ({experiment.name}): {exc}", file=sys.stderr)

    summary_rows.append(
        {
            "run_id": mlflow.active_run().info.run_id,
            "experiment_name": experiment.name,
            "model_family": "baseline",
            "test_f1": metrics["test_f1"],
            "test_roc_auc": metrics["test_roc_auc"],
        }
    )


def run_neural(config, experiment: ExperimentConfig, split, summary_rows: list[dict[str, object]]) -> None:
    vectorizer = adapt_vectorizer(split.x_train, config.neural)
    vocabulary = vectorizer.get_vocabulary()

    pretrained_matrix = None
    embedding_artifacts: dict[str, object] = {}
    if experiment.embedding and experiment.embedding.kind == "pretrained":
        pretrained_matrix = build_embedding_matrix(vocabulary, experiment.embedding)
        embedding_artifacts = {
            "coverage": pretrained_matrix.coverage,
            "covered_tokens": pretrained_matrix.covered_tokens,
            "total_tokens": pretrained_matrix.total_tokens,
            "missing_tokens_sample": pretrained_matrix.missing_tokens,
            "source_name": pretrained_matrix.source_name,
        }

    neural_artifacts = build_neural_model(
        config=config.neural,
        vectorizer=vectorizer,
        embedding_config=experiment.embedding,
        pretrained_matrix=pretrained_matrix,
    )
    history = fit_neural_model(
        neural_artifacts.model,
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        config.neural,
    )

    val_probabilities = np.asarray(neural_artifacts.model.predict(split.x_val, verbose=0)).reshape(-1)
    test_probabilities = np.asarray(neural_artifacts.model.predict(split.x_test, verbose=0)).reshape(-1)
    val_metrics, val_predictions = binary_metrics(split.y_val, val_probabilities, threshold=config.neural.threshold, prefix="val")
    test_metrics, test_predictions = binary_metrics(split.y_test, test_probabilities, threshold=config.neural.threshold, prefix="test")
    metrics = val_metrics | test_metrics

    if pretrained_matrix is not None:
        metrics["embedding_coverage"] = float(pretrained_matrix.coverage)

    mlflow.set_tags(
        {
            "task": "sentiment-analysis",
            "dataset": config.dataset.handle,
            "model_family": "neural",
            "embedding_mode": experiment.embedding.kind if experiment.embedding else "trainable",
            "embedding_provider": experiment.embedding.provider if experiment.embedding else "trainable",
            "embedding_trainable": str(experiment.embedding.trainable if experiment.embedding else True),
            "run_name": experiment.name,
        }
    )
    params = flatten_mapping(_common_run_payload(config, experiment, split))
    params["vocabulary_size"] = len(vocabulary)
    params["effective_embedding_dim"] = neural_artifacts.embedding_info["embedding_dim"]
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    with tempfile.TemporaryDirectory() as temp_dir:
        analysis_dir = Path(temp_dir) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        save_json(split.dataset_metadata, analysis_dir / "dataset_metadata.json")
        save_json(asdict(experiment), analysis_dir / "experiment_config.json")
        save_json(neural_artifacts.embedding_info, analysis_dir / "embedding_info.json")
        if embedding_artifacts:
            save_json(embedding_artifacts, analysis_dir / "embedding_coverage.json")
        _save_text_lines(vocabulary, analysis_dir / "vocabulary.txt")
        save_evaluation_artifacts(split.y_val, val_probabilities, val_predictions, analysis_dir, prefix="validation")
        save_evaluation_artifacts(split.y_test, test_probabilities, test_predictions, analysis_dir, prefix="test")
        history_plot = save_history_plot(history, analysis_dir)
        save_model_summary(neural_artifacts.model, analysis_dir)
        if history_plot is None:
            save_json({"warning": "No se generó historial de entrenamiento."}, analysis_dir / "history_warning.json")
        mlflow.log_artifacts(str(analysis_dir), artifact_path="analysis")

    model_metadata = {
        "threshold": config.neural.threshold,
        "backend": "keras",
        "model_name": experiment.name,
    }
    try:
        log_deployable_model(neural_artifacts.model, backend="keras", artifact_path="model", metadata=model_metadata)
    except Exception as exc:
        mlflow.set_tag("model_artifact_upload_error", str(exc)[:500])
        print(f"WARNING: No se pudo subir el artefacto del modelo ({experiment.name}): {exc}", file=sys.stderr)

    summary_rows.append(
        {
            "run_id": mlflow.active_run().info.run_id,
            "experiment_name": experiment.name,
            "model_family": "neural",
            "embedding_source": neural_artifacts.embedding_info["embedding_source"],
            "embedding_trainable": neural_artifacts.embedding_info["embedding_trainable"],
            "test_f1": metrics["test_f1"],
            "test_roc_auc": metrics["test_roc_auc"],
        }
    )


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    seed = config.splits.random_state
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

    tracking_uri = _require_remote_tracking_uri(args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    split = prepare_dataset_splits(config.dataset, config.splits)
    summary_rows: list[dict[str, object]] = []
    experiments = _selected_experiments(config.experiments, args.only)

    for experiment in experiments:
        with mlflow.start_run(run_name=experiment.name):
            if experiment.type == "baseline":
                run_baseline(config, experiment, split, summary_rows)
            else:
                run_neural(config, experiment, split, summary_rows)

    summary_dir = PROJECT_ROOT / config.tracking.artifact_location
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).sort_values("test_f1", ascending=False).to_csv(summary_path, index=False)
    print(json.dumps({"summary_path": str(summary_path), "experiments_executed": len(summary_rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

