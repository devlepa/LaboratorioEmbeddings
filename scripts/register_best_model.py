#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Registra el mejor modelo desde un experimento de MLflow.")
    parser.add_argument("--experiment-name", required=True, help="Nombre del experimento en MLflow.")
    parser.add_argument("--registered-model-name", required=True, help="Nombre del modelo en el registro de MLflow.")
    parser.add_argument("--metric", default="test_f1", help="Métrica con la que se elegirá el mejor modelo.")
    parser.add_argument("--tracking-uri", default=None, help="URI de MLflow. Si se omite, usa MLFLOW_TRACKING_URI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise ValueError(f"No existe el experimento '{args.experiment_name}'.")

    runs = client.search_runs([experiment.experiment_id], order_by=[f"metrics.{args.metric} DESC"])
    if not runs:
        raise ValueError("No hay corridas disponibles para registrar.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"

    try:
        client.create_registered_model(args.registered_model_name)
    except Exception:
        pass

    version = mlflow.register_model(model_uri=model_uri, name=args.registered_model_name)
    try:
        client.set_registered_model_alias(args.registered_model_name, "champion", version.version)
    except Exception:
        pass

    print(
        {
            "best_run_id": best_run.info.run_id,
            "metric": args.metric,
            "metric_value": best_run.data.metrics.get(args.metric),
            "registered_model_name": args.registered_model_name,
            "registered_version": version.version,
            "model_uri": model_uri,
        }
    )


if __name__ == "__main__":
    main()

