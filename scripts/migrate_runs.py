#!/usr/bin/env python
"""Migra todos los runs de un experimento desde un MLflow local a uno remoto."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migra runs entre servidores MLflow.")
    parser.add_argument("--src", required=True, help="URI origen (ej: sqlite:///mlflow.db)")
    parser.add_argument("--dst", required=True, help="URI destino (ej: http://host:5000)")
    parser.add_argument("--experiment", required=True, help="Nombre del experimento a migrar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src = MlflowClient(tracking_uri=args.src)
    dst = MlflowClient(tracking_uri=args.dst)

    src_exp = src.get_experiment_by_name(args.experiment)
    if src_exp is None:
        raise ValueError(f"Experimento '{args.experiment}' no encontrado en origen.")

    try:
        dst_exp_id = dst.create_experiment(args.experiment)
    except Exception:
        dst_exp_id = dst.get_experiment_by_name(args.experiment).experiment_id

    runs = src.search_runs([src_exp.experiment_id])
    print(f"Migrando {len(runs)} runs a {args.dst} ...")

    mlflow.set_tracking_uri(args.dst)

    for i, run in enumerate(runs, 1):
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        print(f"  [{i}/{len(runs)}] {run_name}")

        with mlflow.start_run(experiment_id=dst_exp_id, run_name=run_name):
            if run.data.params:
                mlflow.log_params(run.data.params)

            if run.data.metrics:
                mlflow.log_metrics(run.data.metrics)

            clean_tags = {
                k: v for k, v in run.data.tags.items()
                if not k.startswith("mlflow.")
            }
            if clean_tags:
                mlflow.set_tags(clean_tags)

            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    src.download_artifacts(run.info.run_id, "", tmpdir)
                    artifacts = list(Path(tmpdir).iterdir())
                    if artifacts:
                        mlflow.log_artifacts(tmpdir)
                except Exception as exc:
                    print(f"    WARNING: no se pudieron migrar artefactos: {exc}")

    print("Migración completada.")


if __name__ == "__main__":
    main()
