#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera el reporte comparativo desde MLflow.")
    parser.add_argument("--experiment-name", required=True, help="Nombre del experimento en MLflow.")
    parser.add_argument("--tracking-uri", default=None, help="URI de MLflow. Si se omite, usa MLFLOW_TRACKING_URI.")
    parser.add_argument("--output-dir", default="artifacts/reports", help="Directorio local donde guardar el reporte.")
    return parser.parse_args()


def _runs_to_frame(client: mlflow.MlflowClient, experiment_name: str) -> pd.DataFrame:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No existe el experimento '{experiment_name}' en MLflow.")

    rows: list[dict[str, object]] = []
    for run in client.search_runs([experiment.experiment_id], order_by=["metrics.test_f1 DESC"]):
        row: dict[str, object] = {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
            "model_family": run.data.tags.get("model_family", ""),
            "embedding_provider": run.data.tags.get("embedding_provider", ""),
            "embedding_trainable": run.data.tags.get("embedding_trainable", ""),
        }
        row.update({key.replace("metrics.", ""): value for key, value in run.data.metrics.items()})
        row.update({key.replace("params.", ""): value for key, value in run.data.params.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _build_observations(runs_frame: pd.DataFrame) -> list[str]:
    observations: list[str] = []
    baseline_row = runs_frame[runs_frame["run_name"] == "baseline_tfidf_logreg"]
    trainable_row = runs_frame[runs_frame["run_name"] == "dense_trainable_embeddings"]

    if not baseline_row.empty and not trainable_row.empty:
        baseline_f1 = float(baseline_row.iloc[0]["test_f1"])
        trainable_f1 = float(trainable_row.iloc[0]["test_f1"])
        if baseline_f1 >= trainable_f1:
            observations.append(
                "El baseline TF-IDF + regresión logística iguala o supera a la red con embeddings entrenables, lo que suele indicar que las pistas léxicas y los n-gramas capturan muy bien la polaridad explícita del corpus."
            )
        else:
            observations.append(
                "La red con embeddings entrenables supera al baseline clásico, lo que sugiere que el modelo logra generalizar mejor a variaciones léxicas y contextos donde los n-gramas por sí solos se quedan cortos."
            )

    pretrained = runs_frame[runs_frame["embedding_provider"].isin(["spacy", "gensim"])].copy()
    if not pretrained.empty:
        frozen = pretrained[pretrained["embedding_trainable"] == "False"]
        tuned = pretrained[pretrained["embedding_trainable"] == "True"]
        if not frozen.empty and not tuned.empty:
            frozen_mean = float(frozen["test_f1"].mean())
            tuned_mean = float(tuned["test_f1"].mean())
            if tuned_mean > frozen_mean:
                observations.append(
                    "En promedio, ajustar los embeddings durante el entrenamiento mejora el F1 frente a dejarlos congelados; esto apunta a que la adaptación al dominio de reseñas de cine aporta información útil."
                )
            else:
                observations.append(
                    "En promedio, congelar los embeddings mantiene un desempeño comparable o mejor que afinarlos, lo cual sugiere que los vectores preentrenados ya traen suficiente estructura semántica y que el ajuste puede inducir sobreajuste."
                )

        if "embedding_coverage" in pretrained.columns:
            best_coverage = pretrained.sort_values("embedding_coverage", ascending=False).iloc[0]
            observations.append(
                f"El embedding con mayor cobertura de vocabulario fue {best_coverage['run_name']} ({best_coverage['embedding_coverage']:.2%}); esta variable ayuda a explicar por qué algunos modelos transfieren mejor el conocimiento léxico al dominio."
            )

    best_row = runs_frame.iloc[0]
    observations.append(
        f"El mejor modelo por `test_f1` fue {best_row['run_name']} con F1={best_row['test_f1']:.4f} y ROC-AUC={best_row['test_roc_auc']:.4f}. La selección final debe considerar también estabilidad, costo de inferencia y facilidad de despliegue."
    )
    return observations


def main() -> None:
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = mlflow.MlflowClient()
    runs_frame = _runs_to_frame(client, args.experiment_name)
    if runs_frame.empty:
        raise ValueError("No se encontraron corridas terminadas para generar el reporte.")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparative_report.md"
    plot_path = output_dir / "test_f1_comparison.png"

    sort_columns = [column for column in ["test_f1", "test_roc_auc"] if column in runs_frame.columns]
    runs_frame = runs_frame.sort_values(sort_columns, ascending=False)
    runs_frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=runs_frame, x="run_name", y="test_f1", hue="model_family")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    observations = _build_observations(runs_frame)
    table = runs_frame[["run_name", "model_family", "test_f1", "test_roc_auc", "val_f1", "val_roc_auc"]].to_markdown(index=False)
    md_content = "\n".join(
        [
            "# Reporte comparativo",
            "",
            "## Tabla resumen",
            "",
            table,
            "",
            "## Interpretación",
            "",
        ]
        + [f"- {observation}" for observation in observations]
        + [
            "",
            "## Artefactos generados",
            "",
            f"- CSV comparativo: `{csv_path}`",
            f"- Gráfico de F1: `{plot_path}`",
        ]
    )
    md_path.write_text(md_content, encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()

