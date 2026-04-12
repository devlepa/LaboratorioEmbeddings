#!/usr/bin/env python
"""Selecciona el mejor modelo desde los artefactos locales de MLflow.

Lee los test_classification_report.json de cada run, ordena por la métrica
indicada, localiza el directorio del modelo en el registro local y devuelve
su ruta absoluta por stdout (para usarlo como MODEL_URI en la API).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = PROJECT_ROOT / "mlruns" / "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selecciona el mejor modelo desde artefactos locales.")
    parser.add_argument(
        "--metric",
        default="test_f1",
        choices=["test_f1", "test_accuracy", "test_roc_auc", "test_precision", "test_recall"],
        help="Métrica para elegir el mejor modelo (default: test_f1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra la tabla completa de resultados por stderr.",
    )
    return parser.parse_args()


def _metric_from_report(report: dict, metric: str) -> float:
    if metric == "test_accuracy":
        return float(report.get("accuracy", 0.0))
    if metric == "test_f1":
        return float(report.get("macro avg", {}).get("f1-score", 0.0))
    if metric == "test_precision":
        return float(report.get("macro avg", {}).get("precision", 0.0))
    if metric == "test_recall":
        return float(report.get("macro avg", {}).get("recall", 0.0))
    if metric == "test_roc_auc":
        # ROC-AUC no está en el classification report; se omite y vale 0.
        return 0.0
    return 0.0


def _build_run_index() -> dict[str, Path]:
    """Devuelve {run_id: model_dir} leyendo los MLmodel de cada versión registrada."""
    index: dict[str, Path] = {}
    models_dir = MLRUNS_DIR / "models"
    if not models_dir.exists():
        return index
    for version_dir in models_dir.iterdir():
        mlmodel_path = version_dir / "artifacts" / "MLmodel"
        if not mlmodel_path.exists():
            continue
        for line in mlmodel_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("run_id:"):
                run_id = line.split(":", 1)[1].strip()
                index[run_id] = version_dir / "artifacts"
                break
    return index


def main() -> None:
    args = parse_args()
    run_to_model = _build_run_index()

    results: list[dict] = []
    for run_dir in MLRUNS_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name == "models":
            continue
        report_path = run_dir / "artifacts" / "analysis" / "test_classification_report.json"
        config_path = run_dir / "artifacts" / "analysis" / "experiment_config.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        model_dir = run_to_model.get(run_dir.name)
        if model_dir is None:
            continue
        value = _metric_from_report(report, args.metric)
        results.append(
            {
                "run_id": run_dir.name,
                "model": config.get("name", "unknown"),
                "metric_value": value,
                "model_dir": model_dir,
                # Para desempate secundario: preferir el directorio con nombre menor
                # (reproducible entre ejecuciones).
                "tiebreak": str(model_dir),
            }
        )

    if not results:
        print(
            f"ERROR: No se encontraron runs con artefactos de análisis en {MLRUNS_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    results.sort(key=lambda r: (-r["metric_value"], r["tiebreak"]))

    if args.verbose:
        print(
            f"{'Run ID':<36} {'Modelo':<35} {args.metric:>10}  Ruta",
            file=sys.stderr,
        )
        print("-" * 110, file=sys.stderr)
        for r in results:
            print(
                f"{r['run_id']} {r['model']:<35} {r['metric_value']:>10.4f}  {r['model_dir']}",
                file=sys.stderr,
            )

    best = results[0]
    if args.verbose:
        print(
            f"\nMejor modelo: {best['model']}  ({args.metric}={best['metric_value']:.4f})",
            file=sys.stderr,
        )
    print(best["model_dir"])


if __name__ == "__main__":
    main()
