from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5, prefix: str = "test") -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, predictions)),
        f"{prefix}_precision": float(precision_score(y_true, predictions, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, predictions, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, predictions, zero_division=0)),
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, probabilities)),
    }
    return metrics, predictions


def save_evaluation_artifacts(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    target_dir: str | Path,
    prefix: str,
) -> list[Path]:
    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    report_path = output_dir / f"{prefix}_classification_report.json"
    report_payload = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts.append(report_path)

    confusion_path = output_dir / f"{prefix}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, predictions)).plot(ax=ax, colorbar=False)
    ax.set_title(f"Matriz de confusión ({prefix})")
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=150)
    plt.close(fig)
    artifacts.append(confusion_path)

    roc_path = output_dir / f"{prefix}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title(f"Curva ROC ({prefix})")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    artifacts.append(roc_path)

    predictions_frame = pd.DataFrame(
        {
            "y_true": y_true,
            "probability": probabilities,
            "prediction": predictions,
        }
    )
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)
    artifacts.append(predictions_path)
    return artifacts


def save_history_plot(history, target_dir: str | Path) -> Path | None:
    if history is None:
        return None

    history_frame = pd.DataFrame(history.history)
    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_history.csv"
    history_frame.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    history_frame[["loss", "val_loss"]].plot(ax=axes[0], title="Loss")
    metric_columns = [column for column in history_frame.columns if column in {"accuracy", "val_accuracy", "auc", "val_auc"}]
    if metric_columns:
        history_frame[metric_columns].plot(ax=axes[1], title="Métricas")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    plot_path = output_dir / "training_history.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_model_summary(model, target_dir: str | Path) -> Path:
    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "model_summary.txt"
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
    summary_path.write_text(buffer.getvalue(), encoding="utf-8")
    return summary_path
