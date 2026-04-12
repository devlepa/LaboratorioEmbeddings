from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SentimentPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, backend: str):
        self.backend = backend

    def load_context(self, context):
        metadata_path = context.artifacts["metadata"]
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if self.backend == "sklearn":
            self.model = joblib.load(context.artifacts["model"])
        elif self.backend == "keras":
            import tensorflow as tf
            from imdb_sentiment.preprocessing import keras_standardization  # noqa: F401

            self.model = tf.keras.models.load_model(context.artifacts["model"])
        else:
            raise ValueError(f"Backend no soportado: {self.backend}")

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = model_input.iloc[:, 0].astype(str).tolist()
        else:
            texts = pd.Series(model_input).astype(str).tolist()

        if self.backend == "sklearn":
            probabilities = self.model.predict_proba(texts)[:, 1]
        else:
            probabilities = np.asarray(self.model.predict(np.asarray(texts, dtype=object), verbose=0)).reshape(-1)

        threshold = float(self.metadata.get("threshold", 0.5))
        predicted_class = (probabilities >= threshold).astype(int)
        labels = np.where(predicted_class == 1, "positive", "negative")
        return pd.DataFrame(
            {
                "text": texts,
                "probability": probabilities,
                "predicted_class": predicted_class,
                "predicted_label": labels,
            }
        )


def log_deployable_model(model, backend: str, artifact_path: str, metadata: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metadata_path = temp_path / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        if backend == "sklearn":
            model_path = temp_path / "baseline_pipeline.joblib"
            joblib.dump(model, model_path)
        elif backend == "keras":
            model_path = temp_path / "sentiment_model.keras"
            model.save(model_path)
        else:
            raise ValueError(f"Backend no soportado: {backend}")

        example_input = pd.DataFrame({"text": ["Una película maravillosa con actuaciones memorables."]})
        if backend == "sklearn":
            example_probabilities = model.predict_proba(example_input["text"].tolist())[:, 1]
        else:
            example_probabilities = np.asarray(
                model.predict(example_input["text"].to_numpy(dtype=object), verbose=0)
            ).reshape(-1)
        example_classes = (example_probabilities >= float(metadata.get("threshold", 0.5))).astype(int)
        example_output = pd.DataFrame(
            {
                "text": example_input["text"],
                "probability": example_probabilities,
                "predicted_class": example_classes,
                "predicted_label": np.where(example_classes == 1, "positive", "negative"),
            }
        )
        signature = mlflow.models.infer_signature(example_input, example_output)
        if backend == "sklearn":
            pip_requirements = [
                "mlflow",
                "cloudpickle",
                "scikit-learn",
                "pandas",
                "numpy",
                "joblib",
            ]
        else:
            pip_requirements = [
                "mlflow",
                "cloudpickle",
                "tensorflow",
                "pandas",
                "numpy",
            ]

        mlflow.pyfunc.log_model(
            name=artifact_path,
            python_model=SentimentPyfuncModel(backend),
            artifacts={
                "model": str(model_path),
                "metadata": str(metadata_path),
            },
            code_paths=[str(PROJECT_ROOT / "src")],
            input_example=example_input,
            signature=signature,
            pip_requirements=pip_requirements,
        )
