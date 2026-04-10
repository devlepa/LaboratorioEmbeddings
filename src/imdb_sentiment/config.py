from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    handle: str
    file_path: str
    text_column_candidates: list[str]
    label_column_candidates: list[str]


@dataclass(slots=True)
class SplitConfig:
    train_size: float
    validation_size: float
    test_size: float
    random_state: int


@dataclass(slots=True)
class BaselineConfig:
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    C: float
    max_iter: int


@dataclass(slots=True)
class NeuralConfig:
    max_tokens: int
    sequence_length: int
    embedding_dim: int
    hidden_units: list[int]
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    threshold: float


@dataclass(slots=True)
class TrackingConfig:
    experiment_name: str
    artifact_location: str
    metric_for_best_model: str
    registered_model_name: str


@dataclass(slots=True)
class EmbeddingConfig:
    kind: str
    trainable: bool
    provider: str | None = None
    identifier: str | None = None
    source_name: str | None = None
    lookup_style: str | None = None


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    type: str
    description: str
    embedding: EmbeddingConfig | None = None


@dataclass(slots=True)
class ProjectConfig:
    dataset: DatasetConfig
    splits: SplitConfig
    baseline: BaselineConfig
    neural: NeuralConfig
    tracking: TrackingConfig
    experiments: list[ExperimentConfig] = field(default_factory=list)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_config(path: str | Path) -> ProjectConfig:
    payload = _read_yaml(path)

    experiments: list[ExperimentConfig] = []
    for item in payload["experiments"]:
        embedding = item.get("embedding")
        experiments.append(
            ExperimentConfig(
                name=item["name"],
                type=item["type"],
                description=item["description"],
                embedding=EmbeddingConfig(**embedding) if embedding else None,
            )
        )

    return ProjectConfig(
        dataset=DatasetConfig(**payload["dataset"]),
        splits=SplitConfig(**payload["splits"]),
        baseline=BaselineConfig(
            max_features=payload["baseline"]["max_features"],
            ngram_range=tuple(payload["baseline"]["ngram_range"]),
            min_df=payload["baseline"]["min_df"],
            C=payload["baseline"]["C"],
            max_iter=payload["baseline"]["max_iter"],
        ),
        neural=NeuralConfig(**payload["neural"]),
        tracking=TrackingConfig(**payload["tracking"]),
        experiments=experiments,
    )
