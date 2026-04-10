from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from imdb_sentiment.config import DatasetConfig, SplitConfig


@dataclass(slots=True)
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    text_column: str
    label_column: str
    dataset_metadata: dict[str, Any]


def _pick_column(columns: list[str], candidates: list[str], field_name: str) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"No se encontró una columna válida para {field_name}. Columnas disponibles: {columns}")


def _normalize_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "positive": 1,
        "positivo": 1,
        "pos": 1,
        "1": 1,
        "negative": 0,
        "negativo": 0,
        "neg": 0,
        "0": 0,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        unknown = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Etiquetas no reconocidas: {unknown}")
    return normalized.astype(int)


def _discover_kaggle_file(handle: str, requested_file_path: str) -> str:
    if requested_file_path.strip():
        return requested_file_path

    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(handle))
    csv_candidates = sorted(dataset_dir.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No se encontró ningún CSV dentro de {dataset_dir}.")
    return csv_candidates[0].relative_to(dataset_dir).as_posix()


def load_imdb_spanish_dataframe(config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    file_path = _discover_kaggle_file(config.handle, config.file_path)
    load_fn = getattr(kagglehub, "load_dataset", None) or getattr(kagglehub, "dataset_load", None)
    if load_fn is None:
        raise AttributeError("La versión instalada de kagglehub no expone load_dataset ni dataset_load.")

    dataframe = load_fn(
        KaggleDatasetAdapter.PANDAS,
        config.handle,
        file_path,
    )
    metadata = {
        "dataset_handle": config.handle,
        "dataset_file_path": file_path,
        "rows": int(len(dataframe)),
    }
    return dataframe, metadata


def prepare_dataset_splits(dataset_config: DatasetConfig, split_config: SplitConfig) -> DataSplit:
    dataframe, metadata = load_imdb_spanish_dataframe(dataset_config)
    text_column = _pick_column(dataframe.columns.tolist(), dataset_config.text_column_candidates, "texto")
    label_column = _pick_column(dataframe.columns.tolist(), dataset_config.label_column_candidates, "etiqueta")

    prepared = dataframe[[text_column, label_column]].dropna().copy()
    prepared[text_column] = prepared[text_column].astype(str).str.strip()
    prepared = prepared[prepared[text_column] != ""]
    prepared[label_column] = _normalize_labels(prepared[label_column])

    x = prepared[text_column].to_numpy(dtype=object)
    y = prepared[label_column].to_numpy(dtype=np.int32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=y,
    )

    validation_share = split_config.validation_size / (split_config.train_size + split_config.validation_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_share,
        random_state=split_config.random_state,
        stratify=y_train_val,
    )

    class_distribution = {
        "train": {
            "negative": int((y_train == 0).sum()),
            "positive": int((y_train == 1).sum()),
        },
        "validation": {
            "negative": int((y_val == 0).sum()),
            "positive": int((y_val == 1).sum()),
        },
        "test": {
            "negative": int((y_test == 0).sum()),
            "positive": int((y_test == 1).sum()),
        },
    }
    metadata.update(
        {
            "text_column": text_column,
            "label_column": label_column,
            "rows_after_cleaning": int(len(prepared)),
            "class_distribution": class_distribution,
        }
    )

    return DataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        text_column=text_column,
        label_column=label_column,
        dataset_metadata=metadata,
    )

