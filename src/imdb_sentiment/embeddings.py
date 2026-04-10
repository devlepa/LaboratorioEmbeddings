from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from imdb_sentiment.config import EmbeddingConfig


@dataclass(slots=True)
class EmbeddingMatrix:
    matrix: np.ndarray
    dimension: int
    source_name: str
    coverage: float
    covered_tokens: int
    total_tokens: int
    missing_tokens: list[str]


@lru_cache(maxsize=8)
def _load_spacy_model(identifier: str):
    import spacy

    return spacy.load(identifier)


@lru_cache(maxsize=4)
def _load_gensim_model(identifier: str):
    import gensim.downloader as api

    return api.load(identifier)


def _spaCy_vector(token: str, identifier: str) -> np.ndarray | None:
    nlp = _load_spacy_model(identifier)
    for candidate in (token, token.lower()):
        lexeme = nlp.vocab[candidate]
        if lexeme.has_vector and lexeme.vector_norm > 0:
            return np.asarray(lexeme.vector, dtype=np.float32)
    return None


def _conceptnet_candidates(token: str) -> list[str]:
    normalized = token.strip().lower().replace(" ", "_")
    return [f"/c/es/{normalized}", normalized, token.lower(), token]


def _gensim_vector(token: str, identifier: str, lookup_style: str | None) -> np.ndarray | None:
    vectors = _load_gensim_model(identifier)
    if lookup_style == "conceptnet_es":
        candidates = _conceptnet_candidates(token)
    else:
        candidates = [token, token.lower()]

    for candidate in candidates:
        if candidate in vectors.key_to_index:
            return np.asarray(vectors[candidate], dtype=np.float32)
    return None


def _vector_dimension(config: EmbeddingConfig) -> int:
    if config.provider == "spacy":
        return int(_load_spacy_model(config.identifier).vocab.vectors_length)
    if config.provider == "gensim":
        return int(_load_gensim_model(config.identifier).vector_size)
    raise ValueError(f"Proveedor de embeddings no soportado: {config.provider}")


def _lookup_vector(token: str, config: EmbeddingConfig) -> np.ndarray | None:
    if config.provider == "spacy":
        return _spaCy_vector(token, config.identifier)
    if config.provider == "gensim":
        return _gensim_vector(token, config.identifier, config.lookup_style)
    raise ValueError(f"Proveedor de embeddings no soportado: {config.provider}")


def build_embedding_matrix(vocabulary: list[str], config: EmbeddingConfig) -> EmbeddingMatrix:
    dimension = _vector_dimension(config)
    matrix = np.zeros((len(vocabulary), dimension), dtype=np.float32)
    covered_tokens = 0
    missing_tokens: list[str] = []

    for index, token in enumerate(vocabulary):
        if token in {"", "[UNK]"}:
            continue
        vector = _lookup_vector(token, config)
        if vector is None:
            if len(missing_tokens) < 30:
                missing_tokens.append(token)
            continue
        matrix[index] = vector
        covered_tokens += 1

    return EmbeddingMatrix(
        matrix=matrix,
        dimension=dimension,
        source_name=config.source_name or config.identifier or "pretrained",
        coverage=covered_tokens / max(len(vocabulary) - 2, 1),
        covered_tokens=covered_tokens,
        total_tokens=max(len(vocabulary) - 2, 0),
        missing_tokens=missing_tokens,
    )

