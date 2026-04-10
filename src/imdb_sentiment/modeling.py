from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras

from imdb_sentiment.config import BaselineConfig, EmbeddingConfig, NeuralConfig
from imdb_sentiment.embeddings import EmbeddingMatrix
from imdb_sentiment.preprocessing import build_text_vectorizer, normalize_text


@dataclass(slots=True)
class NeuralArtifacts:
    model: keras.Model
    vectorizer: keras.layers.TextVectorization
    embedding_info: dict[str, object]


def _sanitize_vocabulary(vocabulary: list[object]) -> list[str]:
    sanitized: list[str] = []
    seen: set[str] = set()
    for token in vocabulary:
        normalized = str(token)
        if normalized in {"", "[UNK]"}:
            continue
        if normalized in seen:
            continue
        sanitized.append(normalized)
        seen.add(normalized)
    return sanitized


def build_baseline_pipeline(config: BaselineConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    max_features=config.max_features,
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def adapt_vectorizer(train_texts: np.ndarray, config: NeuralConfig) -> keras.layers.TextVectorization:
    vectorizer = build_text_vectorizer(config.max_tokens, config.sequence_length)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(1024))
    clean_vocabulary = _sanitize_vocabulary(vectorizer.get_vocabulary())

    clean_vectorizer = build_text_vectorizer(config.max_tokens, config.sequence_length)
    clean_vectorizer.set_vocabulary(clean_vocabulary)
    return clean_vectorizer


def build_neural_model(
    config: NeuralConfig,
    vectorizer: keras.layers.TextVectorization,
    embedding_config: EmbeddingConfig | None = None,
    pretrained_matrix: EmbeddingMatrix | None = None,
) -> NeuralArtifacts:

    vocabulary = vectorizer.get_vocabulary()
    text_input = keras.Input(shape=(), dtype=tf.string, name="text")
    token_ids = vectorizer(text_input)

    if pretrained_matrix is None:
        embedding_layer = keras.layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=config.embedding_dim,
            mask_zero=True,
            name="embedding",
        )
        embedding_info = {
            "embedding_source": "trainable",
            "embedding_dim": config.embedding_dim,
            "embedding_trainable": True,
        }
    else:
        embedding_layer = keras.layers.Embedding(
            input_dim=pretrained_matrix.matrix.shape[0],
            output_dim=pretrained_matrix.matrix.shape[1],
            embeddings_initializer=keras.initializers.Constant(pretrained_matrix.matrix),
            trainable=bool(embedding_config and embedding_config.trainable),
            mask_zero=True,
            name="embedding",
        )
        embedding_info = {
            "embedding_source": pretrained_matrix.source_name,
            "embedding_dim": pretrained_matrix.dimension,
            "embedding_trainable": bool(embedding_config and embedding_config.trainable),
            "embedding_coverage": pretrained_matrix.coverage,
            "embedding_covered_tokens": pretrained_matrix.covered_tokens,
            "embedding_total_tokens": pretrained_matrix.total_tokens,
            "embedding_missing_tokens_sample": pretrained_matrix.missing_tokens,
        }

    hidden = embedding_layer(token_ids)
    hidden = keras.layers.GlobalAveragePooling1D(name="pooling")(hidden)
    for index, units in enumerate(config.hidden_units, start=1):
        hidden = keras.layers.Dense(units, activation="relu", name=f"dense_{index}")(hidden)
        hidden = keras.layers.Dropout(config.dropout_rate, name=f"dropout_{index}")(hidden)
    output = keras.layers.Dense(1, activation="sigmoid", name="sentiment")(hidden)

    model = keras.Model(inputs=text_input, outputs=output, name="dense_sentiment_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return NeuralArtifacts(model=model, vectorizer=vectorizer, embedding_info=embedding_info)


def fit_neural_model(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, config: NeuralConfig):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        )
    ]
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )
