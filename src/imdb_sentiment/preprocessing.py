from __future__ import annotations

import html
import re
import string

from tensorflow import keras
import tensorflow as tf

EXTRA_PUNCTUATION = "¿¡“”«»…´`"
PUNCTUATION_PATTERN = f"[{re.escape(string.punctuation + EXTRA_PUNCTUATION)}]"
HTML_BREAK_PATTERN = r"<br\s*/?>"
HTML_TAG_PATTERN = r"<[^>]+>"
WHITESPACE_PATTERN = r"\s+"


def normalize_text(text: str) -> str:
    cleaned = html.unescape(str(text or "")).lower()
    cleaned = re.sub(HTML_BREAK_PATTERN, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(HTML_TAG_PATTERN, " ", cleaned)
    cleaned = re.sub(PUNCTUATION_PATTERN, " ", cleaned)
    cleaned = re.sub(WHITESPACE_PATTERN, " ", cleaned)
    return cleaned.strip()


@keras.utils.register_keras_serializable(package="imdb_sentiment")
def keras_standardization(inputs: tf.Tensor) -> tf.Tensor:
    cleaned = tf.strings.lower(inputs)
    cleaned = tf.strings.regex_replace(cleaned, HTML_BREAK_PATTERN, " ")
    cleaned = tf.strings.regex_replace(cleaned, HTML_TAG_PATTERN, " ")
    cleaned = tf.strings.regex_replace(cleaned, r"&[a-zA-Z#0-9]+;", " ")
    cleaned = tf.strings.regex_replace(cleaned, PUNCTUATION_PATTERN, " ")
    cleaned = tf.strings.regex_replace(cleaned, WHITESPACE_PATTERN, " ")
    return tf.strings.strip(cleaned)


def build_text_vectorizer(max_tokens: int, sequence_length: int) -> keras.layers.TextVectorization:
    return keras.layers.TextVectorization(
        standardize=keras_standardization,
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        name="text_vectorization",
    )
