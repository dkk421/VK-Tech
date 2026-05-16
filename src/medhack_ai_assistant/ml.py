from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.medhack_ai_assistant.config import ModelConfig


@dataclass(frozen=True)
class ValidationResult:
    threshold: float
    f1: float


def build_model(config: ModelConfig) -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=(config.ngram_min, config.ngram_max),
                min_df=config.min_df,
                lowercase=True,
            ),
        ),
        (
            "clf",
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=config.random_state,
                max_iter=1000,
                tol=1e-3,
            ),
        ),
    ])


def find_best_threshold(
    y_true: pd.Series,
    positive_proba: np.ndarray,
    config: ModelConfig,
) -> ValidationResult:
    thresholds = np.linspace(
        config.threshold_min,
        config.threshold_max,
        config.threshold_count,
    )
    scores = [
        f1_score(y_true, positive_proba >= threshold)
        for threshold in thresholds
    ]

    best_idx = int(np.argmax(scores))
    return ValidationResult(
        threshold=float(thresholds[best_idx]),
        f1=float(scores[best_idx]),
    )


def train_and_validate(
    x: pd.Series,
    y: pd.Series,
    config: ModelConfig,
) -> ValidationResult:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = build_model(config)
    model.fit(x_train, y_train)
    valid_proba = model.predict_proba(x_valid)[:, 1]

    return find_best_threshold(y_valid, valid_proba, config)


def train_final_model(
    x: pd.Series,
    y: pd.Series,
    config: ModelConfig,
) -> Pipeline:
    model = build_model(config)
    model.fit(x, y)
    return model


def predict_with_threshold(
    model: Pipeline,
    x: pd.Series,
    threshold: float,
) -> pd.Series:
    positive_proba = model.predict_proba(x)[:, 1]
    return pd.Series(positive_proba >= threshold, index=x.index)
