"""Baseline TF-IDF + Logistic Regression classifier."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_pipeline(max_features: int = 20000) -> Pipeline:
    """Create a TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train_baseline(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit the baseline model and return the trained pipeline."""
    model = build_baseline_pipeline()
    model.fit(X, y)
    return model


def evaluate_baseline(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Evaluate the baseline model using simple accuracy placeholder."""
    accuracy = model.score(X, y)
    return {"accuracy": float(accuracy)}


if __name__ == "__main__":
    raise SystemExit("Baseline utilities are intended to be imported, not executed directly.")
