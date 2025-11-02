"""Explainability utilities for the emotion classification model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExplanationResult:
    """Container for explanation artifacts."""

    raw_output: Any
    shap_values: Any | None = None
    lime_explanation: Any | None = None


def explain_with_shap(model: Any, inputs: np.ndarray) -> ExplanationResult:
    """Placeholder SHAP explanation stub."""
    # TODO: Integrate SHAP once model artifacts are available.
    return ExplanationResult(raw_output=model.predict(inputs), shap_values=None)


def explain_with_lime(model: Any, inputs: np.ndarray) -> ExplanationResult:
    """Placeholder LIME explanation stub."""
    # TODO: Integrate LIME text explainer.
    return ExplanationResult(raw_output=model.predict(inputs), lime_explanation=None)
