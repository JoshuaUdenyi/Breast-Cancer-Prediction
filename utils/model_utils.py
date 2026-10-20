"""
model_utils.py

Loads the trained Random Forest model and its supporting artifacts, and
provides typed helper functions for making single and batch predictions.
Used exclusively by the Streamlit application (app.py and pages/*.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

TARGET_NAMES = ["malignant", "benign"]


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """Load the trained RandomForestClassifier from disk (cached)."""

    model_path = ROOT / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "model.pkl not found. Run 'python utils/train_model.py' first "
            "(or run notebooks/train_model.ipynb) to train and save the model."
        )
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_feature_names() -> List[str]:
    """Load the ordered list of feature names the model expects."""

    return joblib.load(ROOT / "feature_names.pkl")


@st.cache_resource(show_spinner=False)
def load_metrics() -> Dict[str, Any]:
    """Load the saved evaluation metrics dict produced during training."""

    metrics_path = ROOT / "metrics.pkl"
    if not metrics_path.exists():
        return {}
    return joblib.load(metrics_path)


def predict_single(input_values: Dict[str, float]) -> Tuple[str, float, np.ndarray]:
    """Run a prediction for a single set of feature values.

    Returns:
        diagnosis: "benign" or "malignant"
        confidence: probability of the predicted class (0-1)
        probabilities: array of [P(malignant), P(benign)]
    """

    model = load_model()
    feature_names = load_feature_names()

    row = pd.DataFrame([[input_values[name] for name in feature_names]], columns=feature_names)
    probabilities = model.predict_proba(row)[0]
    predicted_class = int(np.argmax(probabilities))
    diagnosis = TARGET_NAMES[predicted_class]
    confidence = float(probabilities[predicted_class])

    return diagnosis, confidence, probabilities


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions for a batch of rows (e.g. an uploaded CSV).

    The input DataFrame must contain all 30 required feature columns.
    Extra columns are ignored; missing columns raise a ValueError.
    """

    model = load_model()
    feature_names = load_feature_names()

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")

    X = df[feature_names]
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    results = df.copy()
    results["prediction"] = [TARGET_NAMES[p] for p in predictions]
    results["confidence"] = probabilities.max(axis=1)
    results["probability_malignant"] = probabilities[:, 0]
    results["probability_benign"] = probabilities[:, 1]
    return results


def risk_level(confidence: float, diagnosis: str) -> str:
    """Translate a prediction's confidence into a simple risk label."""

    if diagnosis == "malignant":
        return "High risk" if confidence >= 0.75 else "Elevated risk"
    return "Low risk" if confidence >= 0.75 else "Uncertain - recommend follow-up"
