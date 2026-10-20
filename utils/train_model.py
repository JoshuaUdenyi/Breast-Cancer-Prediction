"""
train_model.py

Trains a Random Forest classifier on the Wisconsin Diagnostic Breast
Cancer (WDBC) dataset end to end: EDA artifacts, preprocessing,
hyperparameter tuning via GridSearchCV, cross-validation, evaluation,
model comparison, and artifact saving (model.pkl, feature_names.pkl,
metrics.pkl) for use by the Streamlit application.

Run with:
    python utils/train_model.py

This script is also the source of truth for notebooks/train_model.ipynb,
which wraps the same logic in Colab-friendly notebook cells with
extended EDA/markdown commentary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series, Any]:
    """Load the built-in WDBC dataset and return it as a DataFrame."""

    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")  # 0 = malignant, 1 = benign

    print("Dataset description (excerpt):")
    print(dataset.DESCR[:400], "...\n")
    print("Feature names:", list(dataset.feature_names))
    print("Target names:", list(dataset.target_names))
    print("Shape:", X.shape)
    print("\nFirst five rows:\n", X.head())

    return X, y, dataset


def run_eda(X: pd.DataFrame, y: pd.Series, target_names) -> None:
    """Generate and save exploratory data analysis figures."""

    print("\n--- Exploratory Data Analysis ---")
    print("Missing values per column:\n", X.isnull().sum().sum(), "total missing values")
    print("\nClass distribution:\n", y.value_counts().rename(index=dict(enumerate(target_names))))
    print("\nSummary statistics:\n", X.describe().T[["mean", "std", "min", "max"]])

    # Target balance bar chart
    plt.figure(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    labels = [target_names[i] for i in counts.index]
    sns.barplot(x=labels, y=counts.values, hue=labels, palette="viridis", legend=False)
    plt.title("Class distribution (0 = malignant, 1 = benign)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6})
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Histograms for a handful of key features
    key_features = ["mean radius", "mean texture", "mean smoothness", "mean concavity"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feat in zip(axes.ravel(), key_features):
        sns.histplot(data=X, x=feat, hue=y.map(dict(enumerate(target_names))), kde=True, ax=ax)
        ax.set_title(feat)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_histograms.png", dpi=150)
    plt.close()

    # Pair plot for a few important features (this can be slow, kept small)
    pair_df = X[key_features].copy()
    pair_df["diagnosis"] = y.map(dict(enumerate(target_names)))
    pair_plot = sns.pairplot(pair_df, hue="diagnosis", corner=True, palette="viridis")
    pair_plot.savefig(IMAGES_DIR / "pair_plot.png", dpi=150)
    plt.close("all")

    print(f"\nEDA figures saved to: {IMAGES_DIR}")


def train_and_tune(X_train, y_train) -> GridSearchCV:
    """Run GridSearchCV with cross-validation over a Random Forest."""

    print("\n--- Hyperparameter tuning (GridSearchCV) ---")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    baseline = RandomForestClassifier(random_state=RANDOM_STATE)
    baseline_scores = cross_val_score(baseline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Baseline RF 5-fold CV accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\nGrid search completed in {elapsed:.1f}s")
    print("Best parameters:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    return grid


def evaluate_model(model, X_test, y_test, target_names) -> Dict[str, Any]:
    """Compute standard classification metrics and save evaluation plots."""

    print("\n--- Evaluation ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")

    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification report:\n", report)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=target_names, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC curve")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances = importances.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", legend=False)
    plt.title("Top 15 feature importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=150)
    plt.close()

    metrics["classification_report"] = report
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["feature_importance"] = importances.to_dict()
    return metrics


def compare_models(X_train, y_train, X_test, y_test) -> Dict[str, Dict[str, float]]:
    """Train baseline versions of alternative classifiers for comparison."""

    print("\n--- Model comparison ---")
    candidates = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
    }

    comparison: Dict[str, Dict[str, float]] = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]
        comparison[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        print(f"{name}: accuracy={comparison[name]['accuracy']:.4f}, f1={comparison[name]['f1_score']:.4f}")

    # Comparison chart
    comp_df = pd.DataFrame(comparison).T
    comp_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(
        kind="bar", figsize=(10, 6), colormap="viridis"
    )
    plt.title("Model comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.ylim(0.8, 1.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "model_comparison.png", dpi=150)
    plt.close()

    return comparison


def main() -> None:
    X, y, dataset = load_data()
    target_names = list(dataset.target_names)

    run_eda(X, y, target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    grid = train_and_tune(X_train, y_train)
    best_model = grid.best_estimator_

    metrics = evaluate_model(best_model, X_test, y_test, target_names)
    metrics["best_params"] = grid.best_params_
    metrics["cv_best_score"] = grid.best_score_

    comparison = compare_models(X_train, y_train, X_test, y_test)
    metrics["model_comparison"] = comparison

    # Save artifacts
    joblib.dump(best_model, ROOT / "model.pkl")
    joblib.dump(list(X.columns), ROOT / "feature_names.pkl")
    joblib.dump(metrics, ROOT / "metrics.pkl")

    with open(ROOT / "metrics.json", "w") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k not in ("feature_importance",)},
            f,
            indent=2,
            default=str,
        )

    print(f"\nSaved model.pkl, feature_names.pkl, metrics.pkl, metrics.json to {ROOT}")
    print("Training complete.")


if __name__ == "__main__":
    main()
