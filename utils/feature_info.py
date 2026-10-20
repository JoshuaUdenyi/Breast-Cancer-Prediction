"""
feature_info.py

Static metadata about the 30 Wisconsin Diagnostic Breast Cancer (WDBC)
features: human-readable descriptions, grouping (mean / se / worst), and
medically plausible input ranges used for input validation in the
Streamlit app.

This module has no dependency on scikit-learn at runtime, so it can be
imported cheaply by both the training script and the web app.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


class FeatureMeta(TypedDict):
    """Metadata describing a single input feature."""

    label: str
    description: str
    group: str  # "mean", "se" (standard error), or "worst"
    min_value: float
    max_value: float
    default: float
    step: float


# Base descriptions for the 10 underlying cell-nucleus measurements.
# Each of these appears three times in the dataset: as a mean, a standard
# error, and a "worst" (largest) value across the cell nuclei measured
# in a digitized image of a breast mass.
_BASE_DESCRIPTIONS: Dict[str, str] = {
    "radius": "Mean distance from center to points on the perimeter",
    "texture": "Standard deviation of gray-scale values (surface roughness)",
    "perimeter": "Perimeter length of the cell nucleus",
    "area": "Area of the cell nucleus",
    "smoothness": "Local variation in radius lengths (edge smoothness)",
    "compactness": "Perimeter^2 / area - 1.0 (shape compactness)",
    "concavity": "Severity of concave portions of the contour",
    "concave points": "Number of concave portions of the contour",
    "symmetry": "Symmetry of the cell nucleus shape",
    "fractal dimension": "Coastline approximation - 1 (contour complexity)",
}

# Approximate, medically plausible min/max/default values per base
# measurement, derived from the published value ranges of the WDBC
# dataset. These are used to build sane slider bounds in the UI, not as
# clinical reference ranges.
_RANGES: Dict[str, tuple] = {
    "radius": (6.0, 30.0, 14.0),
    "texture": (9.0, 40.0, 19.0),
    "perimeter": (40.0, 190.0, 92.0),
    "area": (140.0, 2510.0, 655.0),
    "smoothness": (0.05, 0.17, 0.096),
    "compactness": (0.02, 1.06, 0.104),
    "concavity": (0.0, 1.25, 0.089),
    "concave points": (0.0, 0.21, 0.048),
    "symmetry": (0.1, 0.31, 0.181),
    "fractal dimension": (0.05, 0.1, 0.063),
}

_GROUP_LABELS = {
    "mean": "Mean",
    "se": "Standard error",
    "worst": "Worst (largest)",
}

# sklearn's load_breast_cancer() feature_names order, exactly as returned
# by the dataset, e.g. "mean radius", "radius error", "worst radius".
_SKLEARN_SUFFIX = {"mean": "mean", "se": "error", "worst": "worst"}


def _build_feature_metadata() -> Dict[str, FeatureMeta]:
    """Construct the full 30-feature metadata dict keyed by sklearn's
    exact feature name strings, e.g. 'mean radius', 'radius error',
    'worst radius'.
    """

    metadata: Dict[str, FeatureMeta] = {}
    for base, description in _BASE_DESCRIPTIONS.items():
        low, high, default = _RANGES[base]
        step = round((high - low) / 200, 4) or 0.001

        for group, suffix in _SKLEARN_SUFFIX.items():
            if group == "mean":
                key = f"mean {base}"
                scale = 1.0
            elif group == "se":
                key = f"{base} error"
                # Standard errors are typically much smaller than the mean.
                scale = 0.15
            else:  # worst
                key = f"worst {base}"
                scale = 1.3

            metadata[key] = {
                "label": f"{base.title()} ({_GROUP_LABELS[group]})",
                "description": description,
                "group": group,
                "min_value": round(low * scale * 0.2, 4),
                "max_value": round(high * scale * 1.1, 4),
                "default": round(default * scale, 4),
                "step": step,
            }
    return metadata


FEATURE_METADATA: Dict[str, FeatureMeta] = _build_feature_metadata()


def features_by_group(group: str) -> List[str]:
    """Return feature names belonging to a given group in a stable order."""

    return [name for name, meta in FEATURE_METADATA.items() if meta["group"] == group]
