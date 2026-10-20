"""
pages/1_Prediction.py

The core prediction page: single-record input form (grouped into
expandable sections with tooltips and validated ranges), a batch CSV
upload mode, a probability gauge chart, downloadable results, and a
session-based prediction history table.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.feature_info import FEATURE_METADATA, features_by_group
from utils.model_utils import load_feature_names, predict_batch, predict_single, risk_level
from utils.styling import inject_css, render_disclaimer, render_sidebar_branding, result_banner

st.set_page_config(page_title="Prediction | Breast Cancer Predictor", page_icon="🔬", layout="wide")
inject_css()
render_sidebar_branding()

st.title("🔬 Tumor diagnosis prediction")
render_disclaimer()

if "history" not in st.session_state:
    st.session_state.history = []

mode = st.radio("Prediction mode", ["Single patient", "Batch (CSV upload)"], horizontal=True)


def gauge_chart(confidence: float, diagnosis: str) -> go.Figure:
    """Build a Plotly probability gauge for the predicted class."""

    color = "#1d9e75" if diagnosis == "benign" else "#d85a30"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%"},
            title={"text": f"Confidence ({diagnosis})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(127,127,127,0.15)"},
                    {"range": [50, 75], "color": "rgba(127,127,127,0.25)"},
                    {"range": [75, 100], "color": "rgba(127,127,127,0.35)"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ---------------------------------------------------------------------
# Single patient mode
# ---------------------------------------------------------------------
if mode == "Single patient":
    st.markdown("Enter the 30 cell-nucleus measurements below, grouped by category.")

    input_values: dict[str, float] = {}
    group_labels = {"mean": "Mean values", "se": "Standard error values", "worst": "Worst (largest) values"}

    for group, label in group_labels.items():
        with st.expander(label, expanded=(group == "mean")):
            cols = st.columns(2)
            for i, feature_name in enumerate(features_by_group(group)):
                meta = FEATURE_METADATA[feature_name]
                with cols[i % 2]:
                    input_values[feature_name] = st.number_input(
                        meta["label"],
                        min_value=float(meta["min_value"]),
                        max_value=float(meta["max_value"]),
                        value=float(meta["default"]),
                        step=float(meta["step"]),
                        help=meta["description"],
                        key=f"input_{feature_name}",
                    )

    predict_clicked = st.button("Predict", type="primary", use_container_width=False)

    if predict_clicked:
        # Input validation: ensure every value is within its plausible range
        out_of_range = [
            name
            for name, val in input_values.items()
            if not (FEATURE_METADATA[name]["min_value"] <= val <= FEATURE_METADATA[name]["max_value"])
        ]
        if out_of_range:
            st.error(f"The following values are outside medically plausible ranges: {out_of_range}")
        else:
            with st.spinner("Running Random Forest inference..."):
                diagnosis, confidence, probabilities = predict_single(input_values)
                risk = risk_level(confidence, diagnosis)

            st.success("Prediction complete.")
            result_banner(diagnosis, confidence, risk)

            gcol, tcol = st.columns([1, 1])
            with gcol:
                st.plotly_chart(gauge_chart(confidence, diagnosis), use_container_width=True)
            with tcol:
                st.markdown("#### Class probabilities")
                prob_df = pd.DataFrame(
                    {"Class": ["Malignant", "Benign"], "Probability": [probabilities[0], probabilities[1]]}
                )
                st.bar_chart(prob_df.set_index("Class"))

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "diagnosis": diagnosis,
                "confidence": round(confidence, 4),
                "risk_level": risk,
                **input_values,
            }
            st.session_state.history.append(record)

            report_lines = [
                "Breast Cancer Prediction Report",
                "=" * 40,
                f"Generated: {record['timestamp']}",
                f"Prediction: {diagnosis.upper()}",
                f"Confidence: {confidence:.2%}",
                f"Risk level: {risk}",
                "",
                "Input values:",
            ] + [f"  {k}: {v}" for k, v in input_values.items()]
            report_text = "\n".join(report_lines)

            st.download_button(
                "Download report (.txt)",
                data=report_text,
                file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

# ---------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------
else:
    st.markdown(
        "Upload a CSV file containing the 30 required feature columns "
        "(same names as the WDBC dataset, e.g. `mean radius`, `radius error`, `worst radius`, ...)."
    )
    required_cols = load_feature_names()
    st.caption(f"Required columns ({len(required_cols)}): " + ", ".join(required_cols))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            with st.spinner("Running batch inference..."):
                results = predict_batch(df)
            st.success(f"Predicted {len(results)} record(s).")
            st.dataframe(results, use_container_width=True)

            csv_buffer = StringIO()
            results.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download results (.csv)",
                data=csv_buffer.getvalue(),
                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any parsing error to the user
            st.error(f"Could not process the uploaded file: {exc}")

# ---------------------------------------------------------------------
# Prediction history (session only)
# ---------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("Prediction history (this session)")
    hist_df = pd.DataFrame(st.session_state.history)[["timestamp", "diagnosis", "confidence", "risk_level"]]
    st.dataframe(hist_df, use_container_width=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
