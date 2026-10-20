"""
styling.py

Reusable UI helpers: CSS injection, page config, and small presentational
components shared across every page of the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


def inject_css() -> None:
    """Load and inject the custom stylesheet. Safe to call on every page."""

    css_path = ROOT / "css" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_sidebar_branding() -> None:
    """Render the small branding block shown at the top of the sidebar."""

    st.sidebar.markdown(
        """
        <div class="bcp-sidebar-title">🩺 Breast Cancer Predictor</div>
        <div class="bcp-sidebar-subtitle">Random Forest diagnostic support tool</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def render_disclaimer() -> None:
    """Render the standard medical disclaimer box."""

    st.markdown(
        """
        <div class="bcp-disclaimer">
        <strong>Medical disclaimer:</strong> This tool is a final-year academic
        project and is <em>not</em> a certified medical device. Predictions are
        generated from a statistical model trained on a public research dataset
        and must never replace evaluation by a qualified healthcare professional.
        Always consult a doctor for diagnosis and treatment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_banner(diagnosis: str, confidence: float, risk: str) -> None:
    """Render the color-coded prediction result banner."""

    if diagnosis == "benign":
        st.markdown(
            f"""
            <div class="bcp-result-benign">
            🟢 Predicted: Benign<br>
            <span style="font-size:0.95rem;font-weight:400;">
            Confidence: {confidence:.1%} &nbsp;|&nbsp; Risk level: {risk}
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="bcp-result-malignant">
            🔴 Predicted: Malignant<br>
            <span style="font-size:0.95rem;font-weight:400;">
            Confidence: {confidence:.1%} &nbsp;|&nbsp; Risk level: {risk}
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
