"""
app.py

Home page and entry point for the Breast Cancer Prediction System
Streamlit application. Run with:

    streamlit run app.py

Other pages (Prediction, About Model, Help) live in pages/ and are
picked up automatically by Streamlit's multipage app mechanism.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.model_utils import load_metrics
from utils.styling import inject_css, render_sidebar_branding

st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar_branding()

st.title("🩺 Breast Cancer Prediction System")
st.caption("A Random Forest based diagnostic support tool - BSc final year project")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        <div class="bcp-card">
        <h3>What is breast cancer prediction?</h3>
        <p>
        Breast cancer diagnosis traditionally relies on a specialist examining
        digitized images of a fine needle aspirate (FNA) of a breast mass and
        measuring characteristics of the cell nuclei present, such as their
        radius, texture, perimeter, and concavity. This application uses a
        <strong>Random Forest</strong> machine learning model trained on these
        measurements to predict whether a tumor is <strong>benign</strong>
        (non-cancerous) or <strong>malignant</strong> (cancerous), acting as a
        second-opinion decision-support tool rather than a replacement for
        clinical diagnosis.
        </p>
        </div>

        <div class="bcp-card">
        <h3>Purpose of this system</h3>
        <p>
        This project demonstrates a complete, end-to-end applied machine
        learning workflow: data exploration, preprocessing, model training
        with hyperparameter tuning, rigorous evaluation, and deployment as an
        interactive web application - all built on a well-established
        benchmark dataset from the UCI Machine Learning Repository.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="bcp-card">
        <h3>Model used</h3>
        <p><strong>Random Forest Classifier</strong> (scikit-learn), tuned via
        5-fold cross-validated grid search over the number of trees, tree
        depth, and split criteria.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bcp-card">
        <h3>Dataset</h3>
        <p><strong>Wisconsin Diagnostic Breast Cancer (WDBC)</strong><br>
        569 samples &middot; 30 numerical features &middot; binary
        classification (benign / malignant)<br>
        Source: UCI Machine Learning Repository, via
        <code>sklearn.datasets.load_breast_cancer()</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

metrics = load_metrics()
if metrics:
    st.divider()
    st.subheader("Model performance at a glance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    m2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
    m3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
    m4.metric("F1 score", f"{metrics.get('f1_score', 0):.1%}")
    m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.1%}")
else:
    st.info(
        "No trained model metrics found yet. Run `python utils/train_model.py` "
        "to train the model and populate this dashboard."
    )

st.divider()
st.markdown("Use the sidebar to navigate to **Prediction**, **About Model**, or **Help**.")
