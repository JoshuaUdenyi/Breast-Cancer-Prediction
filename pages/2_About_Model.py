"""
pages/2_About_Model.py

Displays dataset details, the trained model's evaluation metrics,
saved evaluation plots (confusion matrix, ROC curve, feature
importance), and a comparison against alternative classifiers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.model_utils import load_feature_names, load_metrics
from utils.styling import inject_css, render_sidebar_branding

st.set_page_config(page_title="About Model | Breast Cancer Predictor", page_icon="📊", layout="wide")
inject_css()
render_sidebar_branding()

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"

st.title("📊 About the model")

st.markdown(
    """
    <div class="bcp-card">
    <h3>Dataset details</h3>
    <ul>
    <li><strong>Name:</strong> Wisconsin Diagnostic Breast Cancer (WDBC)</li>
    <li><strong>Samples:</strong> 569</li>
    <li><strong>Features:</strong> 30 numerical measurements of cell nuclei
    (10 base measurements &times; mean / standard error / worst)</li>
    <li><strong>Target classes:</strong> malignant, benign</li>
    <li><strong>Source:</strong> UCI Machine Learning Repository, via
    <code>sklearn.datasets.load_breast_cancer()</code></li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = load_metrics()

if not metrics:
    st.warning("No metrics found. Run `python utils/train_model.py` to train the model first.")
else:
    st.subheader("Evaluation metrics (held-out test set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    m2.metric("Precision", f"{metrics.get('precision', 0):.2%}")
    m3.metric("Recall", f"{metrics.get('recall', 0):.2%}")
    m4.metric("F1 score", f"{metrics.get('f1_score', 0):.2%}")
    m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2%}")

    if "best_params" in metrics:
        st.markdown("**Best hyperparameters found (GridSearchCV):**")
        st.json(metrics["best_params"])

    with st.expander("Full classification report"):
        st.text(metrics.get("classification_report", "Not available"))

    st.divider()
    st.subheader("Evaluation plots")
    p1, p2 = st.columns(2)
    cm_path = IMAGES_DIR / "confusion_matrix.png"
    roc_path = IMAGES_DIR / "roc_curve.png"
    if cm_path.exists():
        p1.image(str(cm_path), caption="Confusion matrix", use_container_width=True)
    if roc_path.exists():
        p2.image(str(roc_path), caption="ROC curve", use_container_width=True)

    st.subheader("Feature importance")
    importance = metrics.get("feature_importance")
    if importance:
        imp_df = pd.DataFrame(
            {"feature": list(importance.keys()), "importance": list(importance.values())}
        ).sort_values("importance", ascending=True)
        fig = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Top feature importances (Random Forest)",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=500, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Model comparison")
    st.caption(
        "Random Forest was selected as the final model based on the balance "
        "of accuracy, recall (sensitivity to malignant cases), and ROC-AUC, "
        "along with its built-in feature importance and lower variance "
        "compared to a single decision tree."
    )
    comparison = metrics.get("model_comparison")
    if comparison:
        comp_df = pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "model"})
        st.dataframe(comp_df.style.format({c: "{:.2%}" for c in comp_df.columns if c != "model"}), use_container_width=True)

        comp_img = IMAGES_DIR / "model_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Model comparison chart", use_container_width=True)

st.divider()
with st.expander("All 30 feature names used by the model"):
    st.write(load_feature_names())
