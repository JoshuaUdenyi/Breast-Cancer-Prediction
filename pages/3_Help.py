"""
pages/3_Help.py

Usage instructions and the medical disclaimer.
"""

from __future__ import annotations

import streamlit as st

from utils.styling import inject_css, render_disclaimer, render_sidebar_branding

st.set_page_config(page_title="Help | Breast Cancer Predictor", page_icon="❓", layout="wide")
inject_css()
render_sidebar_branding()

st.title("❓ Help & usage guide")

st.markdown(
    """
    <div class="bcp-card">
    <h3>How to use this application</h3>
    <ol>
    <li>Go to the <strong>Prediction</strong> page from the sidebar.</li>
    <li>Choose <strong>Single patient</strong> to manually enter the 30 cell-nucleus
    measurements, grouped into Mean, Standard error, and Worst sections. Hover
    over the <code>?</code> icon next to each field for an explanation.</li>
    <li>Alternatively, choose <strong>Batch (CSV upload)</strong> to upload a CSV
    file containing multiple records at once and download the predictions.</li>
    <li>Click <strong>Predict</strong> to see the diagnosis, a confidence gauge,
    and a color-coded risk banner (🟢 benign / 🔴 malignant).</li>
    <li>Download a text report of a single prediction, or a CSV of batch
    results, using the download buttons provided.</li>
    <li>Visit the <strong>About Model</strong> page to review the dataset,
    evaluation metrics, feature importance, and how Random Forest compares
    against other algorithms.</li>
    </ol>
    </div>

    <div class="bcp-card">
    <h3>Where do these input values come from?</h3>
    <p>
    In a real clinical setting, the 30 features are computed automatically by
    image-analysis software from a digitized fine needle aspirate (FNA) image
    of a breast mass. For this academic demonstration, you can enter them
    manually or upload a CSV exported from such software (or from the WDBC
    dataset itself, for testing).
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Medical disclaimer")
render_disclaimer()

st.markdown(
    """
    <div class="bcp-card" style="margin-top:1rem;">
    <h3>Troubleshooting</h3>
    <ul>
    <li><strong>"model.pkl not found"</strong> - run
    <code>python utils/train_model.py</code> from the project root first, or
    run <code>notebooks/train_model.ipynb</code> end to end.</li>
    <li><strong>Batch upload error about missing columns</strong> - your CSV
    must contain all 30 feature columns with the exact names shown on the
    Prediction page (case-sensitive).</li>
    <li><strong>App looks unstyled</strong> - ensure <code>css/style.css</code>
    is present in the project folder; it is loaded automatically on every page.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
