"""Streamlit interface for Emotion-Aware Customer Feedback Analysis."""

import streamlit as st

st.set_page_config(page_title="Emotion-Aware Feedback", layout="wide")

st.title("Emotion-Aware Customer Feedback Analysis")
st.write(
    "Use this app to submit customer feedback and explore predicted emotions, key themes, "
    "and interpretability insights powered by SHAP/LIME."
)

feedback = st.text_area("Enter customer feedback", height=150)

if st.button("Analyze Feedback"):
    if not feedback.strip():
        st.warning("Please enter feedback text before running the analysis.")
    else:
        st.info("Model inference pipeline not yet implemented. Results will appear here.")
