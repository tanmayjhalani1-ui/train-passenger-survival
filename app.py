import json

import pandas as pd
import streamlit as st

from src.inference import load_model, predict_one
from src.train import METADATA_PATH

st.set_page_config(page_title="Passenger Survival Predictor", page_icon="🚆", layout="wide")

st.title("Passenger Survival Predictor")
st.caption("An end-to-end machine learning demonstration using the repository's original dataset.")

try:
    model = load_model()
except FileNotFoundError:
    st.warning("No trained model found. Train the model first using the command shown below.")
    st.code("python -m src.train")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Passenger details")
    age = st.number_input("Age", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
    fare = st.number_input("Fare", min_value=0.0, max_value=10000.0, value=50.0, step=1.0)
    gender = st.selectbox("Gender", ["Female", "Male"])
    seat_class = st.selectbox("Seat class", ["First", "Second", "Third"])
    submitted = st.button("Predict", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")
    if submitted:
        result = predict_one(model, age, fare, gender, seat_class)
        st.metric("Model output", result["label"])
        st.progress(result["survival_probability"])
        st.metric("Estimated probability", f"{result['survival_probability']:.1%}")
        st.caption("This is a model estimate from a small illustrative dataset, not a reliable real-world risk assessment.")
    else:
        st.info("Enter passenger details and select Predict.")

if METADATA_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    st.divider()
    st.subheader("Training run overview")
    metric_cols = st.columns(4)
    for column, key, label in zip(
        metric_cols, ["accuracy", "precision", "recall", "roc_auc"],
        ["Holdout accuracy", "Holdout precision", "Holdout recall", "Holdout ROC-AUC"]
    ):
        column.metric(label, f"{metadata['holdout_metrics'][key]:.3f}")
    st.caption(
        f"Selected model: {metadata['model_name']} · "
        f"{metadata['dataset_rows']} dataset rows · "
        "Metrics are illustrative because the dataset is very small."
    )
    with st.expander("Compare cross-validation ROC-AUC"):
        scores = pd.Series(metadata["cross_validation_roc_auc"], name="Mean CV ROC-AUC")
        st.bar_chart(scores)
