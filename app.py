import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "model.pkl")
DATA = pd.read_csv(ROOT / "heart_disease_uci.csv")
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="🫀")
st.title("Heart Disease Risk Prediction")
st.caption("Enter the clinical measurements below to estimate the model's predicted risk.")

with st.form("prediction"):
    values = {}
    cols = st.columns(2)
    for index, feature in enumerate(FEATURES):
        series = DATA[feature].dropna()
        with cols[index % 2]:
            if feature in CATEGORICAL:
                options = sorted(series.unique().tolist(), key=str)
                default_index = options.index(series.mode().iloc[0]) if not series.mode().empty else 0
                values[feature] = st.selectbox(feature, options, index=default_index)
            else:
                low, high = float(series.min()), float(series.max())
                median = float(series.median())
                step = 1.0 if feature in {"age", "trestbps", "chol", "thalch", "ca"} else 0.1
                values[feature] = st.number_input(feature, min_value=low, max_value=high, value=median, step=step)
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(MODEL.predict_proba(row)[0, 1])
    label = "High risk" if probability >= 0.5 else "Low risk"
    st.metric("Predicted probability of heart disease", f"{probability:.1%}")
    st.subheader(label)
    st.caption("Educational project, not medical advice.")
else:
    st.caption("Educational project, not medical advice.")
