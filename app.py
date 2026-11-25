import io
import re
from pathlib import Path

import joblib
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "model.pkl")
DATA = pd.read_csv(ROOT / "heart_disease_uci.csv")
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
LABELS = {
    "age": r"\bage\b", "sex": r"\b(?:sex|gender)\b",
    "cp": r"\b(?:chest\s*pain(?:\s*type)?|cp)\b",
    "trestbps": r"\b(?:resting\s*(?:blood\s*)?pressure|blood\s*pressure|trestbps)\b",
    "chol": r"\b(?:cholesterol|serum\s*chol|chol)\b",
    "fbs": r"\b(?:fasting\s*(?:blood\s*)?sugar|fasting\s*glucose|fbs)\b",
    "restecg": r"\b(?:resting\s*ecg|restecg|ecg)\b",
    "thalch": r"\b(?:max(?:imum)?\s*(?:heart\s*)?rate|thalch)\b",
    "exang": r"\b(?:exercise(?:-induced)?\s*angina|exang)\b",
    "oldpeak": r"\b(?:st\s*depression|oldpeak)\b",
    "slope": r"\b(?:st\s*slope|slope)\b",
    "ca": r"\b(?:major\s*vessels|number\s*of\s*vessels|ca)\b",
    "thal": r"\b(?:thalassemia|thal)\b",
}


def read_report(image):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image.getvalue()))).convert("RGB")
    return pytesseract.image_to_string(ImageOps.autocontrast(image))


def extract_values(text):
    found = {}
    for line in text.lower().splitlines():
        for feature, label in LABELS.items():
            match = re.search(label + r"\s*[:=\-]?\s*(.*)", line)
            if not match:
                continue
            value = match.group(1).strip()
            if feature in CATEGORICAL:
                choices = {
                    "sex": [("female", "Female"), ("male", "Male")],
                    "cp": [("atypical", "atypical angina"), ("non-anginal", "non-anginal"), ("typical", "typical angina"), ("asymptomatic", "asymptomatic")],
                    "fbs": [("true", True), ("yes", True), ("false", False), ("no", False)],
                    "restecg": [("lv hypertrophy", "lv hypertrophy"), ("st-t", "st-t abnormality"), ("abnormality", "st-t abnormality"), ("normal", "normal")],
                    "exang": [("yes", True), ("true", True), ("no", False), ("false", False)],
                    "slope": [("down", "downsloping"), ("flat", "flat"), ("up", "upsloping")],
                    "thal": [("revers", "reversable defect"), ("fixed", "fixed defect"), ("normal", "normal")],
                }[feature]
                parsed = next((choice for key, choice in choices if key in value), None)
                if parsed is None and feature in {"fbs", "exang"}:
                    binary = re.match(r"[01]\b", value)
                    parsed = binary is not None and binary.group() == "1"
                if parsed is not None:
                    found.setdefault(feature, set()).add(parsed)
            else:
                number = re.search(r"-?\d+(?:\.\d+)?", value)
                if number:
                    parsed = float(number.group())
                    if feature in {"age", "trestbps", "chol", "thalch", "ca"}:
                        parsed = int(parsed)
                    series = DATA[feature].dropna()
                    if series.min() <= parsed <= series.max():
                        found.setdefault(feature, set()).add(parsed)
            break
    return {key: next(iter(vals)) for key, vals in found.items() if len(vals) == 1}, [key for key, vals in found.items() if len(vals) > 1]


st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="🫀")
st.title("Heart Disease Risk Prediction")
st.caption("Enter measurements manually or upload photos of a printed report to prefill readable values.")
uploads = st.file_uploader("Upload one or more report photos", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if uploads:
    st.image(uploads, caption=[image.name for image in uploads], use_container_width=True)
    if st.button("Read values from photos"):
        texts = [read_report(image) for image in uploads]
        parsed, conflicts = {}, set()
        for text in texts:
            values, duplicate = extract_values(text)
            conflicts.update(duplicate)
            for name, value in values.items():
                if name in parsed and parsed[name] != value:
                    conflicts.add(name)
                else:
                    parsed[name] = value
        for name, value in parsed.items():
            if name not in conflicts:
                st.session_state[f"input_{name}"] = value
        st.session_state["ocr_summary"] = (parsed, sorted(conflicts))
if "ocr_summary" in st.session_state:
    parsed, conflicts = st.session_state["ocr_summary"]
    st.info("Photo text is only a suggestion. Review every field below before estimating risk.")
    st.write("Read from photos:", ", ".join(f"{k} = {v}" for k, v in parsed.items() if k not in conflicts) or "No recognized values")
    if conflicts:
        st.warning("Conflicting or ambiguous values need manual entry: " + ", ".join(conflicts))

with st.form("prediction"):
    values = {}
    cols = st.columns(2)
    for index, feature in enumerate(FEATURES):
        series = DATA[feature].dropna()
        with cols[index % 2]:
            if feature in CATEGORICAL:
                options = sorted(series.unique().tolist(), key=str)
                default = st.session_state.get(f"input_{feature}", series.mode().iloc[0])
                if default not in options:
                    default = series.mode().iloc[0]
                values[feature] = st.selectbox(feature, options, index=options.index(default), key=f"input_{feature}")
            else:
                low, high = float(series.min()), float(series.max())
                median = float(series.median())
                step = 1.0 if feature in {"age", "trestbps", "chol", "thalch", "ca"} else 0.1
                values[feature] = st.number_input(feature, min_value=low, max_value=high, value=float(st.session_state.get(f"input_{feature}", median)), step=step, key=f"input_{feature}")
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    probability = float(MODEL.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1])
    st.metric("Predicted probability of heart disease", f"{probability:.1%}")
    st.subheader("High risk" if probability >= 0.5 else "Low risk")
st.caption("Educational project, not medical advice.")
