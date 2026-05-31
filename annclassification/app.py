from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error(
        "TensorFlow is not installed in the Python environment running Streamlit. "
        "This app needs TensorFlow to load `model.h5`.\n\n"
        "Fix: use a Python version supported by TensorFlow (recommended: Python 3.11). "
        "On Streamlit Community Cloud, add a `runtime.txt` like: `python-3.11.9`."
    )
    st.stop()

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_assets():
    model_path = BASE_DIR / "model.h5"
    if not model_path.exists():
        st.error(f"Missing model file: {model_path}")
        st.stop()

    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as exc:
        st.error(f"Failed to load model.h5: {exc}")
        st.stop()

    try:
        with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as file:
            label_encoder_gender = pickle.load(file)

        with open(BASE_DIR / "onehot_encoder_geo.pkl", "rb") as file:
            onehot_encoder_geo = pickle.load(file)

        with open(BASE_DIR / "scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
    except FileNotFoundError as exc:
        st.error(f"Missing required preprocessing file: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load preprocessing artifacts: {exc}")
        st.stop()

    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()


st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="collapsed",
)

st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

:root {
    --ink: #0a0c10;
    --slate: #dfe6ee;
    --mist: #f8f9fb;
    --accent: #b97851;
    --accent-soft: rgba(185, 120, 81, 0.12);
    --card: rgba(255, 255, 255, 0.86);
    --shadow: 0 16px 50px rgba(15, 23, 42, 0.1);
}

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--ink);
}

.stApp * {
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(1400px 420px at 5% -10%, #fff4e9 0%, transparent 65%),
        radial-gradient(1200px 520px at 95% 0%, #eff5ff 0%, transparent 55%),
        linear-gradient(180deg, #fdfdfe 0%, #f2f5f9 45%, #eef1f5 100%);
}

.app-shell {
    background: var(--card);
    border-radius: 28px;
    padding: 28px 32px 24px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(12, 17, 29, 0.08);
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    margin: 0;
    color: var(--ink);
}

.hero-sub {
    font-size: 15px;
    color: rgba(10, 12, 16, 0.7);
    margin-top: 6px;
}

.tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: #7a4a2e;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.metric-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 16px 18px;
    border: 1px solid rgba(12, 17, 29, 0.08);
}

.metric-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: rgba(10, 12, 16, 0.6);
}

.metric-value {
    font-size: 26px;
    font-weight: 600;
    margin-top: 4px;
}

.metric-note {
    font-size: 13px;
    color: rgba(10, 12, 16, 0.65);
    margin-top: 6px;
}

.stButton > button {
    background: var(--ink);
    color: #fff;
    border-radius: 999px;
    padding: 10px 22px;
    border: none;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.stButton > button:hover {
    background: #1c232c;
}

.stSlider > div[data-baseweb="slider"] > div {
    color: var(--accent);
}

.stSlider label, .stNumberInput label, .stSelectbox label {
    color: var(--ink) !important;
}

.stNumberInput input, .stSelectbox div[role="combobox"] {
    color: var(--ink) !important;
    background: #fff !important;
}

.stNumberInput input::placeholder {
    color: rgba(10, 12, 16, 0.5) !important;
}

.stNumberInput input {
    border-radius: 12px;
}

.footer-note {
    font-size: 12px;
    color: rgba(10, 12, 16, 0.6);
    margin-top: 14px;
}
</style>
""",
        unsafe_allow_html=True,
)

## streamlit app
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown('<span class="tag">Churn Intelligence</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Professional scoring for retail banking churn risk.</p>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Model</div>'
        '<div class="metric-value">ANN v1</div>'
        '<div class="metric-note">Binary classification</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

st.subheader("Customer Profile")
input_left, input_mid, input_right = st.columns([1.2, 1, 1])

with input_left:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 32)
    tenure = st.slider('Tenure (years)', 0, 10, 3)

with input_mid:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    balance = st.number_input('Balance', min_value=0.0, value=55000.0, step=500.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=75000.0, step=500.0)

with input_right:
    num_of_products = st.slider('Number of Products', 1, 4, 2)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


action_left, action_right = st.columns([1, 2])
with action_left:
    run_prediction = st.button("Generate Risk Score")

if run_prediction:
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])
    churn_label = 'Likely to churn' if prediction_proba > 0.5 else 'Likely to stay'
    risk_tone = '#b44b3a' if prediction_proba > 0.5 else '#1b6b4e'

    with action_right:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Churn Probability</div>
  <div class="metric-value" style="color: {risk_tone};">{prediction_proba:.2%}</div>
  <div class="metric-note">{churn_label}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer-note">Scores are generated from the trained ANN model and should be interpreted alongside business context.</div>',
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)
