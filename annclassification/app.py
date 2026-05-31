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
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
)

# UI theme + layout styles
st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600&display=swap');

:root {
    --ink: #0b1220;
    --muted: rgba(11, 18, 32, 0.65);
    --accent: #2f6df6;
    --accent-strong: #2549d7;
    --accent-soft: rgba(47, 109, 246, 0.12);
    --card: rgba(255, 255, 255, 0.92);
    --border: rgba(15, 23, 42, 0.12);
    --shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
}

.stApp * {
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(1100px 420px at 5% -5%, #e8f0ff 0%, transparent 70%),
        radial-gradient(900px 360px at 95% 0%, #f1edff 0%, transparent 65%),
        linear-gradient(180deg, #f7f9ff 0%, #eef3ff 55%, #f6f8fb 100%);
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}

.app-shell {
    background: var(--card);
    border-radius: 28px;
    padding: 26px 30px 24px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 40px;
    margin: 0;
    color: #000;
}

.hero-sub {
    font-size: 15px;
    color: var(--muted);
    margin-top: 6px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent-strong);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.card {
    background: #fff;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid var(--border);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--muted);
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    margin-top: 6px;
}

.metric-note {
    font-size: 13px;
    color: var(--muted);
    margin-top: 6px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
}

.section-sub {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 12px;
}

.stButton > button {
    background: var(--accent);
    color: #fff;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
    letter-spacing: 0.02em;
    box-shadow: 0 10px 20px rgba(47, 109, 246, 0.22);
}

.stButton > button:hover {
    background: var(--accent-strong);
}

.stSlider label, .stNumberInput label, .stSelectbox label {
    color: var(--ink) !important;
    font-weight: 600;
}

.stNumberInput input, .stSelectbox div[role="combobox"] {
    color: var(--ink) !important;
    background: #fff !important;
    border-radius: 12px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #fff !important;
}

.stNumberInput input::placeholder {
    color: rgba(11, 18, 32, 0.45) !important;
}

.footer-note {
    font-size: 12px;
    color: var(--muted);
    margin-top: 18px;
}
</style>
""",
        unsafe_allow_html=True,
)

## streamlit app
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

# Hero section
hero_left = st.columns([1])[0]
with hero_left:
    st.markdown('<h1 class="hero-title">Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Predict churn risk for retail banking customers with an ANN model and explainable signals.</p>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Capture customer attributes to evaluate churn risk.</div>',
    unsafe_allow_html=True,
)

profile_cols = st.columns([1, 1, 1])
with profile_cols[0]:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 32)
    tenure = st.slider('Tenure (years)', 0, 10, 3)

with profile_cols[1]:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    balance = st.number_input('Balance', min_value=0.0, value=55000.0, step=500.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=75000.0, step=500.0)

with profile_cols[2]:
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


st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Generate a churn score, risk band, and recommended action.</div>',
    unsafe_allow_html=True,
)

action_left, action_right = st.columns([1, 2])
with action_left:
    run_prediction = st.button("Generate Risk Score")

if run_prediction:
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    if prediction_proba < 0.35:
        risk_label = "Low risk"
        risk_tone = "#1b6b4e"
        recommendation = "Maintain current engagement and monitor usage trends."
    elif prediction_proba < 0.65:
        risk_label = "Medium risk"
        risk_tone = "#d9822b"
        recommendation = "Trigger personalized retention offers and proactive outreach."
    else:
        risk_label = "High risk"
        risk_tone = "#c0392b"
        recommendation = "Escalate to relationship manager and prioritize retention plan."

    with action_right:
        st.markdown(
            f"""
<div class="card">
  <div class="metric-label">Churn Probability</div>
  <div class="metric-value" style="color: {risk_tone};">{prediction_proba:.2%}</div>
  <div class="metric-note">Risk level: <strong style="color:{risk_tone};">{risk_label}</strong></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.progress(min(max(prediction_proba, 0.0), 1.0))
        st.markdown(
            f"""
<div class="card" style="margin-top: 14px;">
  <div class="metric-label">Recommendation</div>
  <div class="metric-note">{recommendation}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Why this prediction section
    st.markdown('<div class="section-title">Why This Prediction?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Top contributing factors based on standardized feature impact.</div>',
        unsafe_allow_html=True,
    )

    feature_names = list(input_data.columns)
    importances = np.abs(input_data_scaled[0])
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": importances,
    }).sort_values("Impact", ascending=False)

    top_factors = importance_df.head(5)
    st.bar_chart(top_factors.set_index("Feature"))

    factor_notes = ", ".join(top_factors["Feature"].tolist())
    st.markdown(
        f"<div class=\"card\"><div class=\"metric-label\">Top Drivers</div><div class=\"metric-note\">{factor_notes}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Summary of evaluation metrics and dataset context.</div>',
    unsafe_allow_html=True,
)

perf_cols = st.columns(3)
with perf_cols[0]:
    st.markdown(
        '<div class="card"><div class="metric-label">Dataset</div>'
        '<div class="metric-value">10,000+</div>'
        '<div class="metric-note">Retail bank customers</div></div>',
        unsafe_allow_html=True,
    )
with perf_cols[1]:
    st.markdown(
        '<div class="card"><div class="metric-label">Features</div>'
        '<div class="metric-value">12</div>'
        '<div class="metric-note">Behavior + finance</div></div>',
        unsafe_allow_html=True,
    )
with perf_cols[2]:
    st.markdown(
        '<div class="card"><div class="metric-label">Target</div>'
        '<div class="metric-value">Churn</div>'
        '<div class="metric-note">Binary label</div></div>',
        unsafe_allow_html=True,
    )

stack_cols = st.columns([1, 1])
with stack_cols[0]:
    st.markdown(
        '<div class="card"><div class="metric-label">Technology Stack</div>'
        '<div class="metric-note">Streamlit, TensorFlow, Scikit-learn, Pandas, NumPy</div></div>',
        unsafe_allow_html=True,
    )
with stack_cols[1]:
    st.markdown(
        '<div class="card"><div class="metric-label">Notes</div>'
        '<div class="metric-note">Metrics shown are representative validation results.</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)
