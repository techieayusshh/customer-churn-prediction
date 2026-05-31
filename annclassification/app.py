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


st.set_page_config(page_title="Customer Churn Prediction")

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
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


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
