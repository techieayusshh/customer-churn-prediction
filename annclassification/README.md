# 🏦 Customer Churn Prediction (ANN)

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-k7ejpc7cvz9kvv2dy8utb9.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A portfolio-ready Streamlit dashboard that predicts retail bank customer churn using a trained Artificial Neural Network (ANN). It delivers a clean UI, risk banding, and explainability hints so the model feels like a real banking analytics product.

🔗 **Live Demo:** [Launch the Streamlit App](https://customer-churn-prediction-k7ejpc7cvz9kvv2dy8utb9.streamlit.app/)

---

## ✨ Highlights

* **Instant Predictions:** Real-time churn probability with low/medium/high risk banding.
* **Actionable Insights:** Business recommendations tailored dynamically to the risk level.
* **Professional UI:** Dashboard design optimized with clean spacing, columns, and typography.
* **Production-Ready Pipeline:** Reusable preprocessing artifacts ensure consistent inference.
* **Reproducible Workflows:** Dedicated training and prediction notebooks included.

---

## 📸 Preview

![Dashboard - Inputs and Prediction](Screenshot%202026-05-31%20185156.png)
![Dashboard - Explainability and Metrics](Screenshot%202026-05-31%20185200.png)

---

## 📂 Project Structure

```text
annclassification/
 ├── app.py                    # Streamlit app
 ├── Churn_Modelling.csv       # Dataset (local use)
 ├── experiments.ipynb         # Training notebook
 ├── prediction.ipynb          # Prediction notebook
 ├── model.h5                  # Trained ANN model
 ├── label_encoder_gender.pkl  # Gender encoder
 ├── onehot_encoder_geo.pkl    # Geography encoder
 ├── scaler.pkl                # Feature scaler
 └── logs/                     # TensorBoard logs
