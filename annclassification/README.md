# 📊 Retail Bank Customer Churn Prediction Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://customer-churn-prediction-k7ejpc7cvz9kvv2dy8utb9.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A portfolio-ready production analytics dashboard that leverages a deep learning **Artificial Neural Network (ANN)** to predict individual bank customer churn. Built to mimic an enterprise banking application, it provides risk stratification, custom business mitigation recommendations, and localized feature-impact explainability hints.

🔴 **[Live Interactive Demo](https://customer-churn-prediction-k7ejpc7cvz9kvv2dy8utb9.streamlit.app/)**

---

## ✨ Key Features & Highlights

* 🧠 **Deep Learning Inference:** Implements a fully connected Multi-Layer Perceptron (MLP) built with TensorFlow/Keras.
* 🚦 **Dynamic Risk Stratification:** Automatically calculates and flags profiles into **Low**, **Medium**, and **High** risk bands.
* 📈 **Actionable Intelligence:** Renders tailor-made business retention recommendations mapped directly to the predicted risk profile.
* 🔒 **Production Pipeline Hygiene:** Employs strictly isolated, serialized data-preprocessing artifacts (`StandardScaler`, `OneHotEncoder`) to completely eradicate training data leakage during inference.
* 🧪 **End-to-End Reproducibility:** Clean Jupyter Notebook workflows documenting every milestone from Exploratory Data Analysis (EDA) to structural optimization.

---

## 📸 Product Interface

| User Inputs & Probability Assessment | Financial Impact & Explainability Hints |
| :---: | :---: |
| ![Dashboard - Inputs and Prediction](Screenshot%202026-05-31%20185156.png) | ![Dashboard - Explainability and Metrics](Screenshot%202026-05-31%20185200.png) |

---

## 🛠️ Built with the Modern ML Stack

* **Core Language:** `Python 3.11`
* **Deep Learning Engine:** `TensorFlow` / `Keras`
* **Data Engineering & Scaling:** `Scikit-learn`, `Pandas`, `NumPy`
* **Web Delivery Architecture:** `Streamlit Cloud`

---

## 🗺️ Machine Learning Pipeline Architecture

The application implements a strict stateless inference loop ensuring production consistency:
