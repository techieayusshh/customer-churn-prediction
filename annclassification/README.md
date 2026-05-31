# Customer Churn Prediction (ANN)

Modern Streamlit dashboard for predicting retail bank customer churn using a trained ANN model. Designed for portfolio-ready demos with clear inputs, risk scoring, and explainability cues.

## Demo

- Streamlit app: add your link here after deployment

## Features

- Churn probability with risk banding (low/medium/high)
- Business recommendation based on risk level
- Clean, dashboard-style UI for portfolio presentations
- Reusable preprocessing artifacts (encoders + scaler)

## Project Structure

```
annclassification/
	app.py                     # Streamlit app
	Churn_Modelling.csv         # Dataset (local use)
	experiments.ipynb           # Training notebook
	prediction.ipynb            # Prediction notebook
	model.h5                    # Trained ANN model
	label_encoder_gender.pkl    # Gender encoder
	onehot_encoder_geo.pkl      # Geography encoder
	scaler.pkl                  # Feature scaler
	logs/                       # TensorBoard logs
```

## Local Setup

1) Install dependencies

```
pip install -r requirements.txt
```

2) Run the app

```
streamlit run annclassification/app.py
```

3) Open in browser

```
http://localhost:8501
```

## Streamlit Community Cloud

Use these settings:

- Repository: techieayusshh/customer-churn-prediction
- Branch: main
- Main file path: annclassification/app.py
- Python version: 3.11

The repo already includes `runtime.txt` at the root for Python 3.11.

## Model Notes

- Architecture: 2 hidden layers with ReLU, sigmoid output
- Loss: binary cross-entropy
- Metrics: accuracy

If you retrain the model, replace `model.h5`, `scaler.pkl`, and encoder files.

## Tech Stack

- Python, Streamlit, TensorFlow/Keras
- Scikit-learn, Pandas, NumPy

## License

MIT License (see LICENSE).