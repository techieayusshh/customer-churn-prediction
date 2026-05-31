# Customer Churn Prediction (ANN)

Portfolio-ready Streamlit dashboard that predicts retail bank customer churn using a trained Artificial Neural Network (ANN). It delivers a clean UI, risk banding, and explainability hints so the model feels like a real banking analytics product.

## Live Demo

- Streamlit app: https://customer-churn-prediction-k7ejpc7cvz9kvv2dy8utb9.streamlit.app/

## Highlights

- Instant churn probability with low/medium/high risk banding
- Business recommendation tailored to risk level
- Professional dashboard UI with clean spacing and typography
- Reusable preprocessing artifacts for consistent inference
- Training and prediction notebooks for reproducibility

## Preview

![Dashboard - Inputs and Prediction](Screenshot%202026-05-31%20185156.png)
![Dashboard - Explainability and Metrics](Screenshot%202026-05-31%20185200.png)

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

## Quickstart (Local)

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

Note: the repo includes a root `runtime.txt` pinned to Python 3.11.

## Data Overview

The dataset contains customer demographics, account details, and engagement signals used to predict churn.

Key feature groups:

- Customer profile: geography, gender, age, tenure
- Account value: balance, estimated salary
- Product usage: number of products, credit card, active member
- Financial signals: credit score

## Model Overview

- Type: Fully connected ANN
- Activation: ReLU (hidden), sigmoid (output)
- Loss: Binary cross-entropy
- Output: Churn probability in the range [0, 1]

If you retrain the model, update these artifacts:

- `model.h5`
- `scaler.pkl`
- `label_encoder_gender.pkl`
- `onehot_encoder_geo.pkl`

## Inference Flow

1) User inputs collected via Streamlit UI
2) Categorical fields encoded with saved encoders
3) Numeric fields scaled using the saved scaler
4) ANN model outputs churn probability
5) UI renders risk band + recommendation

## Notebooks

- `experiments.ipynb`: end-to-end training workflow
- `prediction.ipynb`: inference validation and examples

## Technology Stack

- Python 3.11
- Streamlit
- TensorFlow/Keras
- Scikit-learn
- Pandas, NumPy

## Troubleshooting

If TensorFlow fails to install on Streamlit Cloud:

- Confirm Python is set to 3.11
- Reboot the app after changing settings

If the app loads but predictions fail:

- Verify `model.h5` and the `.pkl` files exist in `annclassification/`

## License

MIT License (see LICENSE).