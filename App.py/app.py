import streamlit as st
from joblib import load
import numpy as np

# Load model
data = load("fraud_model.joblib")
model = data["model"]
features = data["features"]

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

st.divider()

# Input section
inputs = []

for feature in features:
    val = st.number_input(f"{feature}", value=0.0, format="%.5f")
    inputs.append(val)

st.divider()

# Prediction
if st.button("🔍 Predict"):
    input_array = np.array(inputs).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Transaction Detected\n\nConfidence: {probability:.2%}")
    else:
        st.success(f"✅ Normal Transaction\n\nConfidence: {1 - probability:.2%}")