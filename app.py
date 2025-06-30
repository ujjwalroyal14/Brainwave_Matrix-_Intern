# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("credit_card_fraud_model.joblib")

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to predict if it's **fraudulent**.")

# Input sliders for PCA features V1 to V28
v_features = [f"V{i}" for i in range(1, 29)]
v_inputs = []

st.subheader("🔧 Enter PCA-transformed V1–V28 values:")
for feature in v_features:
    val = st.slider(feature, min_value=-30.0, max_value=30.0, value=0.0, step=0.1)
    v_inputs.append(val)

# Input for scaled Time and Amount
st.subheader("⏰ Enter Transaction Details")
scaled_time = st.number_input("Scaled Time", value=0.0, step=0.1)
scaled_amount = st.number_input("Scaled Amount", value=0.0, step=0.1)

# Create DataFrame for prediction
input_array = np.array([scaled_time, scaled_amount] + v_inputs).reshape(1, -1)
columns = ['scaled_time', 'scaled_amount'] + v_features
input_df = pd.DataFrame(input_array, columns=columns)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob:.2%})")

st.markdown("---")
st.info("This is a demo using a trained Random Forest model on PCA-transformed credit card data.")
