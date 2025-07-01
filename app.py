# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

# Custom background color using CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f9f9f9;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stSlider > div > div {
            color: #4CAF50;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter transaction details to predict if it's <b>fraudulent</b> or not.</p>", unsafe_allow_html=True)

# --- Feature Input Section ---

# PCA Features: V1 to V28
st.markdown("### 🔧 PCA-Transformed Features (V1–V28)")
v_features = [f"V{i}" for i in range(1, 29)]
v_inputs = []
cols = st.columns(3)  # Arrange sliders in 3 columns

for i, feature in enumerate(v_features):
    with cols[i % 3]:
        val = st.slider(feature, -30.0, 30.0, 0.0, 0.1)
        v_inputs.append(val)

# Time and Amount Inputs
st.markdown("### ⏰ Transaction Metadata")
scaled_time = st.number_input("Scaled Time (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
scaled_amount = st.number_input("Scaled Amount (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Create input DataFrame
input_array = np.array([scaled_time, scaled_amount] + v_inputs).reshape(1, -1)
columns = ['scaled_time', 'scaled_amount'] + v_features
input_df = pd.DataFrame(input_array, columns=columns)

# --- Load model ---
model = joblib.load("credit_card_fraud_model.joblib")

# --- Prediction Section ---
st.markdown("### 🚦 Prediction Result")

if st.button("🔍 Predict Now"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(f"""
            <div style='background-color:#e74c3c;padding:16px;border-radius:10px;color:white;'>
                <h3>⚠️ Fraud Detected!</h3>
                <b>Confidence:</b> {prob:.2%}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color:#27ae60;padding:16px;border-radius:10px;color:white;'>
                <h3>✅ Legitimate Transaction</h3>
                <b>Confidence:</b> {1 - prob:.2%}
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.info("📊 This is a demo app using a Random Forest model trained on PCA-transformed credit card transaction data.")
