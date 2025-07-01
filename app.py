# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

# 💡 Animated Background CSS (Option 2)
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(270deg, #f0fff4, #e3f2fd, #fff8f0);
            background-size: 600% 600%;
            animation: gradientMove 15s ease infinite;
            font-family: 'Segoe UI', sans-serif;
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stButton > button {
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }

        h1, h3 {
            color: #2c3e50;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and instructions
st.markdown("<h1 style='text-align: center;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter transaction details to predict if it's <b>fraudulent</b>.</p>", unsafe_allow_html=True)

# --- Feature Inputs ---

# PCA Features: V1 to V28
st.markdown("### 🔧 PCA-Transformed Features (V1–V28)")
v_features = [f"V{i}" for i in range(1, 29)]
v_inputs = []
cols = st.columns(3)  # 3 sliders per row

for i, feature in enumerate(v_features):
    with cols[i % 3]:
        val = st.slider(feature, -30.0, 30.0, 0.0, 0.1)
        v_inputs.append(val)

# Scaled Time and Amount
st.markdown("### ⏰ Transaction Metadata")
scaled_time = st.number_input("Scaled Time (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
scaled_amount = st.number_input("Scaled Amount (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Input DataFrame
input_array = np.array([scaled_time, scaled_amount] + v_inputs).reshape(1, -1)
columns = ['scaled_time', 'scaled_amount'] + v_features
input_df = pd.DataFrame(input_array, columns=columns)

# Load trained model
model = joblib.load("credit_card_fraud_model.joblib")

# --- Predict Button and Output ---
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
