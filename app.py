import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Configure Streamlit page
st.set_page_config(page_title="💳 Credit Card Fraud Detector", layout="centered")

st.markdown("""
    <style>
        /* ---------- Static Gradient Background ---------- */
        html, body, .stApp {
            height: 100%;
            margin: 0;
            background: linear-gradient(135deg, #6a00ff, #d9b3ff);  /* Violet to Lavender */
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }

        h1, h2, h3 {
            color: white;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* ---------- Cool Gradient Predict Button ---------- */
        .stButton > button {
            background: linear-gradient(135deg, #ff512f, #dd2476);  /* Orange to Pink */
            color: #fff;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.7em 1.6em;
            box-shadow: 0 4px 14px rgba(221, 36, 118, 0.4);
            transition: all 0.25s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.03);
            box-shadow: 0 6px 20px rgba(221, 36, 118, 0.55);
        }

        .stButton > button:active {
            transform: translateY(1px) scale(0.98);
            box-shadow: 0 3px 10px rgba(221, 36, 118, 0.4);
        }

        /* ---------- Input Styling ---------- */
        .stTextInput > div > input,
        .stNumberInput > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid white;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict whether it's **fraudulent** or not.")

# V1–V28 Inputs
v_features = [f"V{i}" for i in range(1, 29)]
v_inputs = []

st.markdown("### 🔢 PCA Feature Inputs")
cols = st.columns(3)
for i, feature in enumerate(v_features):
    with cols[i % 3]:
        val = st.slider(feature, -30.0, 30.0, 0.0, 0.1)
        v_inputs.append(val)

# Time and Amount
st.markdown("### ⏱️ Transaction Metadata")
scaled_time = st.number_input("Scaled Time (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
scaled_amount = st.number_input("Scaled Amount (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Combine input
input_array = np.array([scaled_time, scaled_amount] + v_inputs).reshape(1, -1)
columns = ['scaled_time', 'scaled_amount'] + v_features
input_df = pd.DataFrame(input_array, columns=columns)

# Load model
try:
    model = joblib.load("credit_card_fraud_model.joblib")
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob:.2%})")

# Footer
st.markdown("---")
st.info("This app uses a trained Random Forest model on PCA-transformed credit card data for educational/demo purposes.")
