import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Configure Streamlit page
st.set_page_config(page_title="💳 Credit Card Fraud Detector", layout="centered")


st.markdown("""
    <style>
        html, body, .stApp {
            height: 100%;
            margin: 0;
            background: linear-gradient(135deg, #6a00ff, #d9b3ff);  /* Violet to Lavender */
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }

        .stButton > button {
            background-color: #ff8c00;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            transition: 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #ffa733;
        }

        .stTextInput > div > input,
        .stNumberInput > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid white;
            border-radius: 5px;
        }

        h1, h2, h3 {
            color: white;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
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
