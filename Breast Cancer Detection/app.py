import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import sklearn.datasets

# -------------------------------
# Load model and scaler
# -------------------------------
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# Load feature names
dataset = sklearn.datasets.load_breast_cancer()
feature_names = dataset.feature_names

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

st.title("🩺 Breast Cancer Detection System")
st.write("AI-based medical decision support tool")

st.markdown("---")

st.subheader("📋 Enter Patient Tumor Measurements")

# Collect inputs
user_inputs = []

for feature in feature_names:
    value = st.number_input(
        label=feature.replace("_", " ").title(),
        min_value=0.0,
        format="%.4f"
    )
    user_inputs.append(value)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Cancer Risk"):
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    probability = model.predict(input_scaled)[0][0]
    prediction = 1 if probability > 0.5 else 0

    st.markdown("---")

    if prediction == 1:
        st.success("🟢 **Benign (Non-Cancerous)**")
    else:
        st.error("🔴 **Malignant (Cancerous)**")

    st.write(f"### 📊 Confidence Level: **{probability*100:.2f}%**")

    st.warning(
        "⚠️ This tool is for educational purposes only and "
        "should not replace professional medical diagnosis."
    )
