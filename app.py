import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load artifacts
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
xgb_model = joblib.load("xgb_model.pkl")
encoder = load_model("encoder_model.keras")

st.title("🧴 Shampoo Launch Simulator")

st.sidebar.header("Customer Attributes")
age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Income", 20000, 150000, 50000)
family_size = st.sidebar.slider("Family Size", 1, 6, 3)
spend_cat_A = st.sidebar.slider("Spend Cat A", 0.0, 1.0, 0.5)
spend_cat_B = st.sidebar.slider("Spend Cat B", 0.0, 1.0, 0.5)
spend_cat_C = st.sidebar.slider("Spend Cat C", 0.0, 1.0, 0.5)

if st.button("Simulate Persona & Affinity"):
    X_new = np.array([[age, income, family_size, spend_cat_A, spend_cat_B, spend_cat_C]])
    X_scaled = scaler.transform(X_new)
    emb = encoder.predict(X_scaled)
    persona = kmeans.predict(emb)[0]
    affinity = xgb_model.predict_proba(emb)[0,1]
    st.success(f"Predicted Persona: {persona}")
    st.info(f"Affinity Score for Shampoo: {affinity:.2f}")
