import streamlit as st
import pandas as pd
import joblib

# Load your model (adjust path as needed)
@st.cache_data
def load_model():
    return joblib.load("models/xgb_tuned_model.pkl")

model = load_model()

st.title("Loan Default Prediction")

st.sidebar.header("Enter Feature Values")
