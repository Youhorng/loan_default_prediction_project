import streamlit as st # type: ignore
import pandas as pd
import joblib # type: ignore

# Load your model (adjust path as needed)
@st.cache_data
def load_model():
    return joblib.load("../models/xgb_tuned_model.pkl")

model = load_model()

st.title("🤖 Loan Default Prediction")

st.sidebar.header("Enter Customer Information")
