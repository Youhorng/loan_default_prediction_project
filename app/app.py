import streamlit as st  # type: ignore
import pandas as pd
import joblib  # type: ignore

@st.cache_data
def load_model():
    return joblib.load("../models/xgb_model_loan_defaulting_prediction.pkl")

model = load_model()

st.title("🤖 Loan Default Prediction")
st.sidebar.header("Enter Customer Information")

input_data = {}

# Example for debt_income_ratio
debt_income_ratio_input = st.sidebar.text_input("Enter Debt to Income Ratio (can be blank)")

if debt_income_ratio_input.strip() == "":
    input_data["debt_income_ratio"] = 0  # or a placeholder like 0 or -1 if used during training
    input_data["debt_income_ratio_missing"] = 1
else:
    try:
        input_data["debt_income_ratio"] = float(debt_income_ratio_input)
        input_data["debt_income_ratio_missing"] = 0
    except ValueError:
        st.error("Please enter a valid number for Debt to Income Ratio.")
