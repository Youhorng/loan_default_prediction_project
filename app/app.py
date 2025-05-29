import streamlit as st  # type: ignore
import pandas as pd
import joblib  # type: ignore

# Store the model in cache to avoid reloading it every time
@st.cache_data
def load_model(path):
    return joblib.load(path)

model = load_model("../models/xgb_model_loan_defaulting_prediction.pkl")
model_columns = load_model("../models/xgb_model_features.pkl")

# Load median and mode values
median_df = pd.read_csv("../data/processed/median_values.csv")
mode_df = pd.read_csv("../data/processed/mode_values.csv")

# Set page configuration
st.title("🤖 Loan Default Prediction")
st.sidebar.header("Enter Customer Information")

# Helper function for flagged numerical inputs 
def get_flagged_numerical_input(label, feature_name, median_df):
    user_input = st.sidebar.text_input(f"{label} (leave empty for blank)")

    if user_input.strip() == "":
        value = median_df.loc[median_df["feature"] == feature_name, "median"].values[0]
        flag = 1
    else:
        try:
            value = float(user_input)
            flag = 0
        except ValueError:
            st.error(f"Please enter valid number for {label}")

    return value, flag


# Helper function for flagged categorical inputs
def get_flagged_categorical_input(label, feature_name, mode_df, options):
    user_input = st.sidebar.selectbox(f"{label} (select or leave blank)", [""] + options)

    if user_input.strip() == "":
        value = mode_df[feature_name].values[0]
        flag = 1
    else:
        value = user_input
        flag = 0
    
    return value, flag


# Helper function for required inputs
def get_required_numeric_input(label):
    user_input = st.sidebar.text_input(f"{label} (required)")

    if user_input.strip() == "":
        return None
        
    try:
        return float(user_input)
    except ValueError:
        st.error(f"Please enter a valid number for {label}.")


# Define feature lists
required_numeric = [
    ("Loan Amount", "loan_amount")
]

numeric_flag = [
    ("Mortgage Due", "mortgage_due"),
    ("Property Value", "property_value"),
    ("Years Employed", "years_on_job"),
    ("Number of Derogatory", "num_derogatory"),
    ("Number of Delinquencies", "num_delinquencies"),
    ("Credit Age in Month", "credit_age"),
    ("Number of Inquiries", "num_inquiries"),
    ("Number of Credit Lines", "num_credit_lines"),
    ("Debt to Income Ratio", "debt_income_ratio")
]

categorical_flag = [
    ("Loan Reason", "loan_reason", ["DebtCon", "HomeImp"]),
    ("Job Type", "job_type", ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"])
]

# Define a dictionary to hold user inputs
input_data = {}

# Collect required numeric inputs
for label, feature_name in required_numeric:
    input_data[feature_name] = get_required_numeric_input(label)

# Collect flagged numerical inputs
for label, feature_name in numeric_flag:
    value, flag = get_flagged_numerical_input(label, feature_name, median_df)
    input_data[feature_name] = value
    input_data[f"{feature_name}_missing_flag"] = flag

# Collect flagged categorical inputs
for label, feature_name, options in categorical_flag:
    value, flag = get_flagged_categorical_input(label, feature_name, mode_df, options)
    input_data[f"{feature_name}_missing_flag"] = flag

    # Add one-hot-encoding 
    for option in options:
        if value == option:
            input_data[f"{feature_name}_{option}"] = 1
        else:
            input_data[f"{feature_name}_{option}"] = 0

# Convert the input data to a DataFrame
df = pd.DataFrame([input_data])
st.write(df)

# Check model features and input features for valdiation
feature_names = model.get_booster().feature_names
missing_in_input = [col for col in feature_names if col not in df.columns]
extra_in_input = [col for col in df.columns if col not in feature_names]

if missing_in_input:
    st.error(f"Missing features: {missing_in_input}")
if extra_in_input:
    st.warning(f"Unexpected extra features: {extra_in_input}")

# Rearrange the input DataFrame to match the model's expected feature order
df = df.reindex(columns=feature_names, fill_value=0)

# Make prediction
y_pred = model.predict(df.head(1))
st.write(y_pred)
