import streamlit as st  # type: ignore
import pandas as pd
import joblib  # type: ignore


# Load model with caching to improve performance
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Function to handle numerical inputs with a fallback to median values
def get_flagged_numerical_input(label, feature_name, median_df):
    user_input = st.text_input(f"{label}", key=feature_name)
    if user_input.strip() == "":  # If input is empty, use the median value
        value = median_df.loc[median_df["feature"] == feature_name, "median"].values[0]
        flag = 1  # Flag indicates that the default value was used
    else:
        try:
            value = float(user_input)  # Convert input to float
            flag = 0  # No flag since the user provided a valid input
        except ValueError:  # Handle invalid input
            st.warning(f"⚠️ Invalid number for {label}.")
            value = median_df.loc[median_df["feature"] == feature_name, "median"].values[0]
            flag = 1
    return value, flag


# Function to handle categorical inputs with a fallback to mode values
def get_flagged_categorical_input(label, feature_name, mode_df, options):
    user_input = st.selectbox(f"{label}", [""] + options, key=feature_name)
    if user_input.strip() == "":  # If input is empty, use the mode value
        value = mode_df[feature_name].values[0]
        flag = 1  # Flag indicates that the default value was used
    else:
        value = user_input
        flag = 0  # No flag since the user provided a valid input
    return value, flag


# Function to handle required numeric inputs
def get_required_numeric_input(label, feature_name):
    user_input = st.text_input(f"{label}*", key=feature_name)
    if user_input.strip() == "":  # If input is empty, show an error
        st.error(f"{label} is required.")
        return None
    try:
        return float(user_input)  # Convert input to float
    except ValueError:  # Handle invalid input
        st.error(f"{label} must be a valid number.")
        return None


# Function to validate input features against the model's expected features
def validate_features(input_df, model_features):
    # Check for missing features
    missing = [col for col in model_features if col not in input_df.columns]
    # Check for extra features
    extras = [col for col in input_df.columns if col not in model_features]

    if missing:
        st.error(f"🚫 Missing features: {missing}")
    if extras:
        st.warning(f"⚠️ Extra features ignored: {extras}")

    # Reorder columns to match the model's expected input and fill missing columns with 0
    return input_df.reindex(columns=model_features, fill_value=0)


# Function to add one-hot encoding for categorical features
def add_one_hot_encoding(input_data, feature_name, value, options):
    for option in options:
        # Create a binary column for each option
        input_data[f"{feature_name}_{option}"] = int(value == option)


# Main function to run the Streamlit app
def main():
    # Set up the Streamlit page
    st.set_page_config(page_title="Loan Default Predictor", page_icon="🤖", layout="centered")
    st.title("🤖 Loan Default Risk Prediction")

    # Load the model and related data
    model = load_model("../models/xgb_model_loan_defaulting_prediction.pkl")
    model_features = load_model("../models/xgb_model_features.pkl")
    median_df = pd.read_csv("../data/processed/median_values.csv")
    mode_df = pd.read_csv("../data/processed/mode_values.csv")

    # Define required numeric features
    required_numeric = [("Loan Amount", "loan_amount")]

    # Define flagged financial data features
    flag_financial_data = [
        ("Mortgage Due", "mortgage_due"),
        ("Property Value", "property_value"),
        ("Number of Derogatory", "num_derogatory"),
        ("Number of Delinquencies", "num_delinquencies"),
        ("Credit Age in Months", "credit_age"),
        ("Number of Inquiries", "num_inquiries"),
        ("Number of Credit Lines", "num_credit_lines"),
        ("Debt to Income Ratio", "debt_income_ratio"),                
    ]

    # Define flagged employment data features
    flag_employment_data = [
        ("Years Employed", "years_on_job"),   
        ("Job Type", "job_type", ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]),           
    ]

    # Define flagged loan purpose data features
    flag_loan_purpose_data = [
        ("Loan Reason", "loan_reason", ["DebtCon", "HomeImp"])       
    ]

    # Dictionary to store user inputs
    data_input = {}

    # Create a form for user input
    with st.form("prediction_form"):
        st.subheader("🔢 Loan Application Profile")
        
        # Collect inputs (same as before)
        with st.expander("🏦 Financial History"):
            for label, name in required_numeric:
                value = get_required_numeric_input(label, name)
                if value is not None:
                    data_input[name] = value

            for label, name in flag_financial_data:
                value, flag = get_flagged_numerical_input(label, name, median_df)
                data_input[name] = value
                data_input[f"{name}_missing_flag"] = flag

        with st.expander("👷 Employment Information"):
            for item in flag_employment_data:
                if len(item) == 3:
                    label, name, options = item
                    value, flag = get_flagged_categorical_input(label, name, mode_df, options)
                    data_input[f"{name}_missing_flag"] = flag
                    add_one_hot_encoding(data_input, name, value, options)
                else:
                    label, name = item
                    value, flag = get_flagged_numerical_input(label, name, median_df)
                    data_input[name] = value
                    data_input[f"{name}_missing_flag"] = flag

        with st.expander("💼 Loan Purpose"):
            for label, name, options in flag_loan_purpose_data:
                value, flag = get_flagged_categorical_input(label, name, mode_df, options)
                data_input[f"{name}_missing_flag"] = flag
                add_one_hot_encoding(data_input, name, value, options)

        # Only one button to submit and trigger prediction
        submitted = st.form_submit_button("Predict Default Risk")


    # Handle prediction immediately after form submit
    if submitted:
        if any(v is None for v in data_input.values()):
            st.warning("Please fix the input errors above.")
        else:
            df = pd.DataFrame([data_input])
            df = validate_features(df, model_features)

            if df.isnull().values.any():
                st.error("🚫 Input data contains NaN values. Please revise your input.")
            else:
                try:
                    y_pred = model.predict(df)[0]
                    y_pred_probab = model.predict_proba(df)[0]
                    if y_pred == 1:
                        prediction_label = "🔴 High Risk Defaulting"
                        prediction_probab = f"{y_pred_probab[1]:.2%}"
                    else:
                        prediction_label = "🟢 Low Risk Defaulting"
                        prediction_probab = f"{y_pred_probab[0]:.2%}"
                    st.success(f"**Prediction:** {prediction_label} with Probabilities of {prediction_probab}")
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")


# Run the app
if __name__ == "__main__":
    main()