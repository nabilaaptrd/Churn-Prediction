import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load the trained model and scaler
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("CNN+BiLSTM+SMOTEENN.h5")

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# Web app title
st.title("Prediksi Churn Pelanggan di Sektor Industri Telekomunikasi")

# Sidebar for input features
st.sidebar.header("Input Features")

# Input function for user data
def get_user_input():
    # Categorical features
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    # Numerical features
    senior_citizen = st.sidebar.slider("Senior Citizen (0 = No, 1 = Yes)", 0, 1, 0)
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0, 150, 50)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

    # Convert categorical data into numerical where needed
    categorical_mapping = {
        "Yes": 1, "No": 0,
        "Male": 0, "Female": 1,
        "No phone service": 2,
        "No internet service": 2,
        "DSL": 0, "Fiber optic": 1, "No": 2,
        "Month-to-month": 0, "One year": 1, "Two year": 2,
        "Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3,
    }

    data = {
        "gender": categorical_mapping[gender],
        "partner": categorical_mapping[partner],
        "dependents": categorical_mapping[dependents],
        "phone_service": categorical_mapping[phone_service],
        "multiple_lines": categorical_mapping[multiple_lines],
        "internet_service": categorical_mapping[internet_service],
        "online_security": categorical_mapping[online_security],
        "online_backup": categorical_mapping[online_backup],
        "device_protection": categorical_mapping[device_protection],
        "tech_support": categorical_mapping[tech_support],
        "streaming_tv": categorical_mapping[streaming_tv],
        "streaming_movies": categorical_mapping[streaming_movies],
        "contract": categorical_mapping[contract],
        "paperless_billing": categorical_mapping[paperless_billing],
        "payment_method": categorical_mapping[payment_method],
        "senior_citizen": senior_citizen,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
    }

    return pd.DataFrame(data, index=[0])

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("User Input")
st.write(user_input)

# Preprocess user input
scaled_input = scaler.transform(user_input)

# Predict churn
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    churn_probability = prediction[0][0]
    st.subheader("Prediction")
    if churn_probability > 0.5:
        st.error(f"The customer is likely to churn with a probability of {churn_probability:.2f}")
    else:
        st.success(f"The customer is unlikely to churn with a probability of {1 - churn_probability:.2f}")

# Footer
st.markdown("---")
st.markdown("Built by Nabila")
