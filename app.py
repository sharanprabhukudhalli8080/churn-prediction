import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer details")

# --- IMPORTANT ---
# You MUST match the number of features used in training
# If your dataset had 19 features → we create 19 inputs
# (Below is SAFE default using zeros for missing fields)

tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("Predict"):

    # Create FULL feature vector (same size as training data)
    # Assume model trained on 19 features → adjust if needed
    input_data = np.zeros((1, scaler.n_features_in_))

    # Put known values into correct positions
    input_data[0, 0] = tenure
    input_data[0, 1] = monthly
    input_data[0, 2] = total

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer is NOT likely to churn ✅")
