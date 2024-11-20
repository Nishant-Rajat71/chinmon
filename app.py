import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title('BANK CHURN PREDICTION')

# Custom CSS for styling the form and button
st.markdown("""
    <style>
    /* Center the main container */
    .main .block-container {
        max-width: 600px;
        margin: auto;
    }
    /* Title styling */
    .title {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Button styling */
    .stButton > button {
        background-color: #28a745;
        color: white;
        width: 100%;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
    }
    /* Input styling */
    .stTextInput, .stNumberInput, .stSelectbox {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Creating the form
with st.form(key='churn_form'):
    age = st.number_input("Age:", min_value=0, max_value=120, value=0)
    credit_score = st.number_input("Credit Score:", min_value=0, max_value=1000, value=0)
    balance = st.number_input("Balance:", min_value=0, max_value=1000000, value=0)
    tenure = st.number_input("Tenure (Years with Bank):", min_value=0, max_value=100, value=0)
    num_products = st.number_input("Number of Products:", min_value=0, max_value=10, value=0)
    has_credit_card = st.selectbox("Has Credit Card (1 = Yes, 0 = No):", [1, 0])
    is_active_member = st.selectbox("Is Active Member (1 = Yes, 0 = No):", [1, 0])

    # Submit button for prediction
    submit_button = st.form_submit_button(label="Predict")

# Prediction logic
if submit_button:
    # Prepare the input features for the model
    input_features = np.array([[credit_score, age, tenure, balance, num_products, has_credit_card, is_active_member]])

    # Make prediction using the loaded model
    prediction = model.predict(input_features)[0]  # Assuming binary classification: 0 or 1
    prediction_text = "Likely to Churn" if prediction == 1 else "Unlikely to Churn"

    # Display the prediction result
    st.markdown(
        f"<div style='text-align: center; font-size: 18px; color: blue; margin-top: 20px;'>Prediction: <b>{prediction_text}</b></div>",
        unsafe_allow_html=True)
