import streamlit as st
import pandas as pd
import requests # To make requests to our FastAPI backend
import json

# --- Page Configuration ---
st.set_page_config(page_title='Loan Approval Predictor', page_icon=':bank:', layout='wide')

# --- Header ---
st.title('🏛️ Intelligent Loan Approval & Credit Risk Prediction System')
st.markdown("Welcome to the Loan Approval Predictor. Please enter the applicant's details below to get an instant loan approval prediction.")

# --- FastAPI Endpoint ---
FASTAPI_URL = "https://loan-api.onrender.com/predict/" # Adjust if your FastAPI runs on a different host/port

# --- Input Form ---
st.header('Applicant Information')

with st.form("loan_application_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income_annum = st.number_input('Annual Income (in INR)', min_value=100000, max_value=100000000, value=5000000, step=100000)
        loan_amount = st.number_input('Loan Amount (in INR)', min_value=100000, max_value=50000000, value=15000000, step=100000)
        loan_term = st.selectbox('Loan Term (in years)', options=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20], index=4) # Default to 10 years
        cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900, value=750, step=1)
        no_of_dependents = st.slider('Number of Dependents', min_value=0, max_value=5, value=2)
        education = st.selectbox('Education', options=['Graduate', 'Not Graduate'])

    with col2:
        residential_assets_value = st.number_input('Residential Assets Value (in INR)', min_value=0, max_value=100000000, value=5000000, step=100000)
        commercial_assets_value = st.number_input('Commercial Assets Value (in INR)', min_value=0, max_value=100000000, value=2000000, step=100000)
        luxury_assets_value = st.number_input('Luxury Assets Value (in INR)', min_value=0, max_value=100000000, value=10000000, step=100000)
        bank_asset_value = st.number_input('Bank Asset Value (in INR)', min_value=0, max_value=100000000, value=3000000, step=100000)
        self_employed = st.selectbox('Self Employed', options=['Yes', 'No'])

    submitted = st.form_submit_button("Get Loan Prediction")

    if submitted:
        # Prepare data for FastAPI
        input_data = {
            'loan_id': 0, # Placeholder, not used in prediction logic directly but needed for schema
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value,
            'loan_status': 'Unknown' # Placeholder, will be predicted
        }

        try:
            response = requests.post(FASTAPI_URL, json=input_data)
            response.raise_for_status() # Raise an exception for HTTP errors
            prediction_result = response.json()

            st.subheader('Prediction Result:')
            if prediction_result['loan_status'] == 'Approved':
                st.success(f"**Loan Status: {prediction_result['loan_status']}**")
                st.balloons()
            else:
                st.error(f"**Loan Status: {prediction_result['loan_status']}**")

            st.write(f"Probability of Approval: {prediction_result['probability_approved']:.2f}")
            st.write(f"Probability of Rejection: {prediction_result['probability_rejected']:.2f}")

            st.markdown("--- Request Details ---")
            st.json(input_data)
            st.markdown("--- Response Details ---")
            st.json(prediction_result)

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI backend at {FASTAPI_URL}. Please ensure the backend server is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error occurred: {e}. Response: {response.text}")
        except json.JSONDecodeError:
            st.error(f"Failed to decode JSON response from API. Response text: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

