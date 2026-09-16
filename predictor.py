import joblib
import pandas as pd
from backend.preprocessing import preprocess_data # Assuming preprocessing.py is in the same directory

class LoanPredictor:
    def __init__(self, model_path='loan-risk-prediction/models/loan_model.pkl',
                 scaler_path='loan-risk-prediction/models/scaler.pkl',
                 feature_columns_path='loan-risk-prediction/models/feature_columns.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(feature_columns_path)

    def predict(self, raw_data: dict) -> dict:
        # Convert raw input data (e.g., from an API request) to a pandas DataFrame
        input_df = pd.DataFrame([raw_data])

        # Preprocess the input data using the trained scaler and feature columns
        processed_df = preprocess_data(input_df, is_train=False, 
                                       scaler=self.scaler, 
                                       feature_columns=self.feature_columns)

        # Make prediction
        prediction = self.model.predict(processed_df)
        prediction_proba = self.model.predict_proba(processed_df)

        # Convert prediction to 'Approved' or 'Rejected'
        loan_status = 'Rejected' if prediction[0] == 1 else 'Approved'
        
        # Return results
        return {
            'loan_status': loan_status,
            'probability_rejected': float(prediction_proba[0][1]),
            'probability_approved': float(prediction_proba[0][0])
        }

    def predict_batch(self, raw_data_list: list) -> list:
        # Convert list of raw input data to a pandas DataFrame
        input_df = pd.DataFrame(raw_data_list)

        # Preprocess the input data
        processed_df = preprocess_data(input_df, is_train=False, 
                                       scaler=self.scaler, 
                                       feature_columns=self.feature_columns)

        # Make batch predictions
        predictions = self.model.predict(processed_df)
        prediction_probas = self.model.predict_proba(processed_df)

        results = []
        for i in range(len(predictions)):
            loan_status = 'Rejected' if predictions[i] == 1 else 'Approved'
            results.append({
                'loan_status': loan_status,
                'probability_rejected': float(prediction_probas[i][1]),
                'probability_approved': float(prediction_probas[i][0])
            })
        return results
