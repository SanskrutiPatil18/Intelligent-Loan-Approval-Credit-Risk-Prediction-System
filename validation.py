import pandas as pd

def validate_loan_data(df: pd.DataFrame) -> (bool, str):
    """
    Validates the input DataFrame for loan application data.
    Checks for expected columns and data types.
    """
    expected_columns = [
        'loan_id', 'no_of_dependents', 'education', 'self_employed', 
        'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
        'residential_assets_value', 'commercial_assets_value', 
        'luxury_assets_value', 'bank_asset_value', 'loan_status'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"

    # Check data types (simplified for this example, adjust as needed)
    # Example: all numeric columns should be int or float
    numeric_cols = [
        'loan_id', 'no_of_dependents', 'income_annum', 'loan_amount', 
        'loan_term', 'cibil_score', 'residential_assets_value', 
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' has incorrect data type. Expected numeric."

    # Check categorical values
    if not all(df['education'].isin(['Graduate', 'Not Graduate'])):
        return False, "Invalid values in 'education' column. Expected 'Graduate' or 'Not Graduate'."
    if not all(df['self_employed'].isin(['Yes', 'No'])):
        return False, "Invalid values in 'self_employed' column. Expected 'Yes' or 'No'."
    # loan_status can contain 'Approved', 'Rejected', or 'Unknown' for prediction input
    if not all(df['loan_status'].isin(['Approved', 'Rejected', 'Unknown'])):
        return False, "Invalid values in 'loan_status' column. Expected 'Approved', 'Rejected', or 'Unknown'."

    # Add more validation rules as necessary (e.g., range checks, cibil score min/max)

    return True, "Validation successful."
