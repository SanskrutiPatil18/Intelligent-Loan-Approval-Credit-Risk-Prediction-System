import pandas as pd
import joblib

def preprocess_data(df: pd.DataFrame, is_train: bool = True, scaler=None, feature_columns=None):
    """
    Preprocesses the raw input data.
    Args:
        df (pd.DataFrame): The input DataFrame.
        is_train (bool): True if processing training data, False otherwise.
        scaler: Fitted StandardScaler object (required for test data).
        feature_columns: List of feature columns (required for test data).
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        StandardScaler: Fitted StandardScaler object if is_train is True.
    """

    # Ensure column names are stripped of leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Handle categorical features using one-hot encoding
    categorical_cols = ['education', 'self_employed'] # 'loan_status' is target
    # Ensure all categorical columns exist before applying get_dummies
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'Unknown' # Or handle with a more appropriate default/error

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    df_encoded.columns = df_encoded.columns.str.strip()

    if is_train:
        # For training data, we want to separate features and target
        if 'loan_status_Rejected' in df_encoded.columns:
            X = df_encoded.drop('loan_status_Rejected', axis=1)
            y = df_encoded['loan_status_Rejected']
        else:
            # If 'loan_status_Rejected' is not present, assume it's for prediction
            X = df_encoded
            y = None

        # Fit and transform the scaler on training features
        scaler = joblib.load('loan-risk-prediction/models/scaler.pkl') # Load existing scaler for consistency
        X_scaled = scaler.fit_transform(X)
        X_processed = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Save feature columns for consistency in prediction
        joblib.dump(X.columns.tolist(), 'loan-risk-prediction/models/feature_columns.pkl')

        return X_processed, y, scaler, X.columns.tolist()
    else:
        # For prediction data, use provided scaler and feature columns
        if scaler is None or feature_columns is None:
            raise ValueError("Scaler and feature_columns must be provided for prediction data.")

        # Align columns with training data features, filling missing with 0
        X_processed = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Transform using the fitted scaler
        X_scaled = scaler.transform(X_processed)
        X_processed = pd.DataFrame(X_scaled, columns=feature_columns, index=X_processed.index)

        return X_processed
