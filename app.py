from fastapi import FastAPI
from pydantic import BaseModel
from backend.predictor import LoanPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval status based on applicant data.",
    version="1.0.0",
)

# Initialize the predictor globally to load the model only once
predictor = LoanPredictor()

# Define Request Body Model
class LoanApplication(BaseModel):
    loan_id: int
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int
    loan_status: str # This will be ignored for prediction, but kept for schema consistency

# Define Batch Request Body Model
class LoanApplicationBatch(BaseModel):
    applications: list[LoanApplication]

# --- API Endpoints ---

@app.get("/", summary="Health Check")
async def read_root():
    return {"message": "Loan Approval Prediction API is running!"}

@app.post("/predict/", response_model=dict, summary="Predict Loan Approval for a single application")
async def predict_loan_status(application: LoanApplication):
    """
    Predicts the loan approval status for a single applicant.

    - **loan_id**: Unique identifier for the loan (integer)
    - **no_of_dependents**: Number of dependents (integer)
    - **education**: Education level (string: 'Graduate', 'Not Graduate')
    - **self_employed**: Self-employed status (string: 'Yes', 'No')
    - **income_annum**: Annual income in INR (integer)
    - **loan_amount**: Requested loan amount in INR (integer)
    - **loan_term**: Loan term in years (integer)
    - **cibil_score**: CIBIL score (integer)
    - **residential_assets_value**: Value of residential assets in INR (integer)
    - **commercial_assets_value**: Value of commercial assets in INR (integer)
    - **luxury_assets_value**: Value of luxury assets in INR (integer)
    - **bank_asset_value**: Value of bank assets in INR (integer)
    - **loan_status**: Placeholder, expected 'Unknown' (string)

    Returns the predicted loan status and probabilities.
    """
    # Convert Pydantic model to dict for the predictor
    raw_data = application.dict()
    prediction_result = predictor.predict(raw_data)
    return prediction_result

@app.post("/predict_batch/", response_model=list[dict], summary="Predict Loan Approval for multiple applications")
async def predict_loan_status_batch(batch_applications: LoanApplicationBatch):
    """
    Predicts the loan approval status for a list of applicants.

    Accepts a list of `LoanApplication` objects and returns a list of prediction results.
    """
    raw_data_list = [app.dict() for app in batch_applications.applications]
    prediction_results = predictor.predict_batch(raw_data_list)
    return prediction_results
