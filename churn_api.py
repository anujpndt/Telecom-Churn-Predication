from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, conint, confloat
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = load('xgboost.joblib')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model.")

# Load the dataset to recreate encoders
try:
    telecom_cust = pd.read_csv('Telco_Customer_Churn.csv')

    # Initialize LabelEncoders
    label_encoder_is = LabelEncoder()
    label_encoder_c = LabelEncoder()

    # Fit encoders with original dataset
    label_encoder_is.fit(telecom_cust['InternetService'])
    label_encoder_c.fit(telecom_cust['Contract'])

    logger.info("Label encoders trained successfully.")
except Exception as e:
    logger.error(f"Error loading dataset or training encoders: {e}")
    raise RuntimeError("Failed to prepare label encoders.")


# Pydantic Model for Input Validation
class ChurnInput(BaseModel):
    tenure: conint(ge=0, le=100)  # Integer between 0 and 100
    internet_service: str  # Will be validated against allowed categories
    contract: str  # Will be validated against allowed categories
    monthly_charges: confloat(ge=0, le=200)  # Float between 0 and 200
    total_charges: confloat(ge=0, le=10000)  # Float between 0 and 10000


def get_label_encoded_values(internet_service: str, contract: str):
    """
    Convert categorical values to numerical using pre-fitted LabelEncoders.
    """
    try:
        encoded_is = label_encoder_is.transform([internet_service])[0]
        encoded_contract = label_encoder_c.transform([contract])[0]
        return encoded_is, encoded_contract
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid categorical value provided.")


@app.post("/predict-churn/", summary="Predict Customer Churn")
async def predict_churn(input_data: ChurnInput):
    """
    Predicts whether a customer is likely to churn or not.

    - **Returns**: `True` (Churn) or `False` (No Churn)
    """
    try:
        # Encode categorical values
        internet_service, contract = get_label_encoded_values(input_data.internet_service, input_data.contract)

        # Prepare input for model
        input_features = [[
            input_data.tenure,
            internet_service,
            contract,
            input_data.monthly_charges,
            input_data.total_charges
        ]]

        # Make prediction
        prediction = model.predict(input_features)

        # Convert output to boolean (True for churn, False for no churn)
        result = bool(prediction[0])

        # Log the prediction
        logger.info(f"Prediction: {result} | Input: {input_data.dict()}")

        return {"churn": result}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Root endpoint
@app.get("/", summary="Health Check")
async def root():
    """
    Basic health check endpoint to verify that the API is running.
    """
    return {"message": "Customer Churn Prediction API is running!"}


### SAMPLE REQUEST

# {
#   "tenure": 3,
#   "internet_service": "Fiber optic",
#   "contract": "Month-to-month",
#   "monthly_charges": 85.0,
#   "total_charges": 255.0
# }


