import os
import pickle
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import r2_score

from model import load_model, train_model, predict
from utils.data_preprocessing import preprocess_input, load_data, preprocess_data
import logging

app = FastAPI()

MODEL_FILENAME = 'house_price_model.pkl'
SCALER_FILENAME = 'scaler.pkl'

# Configure basic logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

X_test, y_test = None, None

# Function to check and initialize model and scaler on first run
def initialize_model():
    global X_test, y_test
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(SCALER_FILENAME):
        logging.info('Model or scaler not found. Training model for the first time...')
        # Paths to dataset and target column
        initial_data_path = 'data/house_price_data.csv'
        target_column = 'SalePrice'

        # Load and preprocess data
        data = load_data(initial_data_path)
        X_train, y_train, X_test, y_test, scaler = preprocess_data(data, target_column)

        # Create and train the model then save it
        train_model(X_train, y_train, MODEL_FILENAME)

        # Save the scaler for future predictions
        with open(SCALER_FILENAME, 'wb') as file:
            pickle.dump(scaler, file)
        logging.info('Initial model training completed.')
    else:
        logging.info('Model and scaler already exist. Skipping initializaiton')


initialize_model()
model = load_model(MODEL_FILENAME)

with open(SCALER_FILENAME, 'rb') as file:
    scaler = pickle.load(file)

# Set up templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Class that will be used for receiving data from front-end user
class PredicitonRequest(BaseModel):
    features: list

# Prediction endpoint
@app.post(path='/predict')
def predict_house_price(request: PredicitonRequest):
    input_data = request.features
    if not input_data:
        logging.error('No input data provided.')
        raise HTTPException(status_code=400, detail='No input data provided.')

    try:
        input_data_scaled = preprocess_input(input_data, scaler)
        prediction = predict(model, input_data_scaled)
        return {"predicted_price": prediction}  # return json with prediction
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

@app.get("/accuracy")
def get_model_accuracy():
    if X_test is None or y_test is None:
        raise HTTPException(status_code=400, detail="Test data not available. Model needs to be retrained.")

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate R² score (can be changed to other metrics if needed)
    accuracy = r2_score(y_test, y_pred)
    return {"accuracy": accuracy}

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    logging.info(f"Received request at '/' from {request.client.host}")
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    logging.info(f"Received request at '/about' from {request.client.host}")
    return templates.TemplateResponse("about.html", {"request": request})
