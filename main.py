import os
import pickle
from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import r2_score
from starlette.responses import JSONResponse
import pandas as pd
from model import load_model, train_model, predict
from utils.Inputs import inputs
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
class InputData(BaseModel):
    LotArea: int
    Neighborhood: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    BsmtQual: str
    BsmtExposure: str
    BsmtFinSF1: int
    TotalBsmtSF: int
    FirstFlrSF: int
    SecondFlrSF: int
    GrLivArea: int
    FullBath: int
    KitchenQual: str
    Fireplaces: int
    GarageCars: int
    GarageArea: int
    OpenPorchSF: int


@app.post(path='/predict')
async def predict_house_price(LotArea: int = Form(...),
                              Neighborhood: str = Form(...),
                              OverallQual: int = Form(...),
                              OverallCond: int = Form(...),
                              YearBuilt: int = Form(...),
                              YearRemodAdd: int = Form(...),
                              BsmtQual: str = Form(...),
                              BsmtExposure: str = Form(...),
                              BsmtFinSF1: int = Form(...),
                              TotalBsmtSF: int = Form(...),
                              FirstFlrSF: int = Form(...),
                              SecondFlrSF: int = Form(...),
                              GrLivArea: int = Form(...),
                              FullBath: int = Form(...),
                              KitchenQual: str = Form(...),
                              Fireplaces: int = Form(...),
                              GarageCars: int = Form(...),
                              GarageArea: int = Form(...),
                              OpenPorchSF: int = Form(...),):
    data = {
        "LotArea": LotArea,
        "Neighborhood": Neighborhood,
        "OverallQual": OverallQual,
        "OverallCond": OverallCond,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "BsmtQual": BsmtQual,
        "BsmtExposure": BsmtExposure,
        "BsmtFinSF1": BsmtFinSF1,
        "TotalBsmtSF": TotalBsmtSF,
        "FirstFlrSF": FirstFlrSF,
        "SecondFlrSF": SecondFlrSF,
        "GrLivArea": GrLivArea,
        "FullBath": FullBath,
        "KitchenQual": KitchenQual,
        "Fireplaces": Fireplaces,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "OpenPorchSF": OpenPorchSF}

    df = pd.DataFrame([data])

    # Load the list of columns 'remaining_columns.pkl' file
    try:
        with open('remaining_columns.pkl', 'rb') as f:
            remaining_columns = pickle.load(f)
    except FileNotFoundError:
        logging.error("remaining_columns.pkl file not found.")
        raise HTTPException(status_code=500, detail="Configuration file missing.")
    
    # Check if any columns are missing from the users input and add them
    missing_columns = set(remaining_columns) - set(df.columns)
    if missing_columns:
        for col in missing_columns:
            # Add missing columns with default values
            if col in df.select_dtypes(include=['number']).columns:
                df[col] = 0  # For numeric columns, use 0 as default
            else:
                df[col] = 'Unknown'  # For categorical columns, use 'Unknown' as default
        logging.info(f"Added missing columns: {missing_columns}")

    result_data = df.to_dict(orient="records")[0]

    return {JSONResponse(content=result_data,status_code=200)}
    if not data:
        logging.error('No input data provided.')
        raise HTTPException(status_code=400, detail='No input data provided.')

    try:
        logging.info(data)
        input_data_scaled = preprocess_input(data, scaler)
        prediction = predict(model, input_data_scaled)
        return JSONResponse(content={"prediction": prediction}, status_code=200)  # return json with prediction
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
    return templates.TemplateResponse("home.html", {"request": request, "inputs": inputs})


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    logging.info(f"Received request at '/about' from {request.client.host}")
    return templates.TemplateResponse("about.html", {"request": request})
