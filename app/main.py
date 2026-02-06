# !pip install fastapi # install it if not installed previously (the fastapi)

from fastapi import FastAPI #creates the web API
import joblib # loads saved ML model
import pandas as pd # handles input data in tabular form
import numpy as np # reverse log transformation (expm1)

app = FastAPI(title="House Price Prediction API")
# Creates your API application. app is the object that handles requests. Title appears in API docs

model = joblib.load("ridge_house_price_model.pkl")
model_features = joblib.load("model_features.pkl")
# This is to load the model and feature list (Column Names)

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}
# Simple endpoint to verify API is alive
# Used by cloud platforms to check health

@app.post("/predict")
def predict_price(data: dict):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])
    # Ensure all required features exist, If some values are missing then 0 will be assigned automatically.
    for col in model_features:
        if col not in input_df:
            input_df[col] = 0
    # Reorder columns
    input_df = input_df[model_features]
    # Prediction (log scale)
    log_price = model.predict(input_df)[0]

    # Convert back to original scale
    price = np.expm1(log_price)

    return {"predicted_house_price": round(price, 2)}
