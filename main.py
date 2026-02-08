from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL ASSETS ---
model = None
scaler_x = None
scaler_y = None
feature_cols = []

# This ensures the model loads correctly on your Mac
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(), 
    'mae': tf.keras.metrics.MeanAbsoluteError()
}

@app.on_event("startup")
async def startup_event():
    global model, scaler_x, scaler_y, feature_cols
    try:
        model = tf.keras.models.load_model('models/model_lstm_final.h5', custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='mse')
        scaler_x = joblib.load('models/scaler_x.pkl')
        scaler_y = joblib.load('models/scaler_y.pkl')
        with open('models/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        print("✅ Backend Assets Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading assets: {e}")

class WeatherRequest(BaseModel):
    daily_history: list 

# --- NEW: HEALTH CHECK ENDPOINT ---
# Use this to show "API Connected" on your Frontend
@app.get("/health")
async def health_check():
    return {
        "status": "connected",
        "model_ready": model is not None,
        "device": "CPU/MPS"
    }

@app.post("/predict")
async def predict_weather(request: WeatherRequest):
    try:
        # 1. Convert to DataFrame
        df = pd.DataFrame(request.daily_history)
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. MATCH NOTEBOOK FEATURE ENGINEERING
        # The AI was trained on: ['MinTemp_Lag1', 'MaxTemp_Lag1', 'Temp_Rolling_7d', 'Month']
        df['Month'] = df['date'].dt.month
        
        # We need to compute features based on the history provided
        df['MinTemp_Lag1'] = df['min_temp'].shift(1)
        df['MaxTemp_Lag1'] = df['max_temp'].shift(1)
        df['Temp_Rolling_7d'] = df['min_temp'].rolling(window=7).mean()

        # IMPORTANT: Fill NaNs from the beginning of the sequence so we don't lose data
        df = df.bfill() 
        
        # 3. SELECT THE LATEST ROW
        # This row represents "Today's features" which the model uses to see "Tomorrow"
        latest_row = df.tail(1)[feature_cols]
        
        # 4. SCALE & RESHAPE
        scaled_row = scaler_x.transform(latest_row)
        # Note: Your model was trained with (samples, 1, features)
        lstm_input = scaled_row.reshape(1, 1, len(feature_cols))
        
        # 5. PREDICT
        pred_scaled = model.predict(lstm_input)
        result_f = float(scaler_y.inverse_transform(pred_scaled)[0][0])
        
        # 6. SAFETY CLIP (Based on Sri Lanka / Training Bounds)
        # Prevents the model from giving unrealistic numbers if the input is noisy
        result_f = max(min(result_f, 105), 35) 

        return {
            "prediction_f": round(result_f, 2),
            "prediction_c": round((result_f - 32) * 5/9, 2),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}