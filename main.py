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

# --- 1. CORS CONFIGURATION ---
# Read from environment variable for production
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173"], # Allows both local and production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. GLOBAL ASSETS & LOADING ---
model = None
scaler_x = None
scaler_y = None
feature_cols = []

custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(), 
    'mae': tf.keras.metrics.MeanAbsoluteError()
}

@app.on_event("startup")
async def startup_event():
    global model, scaler_x, scaler_y, feature_cols
    try:
        # Construct absolute path to avoid issues on cloud servers
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'models', 'model_lstm_final.h5')
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='mse')
        
        scaler_x = joblib.load(os.path.join(base_path, 'models', 'scaler_x.pkl'))
        scaler_y = joblib.load(os.path.join(base_path, 'models', 'scaler_y.pkl'))
        
        with open(os.path.join(base_path, 'models', 'feature_columns.json'), 'r') as f:
            feature_cols = json.load(f)
        print("✅ Backend Assets Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading assets: {e}")

# --- 3. MODELS & ENDPOINTS ---
class WeatherRequest(BaseModel):
    daily_history: list 

@app.get("/health")
async def health_check():
    return {
        "status": "connected",
        "model_ready": model is not None,
        "api_connected_label": "API Connected ✅" # This helps your frontend
    }

@app.post("/predict")
async def predict_weather(request: WeatherRequest):
    try:
        df = pd.DataFrame(request.daily_history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Feature Engineering
        df['Month'] = df['date'].dt.month
        df['MinTemp_Lag1'] = df['min_temp'].shift(1)
        df['MaxTemp_Lag1'] = df['max_temp'].shift(1)
        df['Temp_Rolling_7d'] = df['min_temp'].rolling(window=7).mean()
        df = df.bfill() 
        
        latest_row = df.tail(1)[feature_cols]
        scaled_row = scaler_x.transform(latest_row)
        lstm_input = scaled_row.reshape(1, 1, len(feature_cols))
        
        pred_scaled = model.predict(lstm_input)
        result_f = float(scaler_y.inverse_transform(pred_scaled)[0][0])
        
        # Safety Clip for Sri Lanka context
        result_f = max(min(result_f, 105), 35) 

        return {
            "prediction_f": round(result_f, 2),
            "prediction_c": round((result_f - 32) * 5/9, 2),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

# --- 4. SERVER START (MUST BE AT THE BOTTOM) ---
if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 for Render to bind to all network interfaces
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)