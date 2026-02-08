import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.feature_cols = None
        # From your notebook results: Internal MAE was ~8.6°F
        self.accuracy_mae = 8.62 

    def load_assets(self):
        base_path = os.path.dirname(os.path.dirname(__file__))
        custom_objects = {'mse': 'mse', 'mae': 'mae'}
        
        self.model = tf.keras.models.load_model(
            os.path.join(base_path, 'models/model_lstm_final.h5'), 
            custom_objects=custom_objects, compile=False
        )
        self.scaler_x = joblib.load(os.path.join(base_path, 'models/scaler_x.pkl'))
        self.scaler_y = joblib.load(os.path.join(base_path, 'models/scaler_y.pkl'))
        with open(os.path.join(base_path, 'models/feature_columns.json'), 'r') as f:
            self.feature_cols = json.load(f)

    def predict(self, history):
        df = pd.DataFrame(history)
        df['date'] = pd.to_datetime(df['date'])
        df['Month'] = df['date'].dt.month
        df['MinTemp_Lag1'] = df['min_temp'].shift(1)
        df['MaxTemp_Lag1'] = df['max_temp'].shift(1)
        df['Temp_Rolling_7d'] = df['min_temp'].rolling(window=7).mean()
        df = df.bfill()
        
        latest_features = df.tail(1)[self.feature_cols]
        scaled = self.scaler_x.transform(latest_features)
        lstm_input = scaled.reshape(1, 1, len(self.feature_cols))
        
        pred_scaled = self.model.predict(lstm_input)
        res_f = float(self.scaler_y.inverse_transform(pred_scaled)[0][0])
        return max(min(res_f, 105), 35)

predictor = WeatherPredictor()