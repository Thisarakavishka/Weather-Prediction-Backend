# 🌡️ Minima AI | Backend Engine

The computational heart of **Minima AI**, providing real-time LSTM (Long Short-Term Memory) inference for high-precision weather forecasting.

* **Live API:** [https://weather-prediction-backend-x4pv.onrender.com](https://weather-prediction-backend-x4pv.onrender.com)
* **Production Frontend:** [https://minimaai.vercel.app/](https://minimaai.vercel.app/)
* **Frontend:** [https://github.com/Thisarakavishka/Weather-Prediction-Frontend](https://github.com/Thisarakavishka/Weather-Prediction-Frontend)
---

## 🚀 Overview
The backend is built with **FastAPI** using a modular service-oriented architecture. It serves a serialized **TensorFlow LSTM** model that has been optimized for time-series forecasting.

### 🏗️ Architecture
- **Routes:** Handles HTTP requests and data validation using Pydantic.
- **Services:** Manages the AI lifecycle, including sequence preprocessing, feature engineering, and inference.
- **Models:** Stores the `.h5` model files and `.pkl` scalers.

### 🛣️ API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns system status, model readiness, and accuracy metrics. |
| `POST` | `/predict` | Processes 14-day history to return a 24-hour forecast + Confidence scores. |

---

## 🛠️ Local Setup
1. **Clone the repository:**
   ```bash
   git clone <your-backend-repo-url>
   cd Weather-Prediction-Backend

2. **Setup Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the Server:**
   ```bash
   uvicorn main:app --reload

## 🧠 Model Specifications
- **Type:** Bidirectional LSTM (Deep Learning)

- **Input Shape:** (1, 1, 4) representing [Samples, Timesteps, Features]

- **Features:** MinTemp_Lag1, MaxTemp_Lag1, Temp_Rolling_7d, Month

- **Mean Absolute Error (MAE):** 8.62°F