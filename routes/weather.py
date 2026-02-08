from fastapi import APIRouter
from pydantic import BaseModel
from services.predictor import predictor

router = APIRouter()

class WeatherRequest(BaseModel):
    daily_history: list

@router.on_event("startup")
async def startup():
    predictor.load_assets()

@router.get("/health")
async def health():
    return {"status": "connected", "accuracy_info": f"MAE: {predictor.accuracy_mae}°F"}

@router.post("/predict")
async def predict(request: WeatherRequest):
    try:
        val_f = predictor.predict(request.daily_history)
        
        # Calculate Confidence based on your MAE of 8.62
        confidence = "High" if predictor.accuracy_mae < 9 else "Moderate"
        
        return {
            "prediction_f": round(val_f, 2),
            "prediction_c": round((val_f - 32) * 5/9, 2),
            "metadata": {
                "model": "LSTM-Final-V6.2",
                "mae_score": 8.62,
                "confidence": confidence,
                "location_context": f"{len(request.daily_history)} days history"
            }
        }
    except Exception as e:
        return {"error": str(e)}