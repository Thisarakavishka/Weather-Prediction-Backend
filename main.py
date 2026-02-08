from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import weather 
import os

app = FastAPI(title="Minima AI Engine")

# --- CORS CONFIGURATION (ALLOW ALL) ---
# This fixes the "OPTIONS 400 Bad Request" error in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allows any website (Vercel, Localhost, etc.)
    allow_credentials=False, # Must be False when using "*"
    allow_methods=["*"],     # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],     # Allows all headers
)

# Include the organized weather endpoints
app.include_router(weather.router)

# --- RENDER DEPLOYMENT BINDING ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)