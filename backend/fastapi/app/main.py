from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes.rainfall_forecasting import router as rainfall_forecast_router
from app.routes.get_current_weather import router as current_weather_router
from app.routes.weather_forecasting import router as weather_forecast_router
from app.routes.rainfall_forecasting_zonewise import router as rainfall_forecast_zonewise_router

# Initialize FastAPI app
app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    return {"status": "API is running"}


app.include_router(rainfall_forecast_router)
app.include_router(current_weather_router)
app.include_router(weather_forecast_router)
app.include_router(rainfall_forecast_zonewise_router)