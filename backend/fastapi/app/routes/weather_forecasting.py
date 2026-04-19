from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.schema.weather_api import WeatherForecastResponse
from app.schema.distance import Coordinates
from app.src.forecast_weather_pipeline import forecast_weather_pipeline


router = APIRouter(prefix="/forecast/weather", tags=["Forecast"])


# -----------------------------------
# REQUEST MODEL (UPDATED)
# -----------------------------------

class WeatherForecastRequest(BaseModel):
    coordinates: Coordinates = Field(
        ...,
        description="Latitude and longitude for weather forecast",
        examples=[{"lat": 28.6139, "lon": 77.2090}]
    )

    num_days: int = Field(
        3,
        gt=0,
        le=10,
        description="Number of forecast days (1–10)"
    )


# -----------------------------------
# ENDPOINT
# -----------------------------------

@router.post("/", response_model=WeatherForecastResponse)
async def forecast_endpoint(
    request: WeatherForecastRequest
) -> WeatherForecastResponse:
    """
    Forecast weather using geographic coordinates.
    """
    try:
        result = await forecast_weather_pipeline(
            coordinates=request.coordinates,   # FIXED
            days=request.num_days
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))