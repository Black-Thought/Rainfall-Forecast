from fastapi import APIRouter, HTTPException
from app.schema.weather_api import CurrentWeather, CurrentWeatherRequest
from app.src.current_weather_pipeline import current_weather_pipeline


router = APIRouter(prefix="/weather/current", tags=["Weather"])


@router.post("/", response_model=CurrentWeather)
async def get_current_weather_endpoint(
    request: CurrentWeatherRequest
) -> CurrentWeather:
    """
    Fetch current weather (including AQI) for a given location.
    """
    try:
        result = await current_weather_pipeline(request.coordinates)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))