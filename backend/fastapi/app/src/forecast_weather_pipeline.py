from app.api.weather_client import WeatherClient
from app.core.load_env import get_settings
from app.schema.weather_api import WeatherForecastResponse
from app.schema.distance import Coordinates

settings = get_settings()


async def forecast_weather_pipeline(
    coordinates: Coordinates,
    days: int = 3
) -> WeatherForecastResponse:
    """
    Pipeline to fetch weather forecast using geographic coordinates.

    This function:
    - Initializes the WeatherClient
    - Calls the forecast API using lat/lon
    - Returns validated WeatherForecastResponse schema

    Args:
        coordinates (Coordinates): Latitude and longitude
        days (int): Number of days for forecast (1–10)

    Returns:
        WeatherForecastResponse
    """

    client = WeatherClient(api_key=settings.WEATHER_API_KEY)

    try:
        forecast: WeatherForecastResponse = await client.get_full_forecast(
            coordinates=coordinates,
            days=days
        )
        return forecast
    finally:
        await client.close()