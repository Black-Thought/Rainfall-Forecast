from app.api.weather_client import WeatherClient
from app.core.load_env import get_settings
from app.schema.weather_api import CurrentWeather
from app.schema.distance import Coordinates

settings = get_settings()


async def current_weather_pipeline(
    coordinates: Coordinates
) -> CurrentWeather:
    """
    Pipeline to fetch current weather using geographic coordinates.

    This function:
    - Initializes the WeatherClient
    - Calls the weather API using lat/lon
    - Returns validated CurrentWeather schema

    Args:
        coordinates (Coordinates): Latitude and longitude

    Returns:
        CurrentWeather
    """

    client = WeatherClient(api_key=settings.WEATHER_API_KEY)

    try:
        weather: CurrentWeather = await client.get_full_current_weather(
            coordinates
        )
        return weather
    finally:
        await client.close()