import httpx
from typing import Dict, Any
from app.core.urls import WEATHER_API_BASE_URL, GEOCODE_URL
from app.schema.distance import Coordinates
from app.schema.weather_api import LatLon, CurrentWeather, WeatherForecastDay, WeatherForecastResponse


class WeatherClient:
    """
    Client for fetching geolocation, current weather, AQI, and forecast data.

    This client:
    - Converts location names to latitude/longitude using a geocoding API
    - Fetches weather data from WeatherAPI
    - Validates responses using Pydantic models
    """

    def __init__(self, api_key: str, timeout: int = 10):
        """
        Initialize the WeatherClient.

        Args:
            api_key (str): API key for the weather service.
            timeout (int): Request timeout in seconds.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        """
        Close the underlying HTTP client.
        """
        await self.client.aclose()

    async def _request(self, url: str, params: Dict[str, Any]) -> Dict:
        """
        Perform an HTTP GET request.

        Args:
            url (str): Target URL.
            params (Dict[str, Any]): Query parameters.

        Returns:
            Dict: JSON response.

        Raises:
            httpx.HTTPStatusError: If response status is not successful.
        """
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def get_lat_lon(self, location: str) -> LatLon:
        """
        Convert a location name into latitude and longitude.

        Args:
            location (str): City or place name.

        Returns:
            LatLon: Validated latitude and longitude.

        Raises:
            ValueError: If location is not found.
        """
        data = await self._request(
            GEOCODE_URL,
            {"name": location, "count": 1}
        )

        results = data.get("results")
        if not results:
            raise ValueError(f"Location not found: {location}")

        result = results[0]

        return LatLon(
            latitude=result["latitude"],
            longitude=result["longitude"]
        )

    async def _weather_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """
        Perform a weather API request.

        Args:
            endpoint (str): API endpoint (e.g., 'current.json').
            params (Dict[str, Any]): Query parameters.

        Returns:
            Dict: JSON response from weather API.
        """
        params["key"] = self.api_key
        return await self._request(f"{WEATHER_API_BASE_URL}/{endpoint}", params)

    async def get_full_current_weather(
        self,
        coordinates: Coordinates
    ) -> CurrentWeather:
        """
        Fetch current weather and AQI using latitude & longitude.

        Args:
            coordinates (Coordinates): Geographic coordinates

        Returns:
            CurrentWeather
        """

        data = await self._weather_request(
            "current.json",
            {
                "q": f"{coordinates.lat},{coordinates.lon}",
                "aqi": "yes"
            }
        )

        current = data["current"]
        location_data = data["location"]

        weather_dict = {
            "location": location_data["name"],
            "region": location_data["region"],
            "country": location_data["country"],
            "lat": coordinates.lat,
            "lon": coordinates.lon,

            "temperature_c": current["temp_c"],
            "feels_like_c": current["feelslike_c"],

            "humidity": current["humidity"],
            "pressure_mb": current["pressure_mb"],
            "visibility_km": current["vis_km"],

            "wind_kph": current["wind_kph"],
            "wind_degree": current["wind_degree"],
            "wind_direction": current["wind_dir"],

            "condition": current["condition"]["text"],
            "cloud": current["cloud"],
            "uv": current["uv"],

            "precip_mm": current["precip_mm"],
            "is_raining": current["precip_mm"] > 0,

            "aqi": {
                "pm2_5": current["air_quality"].get("pm2_5"),
                "pm10": current["air_quality"].get("pm10"),
                "co": current["air_quality"].get("co"),
                "no2": current["air_quality"].get("no2"),
                "o3": current["air_quality"].get("o3"),
                "so2": current["air_quality"].get("so2"),
            }
        }

        return CurrentWeather(**weather_dict)


    async def get_full_forecast(
        self,
        coordinates: Coordinates,
        days: int = 3
    ) -> WeatherForecastResponse:
        """
        Fetch weather forecast using latitude & longitude.

        Args:
            coordinates (Coordinates): Geographic coordinates
            days (int): Number of forecast days (1–10)

        Returns:
            WeatherForecastResponse
        """

        data = await self._weather_request(
            "forecast.json",
            {
                "q": f"{coordinates.lat},{coordinates.lon}",
                "days": days,
                "aqi": "yes",
                "alerts": "yes"
            }
        )

        forecast_days = data["forecast"]["forecastday"]

        forecast_list = []

        for day_data in forecast_days:
            day = day_data["day"]

            forecast_list.append(
                WeatherForecastDay(
                    date=day_data["date"],
                    avg_temp_c=day["avgtemp_c"],
                    max_temp_c=day["maxtemp_c"],
                    min_temp_c=day["mintemp_c"],
                    total_precip_mm=day["totalprecip_mm"],
                    rain_probability=int(day["daily_chance_of_rain"]),
                    will_rain=int(day["daily_chance_of_rain"]) > 50,
                    max_wind_kph=day["maxwind_kph"],
                    condition=day["condition"]["text"]
                )
            )

        return WeatherForecastResponse(
            location=coordinates,
            resolved_name=data["location"]["name"],       
            forecast=forecast_list
        )