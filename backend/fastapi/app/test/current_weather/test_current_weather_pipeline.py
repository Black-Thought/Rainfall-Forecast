import os
os.environ["WEATHER_API_KEY"] = "test_key"

import pytest
from unittest.mock import AsyncMock, patch

from app.src.current_weather_pipeline import current_weather_pipeline
from app.schema.weather_api import CurrentWeather
from app.schema.distance import Coordinates


# -----------------------------------
# TEST: SUCCESS
# -----------------------------------

@pytest.mark.asyncio
async def test_current_weather_pipeline_success():

    coords = Coordinates(lat=12.97, lon=77.59)

    mock_weather = CurrentWeather(
        location="Bangalore",     # ✅ FIXED
        region="Karnataka",
        country="India",
        lat=12.97,               # ✅ FIXED
        lon=77.59,               # ✅ FIXED
        temperature_c=28.0,
        feels_like_c=30.0,
        humidity=60,
        pressure_mb=1012.0,
        visibility_km=10.0,
        wind_kph=15.0,
        wind_degree=180,
        wind_direction="S",
        condition="Sunny",
        cloud=20,
        uv=6.0,
        precip_mm=0.0,
        is_raining=False,
        aqi={
            "pm2_5": 30,
            "pm10": 50,
            "co": 200,
            "no2": 20,
            "o3": 100,
            "so2": 5
        }
    )

    with patch("app.src.current_weather_pipeline.WeatherClient") as MockClient:
        mock_instance = MockClient.return_value

        mock_instance.get_full_current_weather = AsyncMock(return_value=mock_weather)
        mock_instance.close = AsyncMock()

        result = await current_weather_pipeline(coords)

        assert isinstance(result, CurrentWeather)
        assert result.location == "Bangalore"     # ✅ FIXED
        assert result.lat == 12.97                # ✅ FIXED
        assert result.temperature_c == 28.0

        mock_instance.get_full_current_weather.assert_called_once_with(coords)
        mock_instance.close.assert_called_once()


# -----------------------------------
# TEST: EXCEPTION
# -----------------------------------

@pytest.mark.asyncio
async def test_current_weather_pipeline_failure():

    coords = Coordinates(lat=0.0, lon=0.0)

    with patch("app.src.current_weather_pipeline.WeatherClient") as MockClient:
        mock_instance = MockClient.return_value

        mock_instance.get_full_current_weather = AsyncMock(
            side_effect=ValueError("Invalid location")
        )
        mock_instance.close = AsyncMock()

        with pytest.raises(ValueError):
            await current_weather_pipeline(coords)

        mock_instance.close.assert_called_once()


# -----------------------------------
# TEST: SCHEMA VALIDATION
# -----------------------------------

@pytest.mark.asyncio
async def test_current_weather_pipeline_schema_validation():

    coords = Coordinates(lat=28.61, lon=77.20)

    mock_weather = CurrentWeather(
        location="Delhi",        # ✅ FIXED
        region="Delhi",
        country="India",
        lat=28.61,              # ✅ FIXED
        lon=77.20,              # ✅ FIXED
        temperature_c=35.0,
        feels_like_c=38.0,
        humidity=50,
        pressure_mb=1008.0,
        visibility_km=8.0,
        wind_kph=10.0,
        wind_degree=90,
        wind_direction="E",
        condition="Hot",
        cloud=10,
        uv=8.0,
        precip_mm=0.0,
        is_raining=False,
        aqi={
            "pm2_5": 80,
            "pm10": 120,
            "co": 300,
            "no2": 40,
            "o3": 150,
            "so2": 10
        }
    )

    with patch("app.src.current_weather_pipeline.WeatherClient") as MockClient:
        mock_instance = MockClient.return_value

        mock_instance.get_full_current_weather = AsyncMock(return_value=mock_weather)
        mock_instance.close = AsyncMock()

        result = await current_weather_pipeline(coords)

        validated = CurrentWeather.model_validate(result)

        assert validated.location == "Delhi"   # ✅ FIXED
        assert validated.lat == 28.61          # ✅ FIXED
        assert validated.aqi.pm2_5 == 80