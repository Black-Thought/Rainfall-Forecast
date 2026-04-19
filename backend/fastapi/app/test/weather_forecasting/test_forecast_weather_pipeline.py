import os
os.environ["WEATHER_API_KEY"] = "test_key"

import pytest
from unittest.mock import AsyncMock, patch

from app.src.forecast_weather_pipeline import forecast_weather_pipeline
from app.schema.weather_api import WeatherForecastResponse, WeatherForecastDay
from app.schema.distance import Coordinates


# -----------------------------------
# TEST: SUCCESS
# -----------------------------------

@pytest.mark.asyncio
async def test_forecast_weather_pipeline_success():

    coords = Coordinates(lat=12.97, lon=77.59)

    mock_forecast = WeatherForecastResponse(
        location=coords,
        resolved_name="Bangalore",
        forecast=[
            WeatherForecastDay(
                date="2026-03-30",
                avg_temp_c=27.0,
                max_temp_c=32.0,
                min_temp_c=22.0,
                total_precip_mm=5.0,
                rain_probability=80,
                will_rain=True,
                max_wind_kph=20.0,
                condition="Rain"
            )
        ]
    )

    with patch("app.src.forecast_weather_pipeline.WeatherClient") as MockClient:
        mock_instance = MockClient.return_value

        mock_instance.get_full_forecast = AsyncMock(return_value=mock_forecast)
        mock_instance.close = AsyncMock()

        result = await forecast_weather_pipeline(coords, days=1)

        # Assertions
        assert isinstance(result, WeatherForecastResponse)
        assert result.resolved_name == "Bangalore"
        assert result.location.lat == 12.97
        assert len(result.forecast) == 1
        assert result.forecast[0].will_rain is True

        mock_instance.get_full_forecast.assert_called_once_with(
            coordinates=coords, days=1
        )
        mock_instance.close.assert_called_once()


# -----------------------------------
# TEST: EXCEPTION
# -----------------------------------

@pytest.mark.asyncio
async def test_forecast_weather_pipeline_failure():

    coords = Coordinates(lat=0.0, lon=0.0)

    with patch("app.src.forecast_weather_pipeline.WeatherClient") as MockClient:
        mock_instance = MockClient.return_value

        mock_instance.get_full_forecast = AsyncMock(
            side_effect=ValueError("Invalid location")
        )
        mock_instance.close = AsyncMock()

        with pytest.raises(ValueError):
            await forecast_weather_pipeline(coords, days=3)

        mock_instance.close.assert_called_once()


# -----------------------------------
# TEST: SCHEMA VALIDATION
# -----------------------------------

@pytest.mark.asyncio
async def test_forecast_weather_pipeline_schema_validation():

    coords = Coordinates(lat=28.61, lon=77.20)

    mock_forecast = WeatherForecastResponse(
        location=coords,
        resolved_name="Delhi",
        forecast=[
            WeatherForecastDay(
                date="2026-04-01",
                avg_temp_c=35.0,
                max_temp_c=40.0,
                min_temp_c=30.0,
                total_precip_mm=0.0,
                rain_probability=10,
                will_rain=False,
                max_wind_kph=12.0,
                condition="Hot"
            )
        ]
    )

    with patch("app.src.forecast_weather_pipeline.WeatherClient") as MockClient:
        mock_instance = MockClient.return_value

        mock_instance.get_full_forecast = AsyncMock(return_value=mock_forecast)
        mock_instance.close = AsyncMock()

        result = await forecast_weather_pipeline(coords, days=1)

        validated = WeatherForecastResponse.model_validate(result)

        assert validated.resolved_name == "Delhi"
        assert validated.location.lat == 28.61
        assert validated.forecast[0].avg_temp_c == 35.0
        assert validated.forecast[0].will_rain is False