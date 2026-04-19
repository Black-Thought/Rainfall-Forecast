import pytest
from unittest.mock import AsyncMock

from app.api.weather_client import WeatherClient
from app.schema.distance import Coordinates


@pytest.fixture
def client():
    return WeatherClient(api_key="test_key")


# -----------------------------------
# TEST: _request
# -----------------------------------

@pytest.mark.asyncio
async def test_request_success(client):
    mock_response = AsyncMock()
    mock_response.json = lambda: {"key": "value"}
    mock_response.raise_for_status = lambda: None

    client.client.get = AsyncMock(return_value=mock_response)

    result = await client._request("http://test.com", {})

    assert result == {"key": "value"}


# -----------------------------------
# TEST: get_full_current_weather (OLD SCHEMA)
# -----------------------------------

@pytest.mark.asyncio
async def test_get_full_current_weather(client):

    coords = Coordinates(lat=12.97, lon=77.59)

    mock_weather_response = {
        "location": {
            "name": "Bangalore",
            "region": "Karnataka",
            "country": "India"
        },
        "current": {
            "temp_c": 28,
            "feelslike_c": 30,
            "humidity": 60,
            "pressure_mb": 1012,
            "vis_km": 10,
            "wind_kph": 15,
            "wind_degree": 180,
            "wind_dir": "S",
            "condition": {"text": "Sunny"},
            "cloud": 20,
            "uv": 6,
            "precip_mm": 0,
            "air_quality": {
                "pm2_5": 30,
                "pm10": 50,
                "co": 200,
                "no2": 20,
                "o3": 100,
                "so2": 5
            }
        }
    }

    client._weather_request = AsyncMock(return_value=mock_weather_response)

    result = await client.get_full_current_weather(coords)

    # MATCHES YOUR CURRENT MODEL
    assert result.location == "Bangalore"
    assert result.lat == 12.97
    assert result.lon == 77.59
    assert result.temperature_c == 28
    assert result.aqi.pm2_5 == 30


# -----------------------------------
# TEST: get_full_forecast (NEW SCHEMA)
# -----------------------------------

@pytest.mark.asyncio
async def test_get_full_forecast(client):

    coords = Coordinates(lat=12.97, lon=77.59)

    mock_forecast_response = {
        "location": {"name": "Bangalore"},
        "forecast": {
            "forecastday": [
                {
                    "date": "2026-03-30",
                    "day": {
                        "avgtemp_c": 27,
                        "maxtemp_c": 32,
                        "mintemp_c": 22,
                        "totalprecip_mm": 5,
                        "daily_chance_of_rain": "80",
                        "maxwind_kph": 20,
                        "condition": {"text": "Rain"}
                    }
                }
            ]
        }
    }

    client._weather_request = AsyncMock(return_value=mock_forecast_response)

    result = await client.get_full_forecast(coords, days=1)

    # FIXED ASSERTIONS
    assert result.resolved_name == "Bangalore"
    assert result.location.lat == 12.97
    assert result.location.lon == 77.59

    assert len(result.forecast) == 1

    day = result.forecast[0]
    assert day.avg_temp_c == 27
    assert day.will_rain is True
    assert day.condition == "Rain"