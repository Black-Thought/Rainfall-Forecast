import requests

AI_SERVICE_URL = "http://localhost:8000"


def test_current_weather_api():
    """
    Integration test for /weather/current endpoint.

    This test:
    - Sends a POST request to the FastAPI current weather endpoint
    - Uses a real location (Bangalore)
    - Validates:
        * HTTP response is 200
        * Response structure is correct
        * Weather data is present
    """

    url = f"{AI_SERVICE_URL}/weather/current/"

    payload = {
        "location": "Bangalore"
    }

    response = requests.post(url, json=payload)

    print("\nStatus Code:", response.status_code)
    print("Response:", response.json())

    # BASIC ASSERTIONS
    assert response.status_code == 200

    data = response.json()

    # STRUCTURE VALIDATION
    assert "location" in data
    assert "region" in data
    assert "country" in data
    assert "lat" in data
    assert "lon" in data

    assert "temperature_c" in data
    assert "feels_like_c" in data
    assert "humidity" in data
    assert "pressure_mb" in data
    assert "visibility_km" in data

    assert "wind_kph" in data
    assert "wind_degree" in data
    assert "wind_direction" in data

    assert "condition" in data
    assert "cloud" in data
    assert "uv" in data

    assert "precip_mm" in data
    assert "is_raining" in data

    assert "aqi" in data

    # VALUE VALIDATION
    assert isinstance(data["temperature_c"], (int, float))
    assert isinstance(data["humidity"], int)
    assert 0 <= data["humidity"] <= 100

    assert isinstance(data["wind_kph"], (int, float))
    assert data["wind_kph"] >= 0

    assert isinstance(data["precip_mm"], (int, float))
    assert data["precip_mm"] >= 0

    # AQI VALIDATION
    aqi = data["aqi"]
    assert isinstance(aqi, dict)

    for key in ["pm2_5", "pm10", "co", "no2", "o3", "so2"]:
        assert key in aqi

    # OPTIONAL DOMAIN CHECK
    assert data["lat"] != 0
    assert data["lon"] != 0