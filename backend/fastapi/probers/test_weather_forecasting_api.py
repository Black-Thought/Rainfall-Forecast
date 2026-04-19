import requests

AI_SERVICE_URL = "http://localhost:8000"


def test_forecast_weather_api():
    """
    Integration test for /forecast/weather endpoint.

    This test:
    - Sends a POST request to the FastAPI forecast weather endpoint
    - Uses a real location (Bangalore)
    - Validates:
        * HTTP response is 200
        * Response structure is correct
        * Forecast data is returned
    """

    url = f"{AI_SERVICE_URL}/forecast/weather/"

    payload = {
        "location": "Bangalore",
        "num_days": 3
    }

    response = requests.post(url, json=payload)

    print("\nStatus Code:", response.status_code)
    print("Response:", response.json())

    # BASIC ASSERTIONS
    assert response.status_code == 200

    data = response.json()

    # STRUCTURE VALIDATION
    assert "location" in data
    assert "lat" in data
    assert "lon" in data
    assert "forecast" in data

    # VALUE VALIDATION
    assert data["location"] == "Bangalore"

    forecast = data["forecast"]

    assert isinstance(forecast, list)
    assert len(forecast) == 3

    # FORECAST ITEM VALIDATION
    for item in forecast:
        assert "date" in item

        assert "avg_temp_c" in item
        assert "max_temp_c" in item
        assert "min_temp_c" in item

        assert "total_precip_mm" in item
        assert "rain_probability" in item
        assert "will_rain" in item

        assert "max_wind_kph" in item
        assert "condition" in item

        # TYPE CHECKS
        assert isinstance(item["avg_temp_c"], (int, float))
        assert isinstance(item["max_temp_c"], (int, float))
        assert isinstance(item["min_temp_c"], (int, float))

        assert isinstance(item["total_precip_mm"], (int, float))
        assert item["total_precip_mm"] >= 0

        assert isinstance(item["rain_probability"], int)
        assert 0 <= item["rain_probability"] <= 100

        assert isinstance(item["will_rain"], bool)

        assert isinstance(item["max_wind_kph"], (int, float))
        assert item["max_wind_kph"] >= 0