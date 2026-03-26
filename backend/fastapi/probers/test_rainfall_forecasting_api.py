import requests
from datetime import date

AI_SERVICE_URL = "http://localhost:8000"

def test_forecast_rainfall_api():
    """
    Integration test for /forecast endpoint.

    This test:
    - Sends a POST request to the FastAPI forecast endpoint
    - Uses a real station name (Belgaum)
    - Validates:
        * HTTP response is 200
        * Response structure is correct
        * Predictions are returned
    """

    url = f"{AI_SERVICE_URL}/forecast/rainfall"

    payload = {
        "station_name": "Belgaum",
        "start_date": date(2026, 5, 1).isoformat(),
        "num_days": 2
    }

    response = requests.post(url, json=payload)

    print("\nStatus Code:", response.status_code)
    print("Response:", response.json())

    # BASIC ASSERTIONS
    assert response.status_code == 200

    data = response.json()

    # STRUCTURE VALIDATION
    assert "station_name" in data
    assert "start_date" in data
    assert "num_days" in data
    assert "predictions" in data

    # VALUE VALIDATION
    assert data["station_name"] == "Belgaum"
    assert data["num_days"] == 2

    predictions = data["predictions"]

    assert isinstance(predictions, list)
    assert len(predictions) == 2

    for item in predictions:
        assert "date_of_record" in item
        assert "predicted_rainfall" in item

        assert isinstance(item["predicted_rainfall"], (int, float))
        assert item["predicted_rainfall"] >= 0  # domain check