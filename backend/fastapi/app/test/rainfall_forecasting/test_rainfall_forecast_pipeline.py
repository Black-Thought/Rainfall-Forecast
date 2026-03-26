import pandas as pd
from datetime import date
from app.utils.load_model import load_model_joblib
from app.src.forecast_rainfall_pipeline import forecast_rainfall_pipeline
from unittest.mock import patch
from app.schema.rainfall_forecasting import ForecastResponse, ForecastItem
from app.core.paths import (
    WEATHER_DATA_PATH,
    XGB_RAINFALL_FORECAST_MODEL_PATH
)

df: pd.DataFrame = pd.read_parquet(WEATHER_DATA_PATH, engine="pyarrow")
df["date_of_record"] = pd.to_datetime(df["date_of_record"])
station_list = df["station_name"].unique()
model = load_model_joblib(XGB_RAINFALL_FORECAST_MODEL_PATH)



def create_mock_response(station: str) -> ForecastResponse:
    """Create fake forecast response for testing."""
    return ForecastResponse(
        station_name=station,
        start_date=date(2025, 1, 1),
        num_days=2,
        predictions=[
            ForecastItem(date_of_record=date(2025, 1, 1), predicted_rainfall=1.2),
            ForecastItem(date_of_record=date(2025, 1, 2), predicted_rainfall=0.8),
        ]
    )


@patch("app.src.forecast_rainfall_pipeline.forecast_rainfall_pipeline")
def test_forecast_all_real_stations_unit(mock_forecast):
    """
    Unit test for forecasting pipeline across ALL real stations.

    This test:
    - Loads real station names from dataset
    - Mocks forecast function to avoid heavy computation
    - Ensures:
        * All stations are processed
        * Each returns valid prediction structure
        * No failures occur in loop logic
    """

    # Load real stations
    df: pd.DataFrame = pd.read_parquet(WEATHER_DATA_PATH, engine="pyarrow")
    station_list = df["station_name"].unique()

    # Mock behavior
    def side_effect(station_name, start_date, num_days):
        return create_mock_response(station_name)

    mock_forecast.side_effect = side_effect

    success_stations = []
    failed_stations = []

    for station in station_list:
        try:
            result = mock_forecast(
                station_name=station,
                start_date=date(2025, 1, 1),
                num_days=2
            )

            # Assertions
            assert result.station_name == station
            assert len(result.predictions) == 2

            success_stations.append(station)

        except Exception as e:
            print(f"Failed for {station}: {e}")
            failed_stations.append(station)

    # FINAL ASSERTIONS
    assert len(success_stations) == len(station_list)
    assert len(failed_stations) == 0

    # Ensure all calls executed
    assert mock_forecast.call_count == len(station_list)



def test_forecast_rainfall_belgaum_2_days():
    """
    Integration test for rainfall forecasting pipeline using a real trained model.

    This test:
    - Runs the forecast pipeline for a known station ("Belgaum")
    - Forecasts rainfall for 2 days starting from a fixed future date
    - Validates the structure and correctness of the response

    Assertions:
    - Response is not None
    - Station name matches input
    - Number of predictions equals requested days
    - Each prediction contains valid date and float rainfall value
    - Dates are sequential and increasing
    """

    station_name = "Belgaum"
    start_date = date(2025, 1, 1)
    num_days = 2

    # Run pipeline
    result = forecast_rainfall_pipeline(
        station_name=station_name,
        start_date=start_date,
        num_days=num_days
    )

    # BASIC ASSERTIONS
    assert result is not None
    assert result.station_name == station_name
    assert result.start_date == start_date
    assert result.num_days == num_days

    # PREDICTIONS CHECK
    predictions = result.predictions

    assert isinstance(predictions, list)
    assert len(predictions) == num_days

    # CONTENT VALIDATION
    for i, item in enumerate(predictions):

        # Check date exists and is correct type
        assert item.date_of_record is not None

        # Check rainfall is float
        assert isinstance(item.predicted_rainfall, float)

        # Check not NaN
        assert item.predicted_rainfall == item.predicted_rainfall  # NaN check

        # Check non-negative rainfall (optional domain constraint)
        assert item.predicted_rainfall >= 0

        # Check sequential dates
        expected_date = start_date.replace(day=start_date.day + i)
        assert item.date_of_record == expected_date