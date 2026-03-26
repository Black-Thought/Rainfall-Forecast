from typing import List
from datetime import date
import pandas as pd
import numpy as np
from app.core.paths import (
    XGB_RAINFALL_FORECAST_MODEL_PATH,
    WEATHER_DATA_PATH
)
from app.utils.load_model import load_model_joblib
from app.schema.rainfall_forecasting import ForecastItem, ForecastResponse
from app.training.config.rainfall_forecast_config import FEATURES


def forecast_rainfall_pipeline(
    station_name: str,
    start_date: date,
    num_days: int
) -> ForecastResponse:
    """
    Forecast rainfall for a station for future dates.
    """

    #  Load model + data
    model = load_model_joblib(XGB_RAINFALL_FORECAST_MODEL_PATH)

    df: pd.DataFrame = pd.read_csv(WEATHER_DATA_PATH)
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])

    # Filter station
    station_df: pd.DataFrame = df[df["station_name"] == station_name].copy()

    if station_df.empty:
        raise ValueError(f"Station '{station_name}' not found")

    station_df = station_df.sort_values("date_of_record")

    # Start from last known row
    last_row: pd.Series = station_df.iloc[-1].copy()
    current_date: pd.Timestamp = pd.to_datetime(start_date)

    predictions: List[ForecastItem] = []

    for _ in range(num_days):

        row = last_row.copy()

        # Update time features
        row["date_of_record"] = current_date
        row["month"] = current_date.month

        row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
        row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)

        # Predict
        X: pd.DataFrame = pd.DataFrame([row[FEATURES]])
        pred: float = float(model.predict(X)[0])

        predictions.append(
            ForecastItem(
                date_of_record=current_date.date(),
                predicted_rainfall=pred
            )
        )

        # Update lag features
        row["rain_lag_1"] = pred
        row["rain_lag_3"] = row.get("rain_lag_1", pred)
        row["rain_lag_7"] = row.get("rain_lag_3", pred)
        row["rain_lag_30"] = row.get("rain_lag_7", pred)

        last_row = row
        current_date += pd.Timedelta(days=1)

    return ForecastResponse(
        station_name=station_name,
        start_date=start_date,
        num_days=num_days,
        predictions=predictions
    )