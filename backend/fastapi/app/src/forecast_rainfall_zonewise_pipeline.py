from typing import List, Dict
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

from app.schema.distance import Coordinates
from app.schema.rainfall_forecasting_zonewise import (
    ZonewiseRainfallForecastItem,
    ZonewiseRainfallForecastResponse,
)

from app.training.config.rainfall_forecasting_zonewise_config import FEATURES
from app.core.paths import WEATHER_DATA_PATH, MODEL_BASE_DIR
from app.utils.distance import haversine_distance
from app.utils.load_model import load_model_joblib


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------

def forecast_rainfall_zonewise_pipeline(
    coordinates: Coordinates,
    start_date: date,
    num_days: int,
    sensitivity: int = 5,
) -> ZonewiseRainfallForecastResponse:
    """
    Zone-wise rainfall forecasting pipeline using nearest weather stations
    with recursive (auto-regressive) prediction.

    - Supports ungauged locations
    - Uses inverse distance weighting
    - Maintains per-station temporal state

    Returns:
        ZonewiseRainfallForecastResponse
    """

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    df: pd.DataFrame = pd.read_parquet(WEATHER_DATA_PATH, engine="pyarrow")
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])

    # -----------------------------------
    # UNIQUE STATIONS
    # -----------------------------------

    stations_df = df[
        ["station_name", "latitude", "longitude", "monsoon_zone"]
    ].drop_duplicates()

    # -----------------------------------
    # DISTANCE COMPUTATION
    # -----------------------------------

    stations_df["distance_km"] = stations_df.apply(
        lambda row: haversine_distance(
            coordinates,
            Coordinates(lat=row["latitude"], lon=row["longitude"]),
        ),
        axis=1,
    )

    nearest = stations_df.nsmallest(sensitivity, "distance_km").copy()

    if nearest.empty:
        raise ValueError("No nearby stations found for given coordinates")

    # -----------------------------------
    # DOMINANT ZONE
    dominant_zone: str = nearest["monsoon_zone"].mode()[0]
    zone_name: str = dominant_zone.upper().replace(" ", "_")

    # LOAD MODEL
    model_path: Path = (
        Path(MODEL_BASE_DIR)
        / "RAINFALL_FORECAST_ZONEWISE"
        / f"{zone_name}_XGB_MODEL.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model_joblib(model_path)

    # WEIGHTS
    nearest["weight"] = 1 / (nearest["distance_km"] + 1e-6)
    nearest["weight"] /= nearest["weight"].sum()

    # PRECOMPUTE STATION DATA (OPTIMIZATION)
    station_data: Dict[str, pd.DataFrame] = {
        name: group.sort_values("date_of_record")
        for name, group in df.groupby("station_name")
    }

    # RECURSIVE STATE (PER STATION)
    prev_rows: Dict[str, pd.Series] = {}

    # PREDICTION LOOP
    predictions: List[ZonewiseRainfallForecastItem] = []
    current_date = pd.to_datetime(start_date)

    for _ in range(num_days):

        weighted_pred: float = 0.0

        for _, station in nearest.iterrows():

            station_name = station["station_name"]
            station_df = station_data.get(station_name)

            if station_df is None or station_df.empty:
                continue

            # INITIAL vs RECURSIVE STEP
            if station_name not in prev_rows:
                last_row = station_df.iloc[-1].copy()
            else:
                last_row = prev_rows[station_name].copy()

            # TIME FEATURES
            last_row["date_of_record"] = current_date
            last_row["month"] = current_date.month
            last_row["month_sin"] = np.sin(2 * np.pi * last_row["month"] / 12)
            last_row["month_cos"] = np.cos(2 * np.pi * last_row["month"] / 12)
            
            # FEATURE SAFETY
            for col in FEATURES:
                if col not in last_row:
                    last_row[col] = 0.0

            X = pd.DataFrame([last_row])[FEATURES]

            # PREDICT
            pred: float = float(model.predict(X)[0])
            pred = max(pred, 0.0)

            weighted_pred += pred * station["weight"]

            # UPDATE STATION STATE
            new_row = last_row.copy()

            new_row["rain_lag_30"] = new_row.get("rain_lag_7", 0)
            new_row["rain_lag_7"] = new_row.get("rain_lag_3", 0)
            new_row["rain_lag_3"] = new_row.get("rain_lag_1", 0)
            new_row["rain_lag_1"] = pred

            prev_rows[station_name] = new_row

        # STORE DAILY PREDICTION
        predictions.append(
            ZonewiseRainfallForecastItem(
                date_of_record=current_date.date(),
                predicted_rainfall=round(max(weighted_pred, 0.0), 2),
            )
        )

        current_date += pd.Timedelta(days=1)

    # RESPONSE
    return ZonewiseRainfallForecastResponse(
        location=coordinates,
        dominant_zone=zone_name,
        start_date=start_date,
        num_days=num_days,
        sensitivity=sensitivity,
        predictions=predictions,
    )
    
# if __name__ == "__main__":
#     # Example usage
#     response = forecast_rainfall_zonewise_pipeline(
#         coordinates=Coordinates(lat=19.0760, lon=72.8777),  # Mumbai
#         start_date=date(2026, 7, 1),
#         num_days=7,
#         sensitivity=5,
#     )

#     print(response.model_dump())