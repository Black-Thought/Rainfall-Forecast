from typing import List
from datetime import date
import pandas as pd
import numpy as np
from pathlib import Path

from app.utils.load_model import load_model_joblib
from app.schema.rainfall_forecasting import ForecastItem, ForecastResponse
from app.training.config.rainfall_forecast_config import FEATURES

from app.core.paths import WEATHER_DATA_PATH, MODEL_BASE_DIR


# -----------------------------------
# HAVERSINE DISTANCE (accurate)
# -----------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------

def forecast_from_location(
    latitude: float,
    longitude: float,
    start_date: date,
    num_days: int
) -> ForecastResponse:

    df: pd.DataFrame = pd.read_parquet(WEATHER_DATA_PATH, engine="pyarrow")
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])

    # -----------------------------------
    # STEP 1: GET UNIQUE STATIONS
    # -----------------------------------

    stations_df = df[[
        "station_name", "latitude", "longitude", "monsoon_zone"
    ]].drop_duplicates()

    # -----------------------------------
    # STEP 2: COMPUTE DISTANCE
    # -----------------------------------

    stations_df["distance_km"] = haversine_distance(
        latitude,
        longitude,
        stations_df["latitude"].values,
        stations_df["longitude"].values
    )

    # -----------------------------------
    # STEP 3: PICK TOP 10 NEAREST
    # -----------------------------------

    nearest = stations_df.nsmallest(10, "distance_km").copy()

    # -----------------------------------
    # STEP 4: DETERMINE DOMINANT ZONE
    # -----------------------------------

    dominant_zone = nearest["monsoon_zone"].mode()[0]

    # -----------------------------------
    # STEP 5: LOAD CORRECT MODEL
    # -----------------------------------

    model_path = Path(MODEL_BASE_DIR) / dominant_zone / "xgb_model.pkl"

    model = load_model_joblib(model_path)

    # -----------------------------------
    # STEP 6: WEIGHTS (inverse distance)
    # -----------------------------------

    nearest["weight"] = 1 / (nearest["distance_km"] + 1e-6)
    nearest["weight"] /= nearest["weight"].sum()

    # -----------------------------------
    # STEP 7: PREDICTION LOOP
    # -----------------------------------

    predictions: List[ForecastItem] = []

    current_date = pd.to_datetime(start_date)

    for _ in range(num_days):

        weighted_pred = 0.0

        for _, station in nearest.iterrows():

            station_df = df[df["station_name"] == station["station_name"]]
            station_df = station_df.sort_values("date_of_record")

            last_row = station_df.iloc[-1].copy()

            # Update time features
            last_row["date_of_record"] = current_date
            last_row["month"] = current_date.month

            last_row["month_sin"] = np.sin(2 * np.pi * last_row["month"] / 12)
            last_row["month_cos"] = np.cos(2 * np.pi * last_row["month"] / 12)

            X = pd.DataFrame([last_row[FEATURES]])

            pred = float(model.predict(X)[0])

            # Weighted sum
            weighted_pred += pred * station["weight"]

        predictions.append(
            ForecastItem(
                date_of_record=current_date.date(),
                predicted_rainfall=weighted_pred
            )
        )

        current_date += pd.Timedelta(days=1)

    # -----------------------------------
    # RETURN
    # -----------------------------------

    return ForecastResponse(
        station_name=f"Lat:{latitude}, Lon:{longitude}",
        start_date=start_date,
        num_days=num_days,
        predictions=predictions
    )