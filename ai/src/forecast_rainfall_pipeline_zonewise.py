"""
forecast_rainfall_pipeline_zonewise.py
---------------------------------------
Standalone inference pipeline: given a (lat, lon), start_date, and num_days,
it finds the nearest stations, determines their dominant monsoon zone, loads
the corresponding trained XGBoost model, and returns a weighted-average daily
rainfall forecast.

Usage (from the project root ai/):
    python -m src.forecast_rainfall_pipeline_zonewise
"""

from typing import List
from datetime import date
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from .ml_forecast_zones import FEATURES

# ── Resolve paths relative to project root (ai/) ─────────────────────────────
_SRC_DIR          = Path(__file__).resolve().parent
_ROOT             = _SRC_DIR.parent
WEATHER_DATA_PATH = _ROOT / "dataset" / "final_monsoon_zones.csv"
MODEL_BASE_DIR    = _ROOT / "models"


# ── Lightweight response dataclasses (no external dependencies) ───────────────
@dataclass
class ForecastItem:
    date_of_record: date
    predicted_rainfall: float


@dataclass
class ForecastResponse:
    station_name: str
    start_date: date
    num_days: int
    predictions: List[ForecastItem] = field(default_factory=list)


# ── Model loading helper ──────────────────────────────────────────────────────
def load_model_joblib(model_path: Path):
    """Load a joblib-serialised XGBoost model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)


# -----------------------------------
# HAVERSINE DISTANCE
# -----------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometres."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------

def forecast_from_location(
    latitude: float,
    longitude: float,
    start_date: date,
    num_days: int,
    data_path: Path = WEATHER_DATA_PATH,
    model_base_dir: Path = MODEL_BASE_DIR,
) -> ForecastResponse:
    """
    Forecast daily rainfall for `num_days` starting from `start_date`
    at the geographic location (latitude, longitude).

    Parameters
    ----------
    latitude, longitude : float
        Target coordinates.
    start_date : datetime.date
        First day of the forecast window.
    num_days : int
        Number of forecast days.
    data_path : Path
        Path to the processed weather CSV (must contain monsoon_zone column).
    model_base_dir : Path
        Directory that contains zone sub-folders with xgb_model.pkl files.

    Returns
    -------
    ForecastResponse
        Dataclass with a list of ForecastItem (date, predicted_rainfall).
    """
    # Load dataset
    df = pd.read_csv(data_path)
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])

    # -----------------------------------
    # STEP 1: GET UNIQUE STATIONS
    # -----------------------------------
    stations_df = df[
        ["station_name", "latitude", "longitude", "monsoon_zone"]
    ].drop_duplicates()

    # -----------------------------------
    # STEP 2: COMPUTE HAVERSINE DISTANCE
    # -----------------------------------
    stations_df = stations_df.copy()
    stations_df["distance_km"] = haversine_distance(
        latitude,
        longitude,
        stations_df["latitude"].values,
        stations_df["longitude"].values,
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
    model_path = Path(model_base_dir) / dominant_zone / "xgb_model.pkl"
    model = load_model_joblib(model_path)

    # -----------------------------------
    # STEP 6: INVERSE-DISTANCE WEIGHTS
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
            station_df = df[df["station_name"] == station["station_name"]].sort_values(
                "date_of_record"
            )
            last_row = station_df.iloc[-1].copy()

            # Update cyclic time features for the target date
            last_row["date_of_record"] = current_date
            last_row["month"] = current_date.month
            last_row["month_sin"] = np.sin(2 * np.pi * last_row["month"] / 12)
            last_row["month_cos"] = np.cos(2 * np.pi * last_row["month"] / 12)

            X = pd.DataFrame([last_row[FEATURES]])
            pred = float(model.predict(X)[0])
            weighted_pred += pred * station["weight"]

        predictions.append(
            ForecastItem(
                date_of_record=current_date.date(),
                predicted_rainfall=max(0.0, weighted_pred),  # rainfall ≥ 0
            )
        )
        current_date += pd.Timedelta(days=1)

    # -----------------------------------
    # RETURN
    # -----------------------------------
    return ForecastResponse(
        station_name=f"Lat:{latitude:.4f}, Lon:{longitude:.4f}",
        start_date=start_date,
        num_days=num_days,
        predictions=predictions,
    )


# ── Quick sanity-check entry point ────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import date as _date

    result = forecast_from_location(
        latitude=19.0760,
        longitude=72.8777,   # Mumbai
        start_date=_date(2024, 6, 1),
        num_days=7,
    )

    print(f"\n📍 Location : {result.station_name}")
    print(f"📅 Start    : {result.start_date}  |  Days: {result.num_days}")
    print(f"\n{'Date':<14}{'Predicted Rainfall (mm)':>24}")
    print("─" * 38)
    for item in result.predictions:
        print(f"{str(item.date_of_record):<14}{item.predicted_rainfall:>24.2f}")