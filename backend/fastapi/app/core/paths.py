from pathlib import Path

# -----------------------------------
# BASE DIRECTORIES
# -----------------------------------

# app/ directory
APP_DIR: Path = Path(__file__).resolve().parent.parent

# project root (fastapi/)
ROOT_DIR: Path = APP_DIR.parent

# -----------------------------------
# DATA DIRECTORIES
# -----------------------------------

DATA_DIR: Path = APP_DIR / "data"

# Main dataset (used in inference pipeline)
WEATHER_DATA_PATH: Path = DATA_DIR / "monsoon_zonewise_df.parquet"

# (Optional alternate dataset — keep if needed)
PROCESSED_WEATHER_DATA_PATH: Path = DATA_DIR / "processed_weather_data.parquet"

# -----------------------------------
# MODEL DIRECTORIES
# -----------------------------------

MODEL_BASE_DIR: Path = APP_DIR / "models"

# Single global model (if used anywhere)
XGB_RAINFALL_FORECAST_MODEL_PATH: Path = (
    MODEL_BASE_DIR / "xgb_rainfall_forecast.pkl"
)

# Zone-wise models directory
ZONEWISE_MODEL_DIR: Path = MODEL_BASE_DIR / "RAINFALL_FORECAST_ZONEWISE"