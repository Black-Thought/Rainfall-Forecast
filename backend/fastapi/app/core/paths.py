from pathlib import Path

# app/ directory
APP_DIR = Path(__file__).resolve().parent.parent

# project root (fastapi/)
ROOT_DIR = APP_DIR.parent

# data path
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"

# dataset path
WEATHER_DATA_PATH = DATA_DIR / "processed_weather_data.csv"

# models path
XGB_RAINFALL_FORECAST_MODEL_PATH = MODEL_DIR / "xgb_rainfall_forecast.pkl"