from pathlib import Path
from typing import Dict
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from app.training.config.rainfall_forecasting_zonewise_config import (
    FEATURES,
    TARGET,
    BEST_XGB_PARAMS,
)

# -----------------------------------
# PATHS
# -----------------------------------

APP_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = APP_DIR / "data" / "monsoon_zonewise_df.parquet"

MODEL_BASE_DIR: Path = APP_DIR / "models"
ZONEWISE_DIR: Path = MODEL_BASE_DIR / "RAINFALL_FORECAST_ZONEWISE"

# -----------------------------------
# BUILD MODEL
# -----------------------------------

def build_xgb_model(params: Dict) -> XGBRegressor:
    return XGBRegressor(
        **params,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

# -----------------------------------
# METRICS (TRAINING METRICS ONLY)
# -----------------------------------

def evaluate_on_training_data(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate model on training data only.
    NOTE: These metrics are optimistic (no generalization guarantee).
    """

    preds = model.predict(X)

    metrics: Dict[str, float] = {
        "mae": mean_absolute_error(y, preds),
        "rmse": root_mean_squared_error(y, preds),
        "r2": r2_score(y, preds),
    }

    print("\nTRAINING PERFORMANCE (100% DATA)")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics

# -----------------------------------
# TRAIN FUNCTION
# -----------------------------------

def train_and_save_zone_models(
    data_path: Path = DATA_PATH,
    model_dir: Path = ZONEWISE_DIR,
) -> Dict[str, XGBRegressor]:

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df: pd.DataFrame = pd.read_parquet(data_path, engine="pyarrow")
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])

    if "monsoon_zone" not in df.columns:
        raise ValueError("'monsoon_zone' column not found in dataset")

    model_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, XGBRegressor] = {}

    print("\n" + "═" * 60)
    print("Training Zone-wise XGBoost Models (100% Data)")
    print("═" * 60)

    # -----------------------------------
    # LOOP THROUGH ZONES
    # -----------------------------------

    for zone in df["monsoon_zone"].dropna().unique():

        print(f"\nZONE: {zone}")

        zone_df: pd.DataFrame = df[df["monsoon_zone"] == zone].copy()

        if zone_df.empty:
            print(f"Skipping {zone} (no data)")
            continue

        print(f"Total samples: {len(zone_df)}")

        X: pd.DataFrame = zone_df[FEATURES]
        y: pd.Series = zone_df[TARGET]

        # -----------------------------------
        # TRAIN
        # -----------------------------------

        model: XGBRegressor = build_xgb_model(BEST_XGB_PARAMS)

        print("Training model on full dataset...")
        model.fit(X, y)

        # -----------------------------------
        # EVALUATE (ON TRAINING DATA)
        # -----------------------------------

        evaluate_on_training_data(model, X, y)

        # -----------------------------------
        # SAVE MODEL
        # -----------------------------------

        zone_name: str = str(zone).upper().replace(" ", "_")
        model_filename: str = f"{zone_name}_XGB_MODEL.pkl"
        model_path: Path = model_dir / model_filename

        joblib.dump(model, model_path)

        print(f"Saved model → {model_path}")

        models[zone_name] = model

    print("\n" + "═" * 60)
    print("ALL ZONE MODELS TRAINED & SAVED")
    print("═" * 60)

    return models


# -----------------------------------
# ENTRY POINT
# -----------------------------------

if __name__ == "__main__":
    train_and_save_zone_models()