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
    """
    Build and return an XGBoost regressor with optimized parameters.

    Args:
        params (Dict): Dictionary of hyperparameters for XGBoost.

    Returns:
        XGBRegressor: Configured XGBoost regression model.
    """
    return XGBRegressor(
        **params,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )


# -----------------------------------
# METRICS
# -----------------------------------

def evaluate_regression(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate regression model performance on train and test data.

    Returns:
        Dict[str, float]: Dictionary containing MAE, RMSE, and R² scores.
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    metrics: Dict[str, float] = {
        "train_mae": mean_absolute_error(y_train, train_preds),
        "test_mae": mean_absolute_error(y_test, test_preds),
        "train_rmse": root_mean_squared_error(y_train, train_preds),
        "test_rmse": root_mean_squared_error(y_test, test_preds),
        "train_r2": r2_score(y_train, train_preds),
        "test_r2": r2_score(y_test, test_preds),
    }

    print("\nMODEL PERFORMANCE")
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
    """
    Train separate XGBoost models for each monsoon zone and save them.

    Workflow:
    - Load dataset from parquet
    - Split data into train/test using time-based split
    - Train one model per monsoon zone
    - Evaluate model performance
    - Save trained models in structured directory

    Args:
        data_path (Path): Path to input dataset.
        model_dir (Path): Directory where trained models will be saved.

    Returns:
        Dict[str, XGBRegressor]: Dictionary mapping zone names to trained models.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df: pd.DataFrame = pd.read_parquet(data_path, engine="pyarrow")
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])

    if "monsoon_zone" not in df.columns:
        raise ValueError("'monsoon_zone' column not found in dataset")

    model_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, XGBRegressor] = {}

    print("\n" + "═" * 60)
    print("Training Zone-wise XGBoost Models")
    print("═" * 60)

    # -----------------------------------
    # LOOP THROUGH ZONES
    # -----------------------------------

    for zone in df["monsoon_zone"].dropna().unique():

        print(f"\nZONE: {zone}")

        zone_df: pd.DataFrame = df[df["monsoon_zone"] == zone].copy()

        # -----------------------------------
        # TIME SPLIT
        # -----------------------------------

        train_df = zone_df[zone_df["date_of_record"] <= "2023-12-31"]
        test_df = zone_df[zone_df["date_of_record"] >= "2024-01-01"]

        print(f"Train size: {len(train_df)}")
        print(f"Test size : {len(test_df)}")

        if train_df.empty or test_df.empty:
            print(f"Skipping {zone} (insufficient data)")
            continue

        X_train: pd.DataFrame = train_df[FEATURES]
        y_train: pd.Series = train_df[TARGET]

        X_test: pd.DataFrame = test_df[FEATURES]
        y_test: pd.Series = test_df[TARGET]

        # -----------------------------------
        # TRAIN
        # -----------------------------------

        model: XGBRegressor = build_xgb_model(BEST_XGB_PARAMS)

        print("Training model...")
        model.fit(X_train, y_train)

        # -----------------------------------
        # EVALUATE
        # -----------------------------------

        evaluate_regression(model, X_train, y_train, X_test, y_test)

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