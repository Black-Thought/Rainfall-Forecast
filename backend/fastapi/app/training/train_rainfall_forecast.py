from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from app.training.config.rainfall_forecast_config import FEATURES, TARGET, BEST_XGB_PARAMS


APP_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = APP_DIR / "data" / "processed_weather_data.csv"
MODEL_PATH = APP_DIR / "models" / "xgb_rainfall_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)


def load_and_split_data(
    data_path: Path,
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and perform time-based train-test split.

    Args:
        data_path (Path): Path to CSV dataset
        train_end (str): End date for training data
        test_start (str): Start date for test data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_df, test_df
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])

    train_df = df[df["date_of_record"] <= train_end].copy()
    test_df = df[df["date_of_record"] >= test_start].copy()

    print(f"Train size: {len(train_df)}")
    print(f"Test size : {len(test_df)}")

    return train_df, test_df


def build_xgb_model(params: Dict) -> XGBRegressor:
    """
    Build XGBoost regressor with given hyperparameters.

    Args:
        params (Dict): Hyperparameter dictionary

    Returns:
        XGBRegressor: Initialized model
    """
    return XGBRegressor(
        **params,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )


def evaluate_regression(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate regression model using MAE, RMSE, and R².

    Args:
        model (XGBRegressor): Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    metrics = {
        "train_mae": mean_absolute_error(y_train, train_preds),
        "test_mae": mean_absolute_error(y_test, test_preds),
        "train_rmse": root_mean_squared_error(y_train, train_preds),
        "test_rmse": root_mean_squared_error(y_test, test_preds),
        "train_r2": r2_score(y_train, train_preds),
        "test_r2": r2_score(y_test, test_preds),
    }

    print("\n MODEL PERFORMANCE")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def train_and_save_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH
) -> XGBRegressor:
    """
    Full training pipeline:
    - Load data
    - Split train/test
    - Train XGBoost model
    - Evaluate performance
    - Save trained model

    Args:
        data_path (Path): Dataset path
        model_path (Path): Path to save trained model

    Returns:
        XGBRegressor: Trained model
    """

    # Load + split
    train_df, test_df = load_and_split_data(data_path)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Build model
    model = build_xgb_model(BEST_XGB_PARAMS)

    print("\n Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    evaluate_regression(model, X_train, y_train, X_test, y_test)

    # Save
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    return model


if __name__ == "__main__":
    train_and_save_model()