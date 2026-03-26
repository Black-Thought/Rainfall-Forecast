import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.training.train_rainfall_forecast import (
    load_and_split_data,
    build_xgb_model,
    evaluate_regression,
    train_and_save_model,
    FEATURES,
    TARGET,
)


# -----------------------------
# MOCK DATA
# -----------------------------
@pytest.fixture
def mock_dataframe():
    """Create a small mock dataframe with required columns."""
    data = {
        "date_of_record": pd.date_range(start="2023-01-01", periods=20, freq="D"),
        "rainfall": [1.0] * 20,
    }

    for col in FEATURES:
        data[col] = [0.5] * 20

    return pd.DataFrame(data)


# -----------------------------
# TEST: MODEL BUILD
# -----------------------------
def test_build_xgb_model():
    model = build_xgb_model({})

    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


# -----------------------------
# TEST: EVALUATION
# -----------------------------
def test_evaluate_regression(mock_dataframe):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1.0] * len(mock_dataframe)

    X = mock_dataframe[FEATURES]
    y = mock_dataframe[TARGET]

    metrics = evaluate_regression(mock_model, X, y, X, y)

    assert "train_mae" in metrics
    assert "test_rmse" in metrics
    assert "train_r2" in metrics
    assert isinstance(metrics["train_mae"], float)



# -----------------------------
# TEST: FILE NOT FOUND
# -----------------------------
@patch("app.training.train_rainfall_forecast.Path.exists")
def test_load_and_split_data_file_not_found(mock_exists):
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError):
        load_and_split_data(Path("missing.parquet"))


# -----------------------------
# TEST: FULL TRAIN PIPELINE
# -----------------------------
@patch("app.training.train_rainfall_forecast.joblib.dump")
@patch("app.training.train_rainfall_forecast.build_xgb_model")
@patch("app.training.train_rainfall_forecast.load_and_split_data")
def test_train_and_save_model(
    mock_split,
    mock_build_model,
    mock_joblib_dump,
    mock_dataframe
):
    # Mock split
    mock_split.return_value = (mock_dataframe, mock_dataframe)

    # Mock model
    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.predict.return_value = [1.0] * len(mock_dataframe)

    mock_build_model.return_value = mock_model

    model = train_and_save_model(
        data_path=Path("dummy.parquet"),
        model_path=Path("dummy.pkl")
    )

    # Assertions
    mock_split.assert_called_once()
    mock_build_model.assert_called_once()
    mock_model.fit.assert_called_once()
    mock_joblib_dump.assert_called_once()

    assert model == mock_model