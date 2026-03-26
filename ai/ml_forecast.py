import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functools import reduce
from typing import Tuple, List, Dict

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
FEATURES = [
    "avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure",
    "air_density", "u_velocity", "v_velocity", "pressure_gradient",
    "divergence", "vorticity", "kinetic_energy", "temp_gradient",
    "rain_lag_1", "rain_lag_3", "rain_lag_7", "rain_lag_30",
    "month_sin", "month_cos"
]

TARGET   = "rainfall"
SIGMA    = 2

ACTUAL_COLOR    = '#00d4ff'
PREDICTED_COLOR = '#ff6b9d'

plt.rcParams.update({
    'font.family'        : 'DejaVu Sans',
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.grid'          : True,
    'grid.alpha'         : 0.3,
    'grid.linestyle'     : '--',
    'grid.color'         : '#cccccc',
    'figure.facecolor'   : '#0f1117',
    'axes.facecolor'     : '#1a1d2e',
    'axes.labelcolor'    : '#e0e0e0',
    'xtick.color'        : '#b0b0b0',
    'ytick.color'        : '#b0b0b0',
    'text.color'         : '#e0e0e0',
    'legend.framealpha'  : 0.3,
    'legend.facecolor'   : '#2a2d3e',
    'legend.edgecolor'   : '#555555',
})


# ══════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════
def load_data(filepath: str) -> pd.DataFrame:
    """Load and sort the raw CSV."""
    return (
        pd.read_csv(filepath)
        .assign(date_of_record=lambda df: pd.to_datetime(df["date_of_record"]))
        .sort_values(["station_name", "date_of_record"])
        .reset_index(drop=True)
    )


def filter_date_range(
    df: pd.DataFrame,
    start: str,
    end: str
) -> pd.DataFrame:
    """Return rows whose date_of_record falls in [start, end]."""
    mask = (df["date_of_record"] >= start) & (df["date_of_record"] <= end)
    return df.loc[mask].copy()


def filter_station(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Return rows for a single station."""
    return df.loc[df["station_name"] == station].copy()


# ══════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════
def build_model(**kwargs) -> XGBRegressor:
    """Construct an XGBRegressor with sensible defaults."""
    defaults = dict(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    return XGBRegressor(**{**defaults, **kwargs})


def train_model(
    df: pd.DataFrame,
    train_start: str = "2015-01-01",
    train_end:   str = "2022-12-31",
    features: List[str] = FEATURES,
    target:   str       = TARGET,
    **model_kwargs
) -> XGBRegressor:
    """
    Filter df to [train_start, train_end], fit and return a trained model.
    Pure function — no side effects other than the returned object.
    """
    train_df = filter_date_range(df, train_start, train_end)
    X = train_df[features]
    y = train_df[target]
    model = build_model(**model_kwargs)
    model.fit(X, y)
    print(f"✔ Model trained on {train_start} → {train_end}  ({len(X):,} rows)")
    return model


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Return a dict of MSE / RMSE / MAE."""
    mse = mean_squared_error(y_true, y_pred)
    return dict(mse=mse, rmse=np.sqrt(mse), mae=mean_absolute_error(y_true, y_pred))


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of metric dicts using reduce."""
    keys = [k for k in metrics_list[0].keys() if k != "station"]
    return reduce(
        lambda acc, m: {k: acc[k] + m[k] / len(metrics_list) for k in keys},
        metrics_list,
        {k: 0.0 for k in keys}
    )


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
def _smooth(arr: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=SIGMA)


def plot_forecast(
    dates:   pd.Series,
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    station: str,
    metrics: Dict[str, float],
    title_suffix: str = ""
) -> None:
    """Render a single station forecast chart."""
    x_idx          = np.arange(len(y_true))
    y_true_smooth  = _smooth(y_true)
    y_pred_smooth  = _smooth(y_pred)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14)

    ax.fill_between(x_idx, y_true_smooth, alpha=0.15, color=ACTUAL_COLOR)
    ax.fill_between(x_idx, y_pred_smooth, alpha=0.15, color=PREDICTED_COLOR)

    ax.scatter(x_idx, y_true, color=ACTUAL_COLOR,    alpha=0.25, s=18, zorder=2)
    ax.scatter(x_idx, y_pred, color=PREDICTED_COLOR, alpha=0.25, s=18, zorder=2)

    ax.plot(x_idx, y_true_smooth, color=ACTUAL_COLOR,    lw=2.2, label="Actual Rainfall",    zorder=3)
    ax.plot(x_idx, y_pred_smooth, color=PREDICTED_COLOR, lw=2.2, label="Predicted Rainfall", zorder=3, ls="--")

    # X-axis: compact date labels, max ~15 ticks
    tick_step = max(1, len(x_idx) // 15)
    tick_pos  = x_idx[::tick_step]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"Day {i}" for i in tick_pos], rotation=35, ha="right", fontsize=9)

    ax.set_xlabel("Days from Start", fontsize=11, labelpad=8)
    ax.set_ylabel("Rainfall (mm)", fontsize=11, labelpad=8)
    ax.set_title(
        f"Rainfall Forecast  ·  XGBoost  ·  {station}{title_suffix}",
        fontsize=14, fontweight="bold", pad=14, color="#ffffff"
    )

    metrics_text = (
        f"RMSE: {metrics['rmse']:.2f}   "
        f"MAE: {metrics['mae']:.2f}   "
        f"MSE: {metrics['mse']:.2f}"
    )
    ax.text(
        0.98, 0.97, metrics_text,
        transform=ax.transAxes, fontsize=9, color="#cccccc",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3e",
                  edgecolor="#555555", alpha=0.8)
    )

    legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.4)
    for txt in legend.get_texts():
        txt.set_color("#e0e0e0")

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  CORE PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════════
def predict(
    model:        XGBRegressor,
    df:           pd.DataFrame,
    station_name: str,
    date_of_record: str,
    num_days:     int,
    features:     List[str] = FEATURES,
    target:       str       = TARGET,
    plot:         bool      = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Predict rainfall for `station_name` starting from `date_of_record`
    for `num_days` consecutive days.

    Returns
    -------
    result_df  : DataFrame with columns [date_of_record, actual, predicted]
    metrics    : dict with mse / rmse / mae  (NaN if actuals unavailable)
    """
    # ── 1. Slice the window ──────────────────────────────────────
    start_dt = pd.to_datetime(date_of_record)
    end_dt   = start_dt + pd.Timedelta(days=num_days - 1)

    station_df = filter_station(df, station_name)
    window     = station_df[
        (station_df["date_of_record"] >= start_dt) &
        (station_df["date_of_record"] <= end_dt)
    ].copy()

    if window.empty:
        raise ValueError(
            f"No data found for station '{station_name}' "
            f"between {start_dt.date()} and {end_dt.date()}."
        )

    # ── 2. Predict ───────────────────────────────────────────────
    X       = window[features]
    y_pred  = model.predict(X)
    y_true  = window[target].values if target in window.columns else None

    # ── 3. Build result DataFrame ────────────────────────────────
    result_df = pd.DataFrame({
        "date_of_record" : window["date_of_record"].values,
        "actual"         : y_true if y_true is not None else np.nan,
        "predicted"      : y_pred,
    })

    # ── 4. Metrics ───────────────────────────────────────────────
    metrics = (
        compute_metrics(y_true, y_pred)
        if y_true is not None
        else dict(mse=np.nan, rmse=np.nan, mae=np.nan)
    )

    print(f"\n{'─'*50}")
    print(f"  Station : {station_name}")
    print(f"  Period  : {start_dt.date()} → {end_dt.date()}  ({len(window)} days)")
    print(f"  MSE     : {metrics['mse']:.4f}")
    print(f"  RMSE    : {metrics['rmse']:.4f}")
    print(f"  MAE     : {metrics['mae']:.4f}")
    print(f"{'─'*50}")

    # ── 5. Plot ───────────────────────────────────────────────────
    if plot and y_true is not None:
        period_label = f"  [{start_dt.strftime('%b %Y')} – {end_dt.strftime('%b %Y')}]"
        plot_forecast(
            dates=window["date_of_record"].reset_index(drop=True),
            y_true=y_true,
            y_pred=y_pred,
            station=station_name,
            metrics=metrics,
            title_suffix=period_label
        )

    return result_df, metrics


# ══════════════════════════════════════════════════════════════════
#  MULTI-STATION COMPARISON  (2023 → 2025)
# ══════════════════════════════════════════════════════════════════
def evaluate_stations(
    model:    XGBRegressor,
    df:       pd.DataFrame,
    stations: List[str],
    start:    str = "2023-01-01",
    end:      str = "2025-12-31",
) -> pd.DataFrame:
    """
    Run predict() for every station over [start, end] and return a
    summary DataFrame of metrics.
    """
    num_days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1

    rows = []
    for station in stations:
        try:
            _, m = predict(
                model=model,
                df=df,
                station_name=station,
                date_of_record=start,
                num_days=num_days,
                plot=True
            )
            rows.append({"station": station, **m})
        except ValueError as e:
            print(f"  ⚠  Skipping {station}: {e}")

    summary = pd.DataFrame(rows).set_index("station")

    # ── Print summary table ──────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  SUMMARY  —  Average Metrics Across All Stations")
    print(f"{'═'*55}")
    avg = aggregate_metrics([r for r in rows if not np.isnan(r["mse"])])
    print(f"  MSE  : {avg['mse']:.4f}")
    print(f"  RMSE : {avg['rmse']:.4f}")
    print(f"  MAE  : {avg['mae']:.4f}")
    print(f"{'═'*55}\n")

    return summary


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Load data
    df = load_data("processed_weather_data.csv")

    # 2. Train on 2015-01-01 → 2022-12-31
    model = train_model(df, train_start="2015-01-01", train_end="2022-12-31")

    # 3. Get station list
    stations = df["station_name"].unique()[:50].tolist()

    # 4. Full comparison: actual vs predicted for 2023-2025
    summary_df = evaluate_stations(
        model=model,
        df=df,
        stations=stations,
        start="2023-01-01",
        end="2025-12-31"
    )

    # ── Example: single ad-hoc prediction ───────────────────────
    # Predict 90 days for one station from a specific date
    result, metrics = predict(
        model=model,
        df=df,
        station_name=stations[0],
        date_of_record="2024-06-01",
        num_days=90
    )
    print(result.head(10))