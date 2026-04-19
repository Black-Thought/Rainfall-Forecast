import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functools import reduce
from typing import Tuple, List, Dict, Optional

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
FEATURES = [
    "avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure",
    "elevation", "latitude", "longitude", "temp_k", "pressure_pa",
    "air_density", "u_velocity", "v_velocity", "pressure_gradient",
    "du_dx", "dv_dy", "divergence", "dv_dx", "du_dy", "vorticity",
    "coriolis", "kinetic_energy", "temp_gradient",
    "rain_lag_1", "rain_lag_3", "rain_lag_7", "rain_lag_30",
    "month_sin", "month_cos"
]

TARGET = "rainfall"
SIGMA  = 2

ZONE_LABELS: Dict[int, str] = {
    0: "High_SW_Western_Ghats",
    1: "Rain_Shadow_Interior",
    2: "Moderate_Monsoon_Plains",
    3: "NE_Monsoon_Dominant",
    4: "Low_Rainfall_Arid",
}

# One colour per zone for consistent plot styling
ZONE_COLORS: Dict[int, Tuple[str, str]] = {
    0: ("#00d4ff", "#ff6b9d"),   # teal / pink
    1: ("#ffd166", "#ef476f"),   # amber / red
    2: ("#06d6a0", "#118ab2"),   # green / blue
    3: ("#f4a261", "#e76f51"),   # orange / coral
    4: ("#a8dadc", "#457b9d"),   # light-teal / navy
}

ACTUAL_COLOR    = "#00d4ff"
PREDICTED_COLOR = "#ff6b9d"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "grid.color":        "#cccccc",
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d2e",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#b0b0b0",
    "ytick.color":       "#b0b0b0",
    "text.color":        "#e0e0e0",
    "legend.framealpha": 0.3,
    "legend.facecolor":  "#2a2d3e",
    "legend.edgecolor":  "#555555",
})


# ══════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════
def load_data(filepath: str) -> pd.DataFrame:
    return (
        pd.read_csv(filepath)
        .assign(date_of_record=lambda df: pd.to_datetime(df["date_of_record"]))
        .sort_values(["station_name", "date_of_record"])
        .reset_index(drop=True)
    )


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date_of_record"] >= start) & (df["date_of_record"] <= end)
    return df.loc[mask].copy()


def filter_station(df: pd.DataFrame, station: str) -> pd.DataFrame:
    return df.loc[df["station_name"] == station].copy()


def filter_zone(df: pd.DataFrame, zone_id: int) -> pd.DataFrame:
    return df.loc[df["zone"] == zone_id].copy()



from sklearn.metrics import r2_score

def compute_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / (denominator + 1e-6))

def compute_all_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nse = compute_nse(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nse": nse
    }




# ══════════════════════════════════════════════════════════════════
#  ZONE MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════
class ZoneModelRegistry:

    def __init__(self):
        self._models = {}
        self._station_split = {}

    def _build_model(self):
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def train_zone(self, df, zone_name):

        zone_df = df[df["monsoon_zone"] == zone_name].copy()

        stations = zone_df["station_name"].unique()
        np.random.shuffle(stations)

        split_idx = int(0.7 * len(stations))

        train_stations = stations[:split_idx]
        test_stations  = stations[split_idx:]

        train_df = zone_df[zone_df["station_name"].isin(train_stations)]
        test_df  = zone_df[zone_df["station_name"].isin(test_stations)]

        if train_df.empty or test_df.empty:
            print(f"⚠ Skipping {zone_name}")
            return

        model = self._build_model()
        model.fit(train_df[FEATURES], train_df[TARGET])

        self._models[zone_name] = model
        self._station_split[zone_name] = (train_stations, test_stations)

        print(f"✔ {zone_name}: {len(train_stations)} train | {len(test_stations)} test stations")

    def train_all(self, df):

        print("\n" + "═"*60)
        print("Training 3 Monsoon Zone Models (Station Split)")
        print("═"*60)

        for zone in df["monsoon_zone"].dropna().unique():
            self.train_zone(df, zone)

        print("═"*60 + "\n")

    def get_model(self, zone_name):
        return self._models[zone_name]

    def get_split(self, zone_name):
        return self._station_split[zone_name]

# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return dict(mse=mse, rmse=np.sqrt(mse), mae=mean_absolute_error(y_true, y_pred))


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [k for k in metrics_list[0] if k not in ("station", "zone_id", "zone_name")]
    return reduce(
        lambda acc, m: {k: acc[k] + m[k] / len(metrics_list) for k in keys},
        metrics_list,
        {k: 0.0 for k in keys},
    )


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
def _smooth(arr: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=SIGMA)

import os
import re

def save_forecast_plot(
    fig,
    station: str,
    zone_name: str,
    dates: pd.Series
):
    """
    Saves the matplotlib figure into:
    forecasts/<ZONE_NAME>/<station>.png
    """

    base_dir = "forecasts"
    zone_dir = os.path.join(base_dir, zone_name)

    os.makedirs(zone_dir, exist_ok=True)

    # 🔥 FIX: clean station name (remove invalid characters)
    safe_station = re.sub(r'[\\/*?:"<>|]', "_", station)

    # Also remove extra spaces
    safe_station = safe_station.replace(" ", "_")

    # Date range
    start_date = pd.to_datetime(dates.iloc[0]).strftime('%b_%Y')
    end_date   = pd.to_datetime(dates.iloc[-1]).strftime('%b_%Y')

    filename = f"{safe_station}_{start_date}_to_{end_date}.png"

    filepath = os.path.join(zone_dir, filename)

    fig.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"💾 Saved: {filepath}")

def plot_forecast(
    dates:    pd.Series,
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    station:  str,
    metrics:  Dict[str, float],
    zone_id:  Optional[int] = None,
    zone_name: Optional[str] = None,
    title_suffix: str = "",
) -> None:

    actual_col, pred_col = ZONE_COLORS.get(zone_id, (ACTUAL_COLOR, PREDICTED_COLOR))

    x_idx         = np.arange(len(y_true))
    y_true_smooth = _smooth(y_true)
    y_pred_smooth = _smooth(y_pred)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14)

    ax.fill_between(x_idx, y_true_smooth, alpha=0.15, color=actual_col)
    ax.fill_between(x_idx, y_pred_smooth, alpha=0.15, color=pred_col)
    ax.scatter(x_idx, y_true, color=actual_col, alpha=0.25, s=18, zorder=2)
    ax.scatter(x_idx, y_pred, color=pred_col,   alpha=0.25, s=18, zorder=2)
    ax.plot(x_idx, y_true_smooth, color=actual_col, lw=2.2, label="Actual Rainfall", zorder=3)
    ax.plot(x_idx, y_pred_smooth, color=pred_col,   lw=2.2, label="Predicted Rainfall", zorder=3, ls="--")

    tick_step = max(1, len(x_idx) // 15)
    tick_pos  = x_idx[::tick_step]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"Day {i}" for i in tick_pos], rotation=35, ha="right", fontsize=9)

    ax.set_xlabel("Days from Start", fontsize=11, labelpad=8)
    ax.set_ylabel("Rainfall (mm)", fontsize=11, labelpad=8)

    start_date = pd.to_datetime(dates.iloc[0]).strftime('%b %Y')
    end_date   = pd.to_datetime(dates.iloc[-1]).strftime('%b %Y')

    zone_tag = f"[{zone_name}]" if zone_name else ""

    ax.set_title(
        f"Rainfall Forecast Model · {station} {zone_tag} · {start_date} – {end_date}",
        fontsize=14,
        fontweight="bold",
        pad=14,
        color="#ffffff",
    )
    metrics_text = (
        f"RMSE: {metrics['rmse']:.2f}   "
        f"MAE: {metrics['mae']:.2f}   "
        f"MSE: {metrics['mse']:.2f}"
    )

    ax.text(
        0.98, 0.97, metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#cccccc",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3e",
                  edgecolor="#555555", alpha=0.8),
    )

    legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.4)
    for txt in legend.get_texts():
        txt.set_color("#e0e0e0")

    plt.tight_layout()
# Save plot
    save_forecast_plot(fig, station, zone_name, dates)



# ══════════════════════════════════════════════════════════════════
#  CORE PREDICT FUNCTION  (zone-aware)
# ══════════════════════════════════════════════════════════════════
def predict(
    registry:       ZoneModelRegistry,
    df:             pd.DataFrame,
    station_name:   str,
    date_of_record: str,
    num_days:       int,
    features:       List[str] = FEATURES,
    target:         str       = TARGET,
    plot:           bool      = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Predict rainfall for `station_name` using the zone-specific model.

    The zone is resolved automatically from the station's `zone_label`
    column in `df`, then the matching model is fetched from `registry`.
    """
    # ── 1. Resolve zone ──────────────────────────────────────────
    zone_id   = registry.zone_for_station(df, station_name)
    zone_name = registry.zone_labels[zone_id]
    model     = registry.get_model(zone_id)

    # ── 2. Slice the time window ─────────────────────────────────
    start_dt   = pd.to_datetime(date_of_record)
    end_dt     = start_dt + pd.Timedelta(days=num_days - 1)
    station_df = filter_station(df, station_name)
    window     = station_df[
        (station_df["date_of_record"] >= start_dt) &
        (station_df["date_of_record"] <= end_dt)
    ].copy()

    if window.empty:
        raise ValueError(
            f"No data for station '{station_name}' "
            f"between {start_dt.date()} and {end_dt.date()}."
        )

    # ── 3. Predict ───────────────────────────────────────────────
    X      = window[features]
    y_pred = model.predict(X)
    y_true = window[target].values if target in window.columns else None

    # ── 4. Result DataFrame ──────────────────────────────────────
    result_df = pd.DataFrame({
        "date_of_record": window["date_of_record"].values,
        "zone_id":        zone_id,
        "zone_name":      zone_name,
        "actual":         y_true if y_true is not None else np.nan,
        "predicted":      y_pred,
    })

    # ── 5. Metrics ───────────────────────────────────────────────
    metrics = (
        compute_metrics(y_true, y_pred)
        if y_true is not None
        else dict(mse=np.nan, rmse=np.nan, mae=np.nan)
    )

    print(f"\n{'─'*56}")
    print(f"  Station  : {station_name}")
    print(f"  Zone     : {zone_id} — {zone_name}")
    print(f"  Period   : {start_dt.date()} → {end_dt.date()}  ({len(window)} days)")
    print(f"  MSE      : {metrics['mse']:.4f}")
    print(f"  RMSE     : {metrics['rmse']:.4f}")
    print(f"  MAE      : {metrics['mae']:.4f}")
    print(f"{'─'*56}")

    # ── 6. Plot ───────────────────────────────────────────────────
    if plot and y_true is not None:
        period_label = f"  [{start_dt.strftime('%b %Y')} – {end_dt.strftime('%b %Y')}]"
        plot_forecast(
            dates=window["date_of_record"].reset_index(drop=True),
            y_true=y_true,
            y_pred=y_pred,
            station=station_name,
            metrics=metrics,
            zone_id=zone_id,
            zone_name=zone_name,
            title_suffix=period_label,
        )

    return result_df, metrics


# ══════════════════════════════════════════════════════════════════
#  MULTI-STATION EVALUATION  (zone-aware)
# ══════════════════════════════════════════════════════════════════
def evaluate_station_split(registry, df):

    results = []

    for zone_name in registry._models:

        model = registry.get_model(zone_name)
        train_stations, test_stations = registry.get_split(zone_name)

        print(f"\n{'='*70}")
        print(f"ZONE: {zone_name}")
        print(f"{'='*70}")

        zone_metrics = []

        for station in test_stations[:10]:   # 🔥 10 stations per zone

            station_df = df[df["station_name"] == station].copy()

            if len(station_df) < 50:
                continue

            X = station_df[FEATURES]
            y_true = station_df[TARGET].values
            y_pred = model.predict(X)

            metrics = compute_all_metrics(y_true, y_pred)
            zone_metrics.append(metrics)

            results.append({
                "station": station,
                "zone": zone_name,
                **metrics
            })

            print(f"\n📍 {station} [{zone_name}]")
            print(f"RMSE: {metrics['rmse']:.3f}")
            print(f"MAE : {metrics['mae']:.3f}")
            print(f"MSE : {metrics['mse']:.3f}")
            print(f"NSE : {metrics['nse']:.3f}")
            print(f"R2  : {metrics['r2']:.3f}")

            # 🔥 Plot
            plot_forecast(
                dates=station_df["date_of_record"].reset_index(drop=True),
                y_true=y_true,
                y_pred=y_pred,
                station=station,
                metrics=metrics,
                zone_id=None,
                zone_name=zone_name,
                title_suffix=" [Station Split]"
            )

        # ── ZONE SUMMARY ───────────────────────────────
        zone_df = pd.DataFrame(zone_metrics)

        print(f"\n🔷 ZONE SUMMARY: {zone_name}")
        print(zone_df.mean())

    # ── OVERALL SUMMARY ───────────────────────────────
    results_df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("🔥 FINAL OVERALL METRICS")
    print(f"{'='*70}")
    print(results_df[["rmse","mae","mse","nse","r2"]].mean())

    return results_df



# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Load your FINAL dataset (must contain 'monsoon_zone')
    df = load_data("final_monsoon_zones.csv")

    # Train models
    registry = ZoneModelRegistry()
    registry.train_all(df)

    # Evaluate
    results_df = evaluate_station_split(registry, df)

    print("\n📊 Final Results:")
    print(results_df.groupby("zone")[["rmse", "mae"]].mean())