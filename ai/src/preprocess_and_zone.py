"""
preprocess_and_zone.py
======================
Combined data preprocessing + monsoon zone assignment pipeline.

Extracted from:
  - data.ipynb          (cleaning, feature engineering, saved as processed_weather_data.csv)
  - zoning_stations.ipynb (monsoon zone classification, saved as final_monsoon_zones.csv)

Usage (from the ai/ root directory):
    python -m src.preprocess_and_zone

Input:
    dataset/dataset.csv              -- raw weather observations

Outputs:
    dataset/processed_weather_data.csv   -- cleaned + feature-engineered data
    dataset/final_monsoon_zones.csv      -- processed data with monsoon_zone column
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent          # ai/
DATASET_DIR = _ROOT / "dataset"

RAW_DATA_PATH       = DATASET_DIR / "dataset.csv"
PROCESSED_DATA_PATH = DATASET_DIR / "processed_weather_data.csv"
ZONED_DATA_PATH     = DATASET_DIR / "final_monsoon_zones.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — DATA CLEANING & FEATURE ENGINEERING  (from data.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(path: Path) -> pd.DataFrame:
    """Load raw CSV, drop rows with missing rainfall, interpolate weather cols."""
    print(f"\n📂 Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"   Shape after load     : {df.shape}")

    # ── Missing value summary ──────────────────────────────────────────────────
    missing = (
        pd.DataFrame({
            "missing_count":   df.isnull().sum(),
            "missing_percent": (df.isnull().sum() / len(df)) * 100,
        })
        .query("missing_count > 0")
        .sort_values("missing_percent", ascending=False)
    )
    if not missing.empty:
        print("\n   Missing values (before cleaning):")
        print(missing.to_string())

    # ── Drop rows where target (rainfall) is missing ───────────────────────────
    df = df.dropna(subset=["rainfall"])
    print(f"\n   Shape after dropping missing rainfall: {df.shape}")

    # ── Interpolate weather columns per station (linear → ffill → bfill) ───────
    weather_cols = ["avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure"]
    df = df.sort_values(["station_name", "date_of_record"])
    df[weather_cols] = df.groupby("station_name")[weather_cols].transform(
        lambda x: x.interpolate(method="linear").ffill().bfill()
    )
    print(f"   Shape after interpolation            : {df.shape}")

    # ── Confirm no remaining NaNs in key columns ───────────────────────────────
    still_missing = df[weather_cols + ["rainfall"]].isnull().sum()
    if still_missing.any():
        print("\n⚠  Still missing after interpolation:")
        print(still_missing[still_missing > 0])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived atmospheric / dynamics / lag / seasonality features."""
    print("\n⚙️  Engineering features...")

    # ── Thermodynamics ─────────────────────────────────────────────────────────
    R = 287                                         # specific gas constant (J/kg·K)
    df["temp_k"]      = df["avg_temp"] + 273.15
    df["pressure_pa"] = df["air_pressure"] * 100
    df["air_density"] = df["pressure_pa"] / (R * df["temp_k"])

    # ── Wind decomposition ─────────────────────────────────────────────────────
    df["lat_rad"]    = np.radians(df["latitude"])
    df["u_velocity"] = df["wind_speed"] * np.cos(df["lat_rad"])
    df["v_velocity"] = df["wind_speed"] * np.sin(df["lat_rad"])

    # ── Pressure gradient (per-station temporal diff) ──────────────────────────
    df["pressure_gradient"] = (
        df.groupby("station_name")["air_pressure"].diff().fillna(0)
    )

    # ── Velocity gradients, divergence, vorticity ──────────────────────────────
    df["du_dx"]      = df.groupby("station_name")["u_velocity"].diff().fillna(0)
    df["dv_dy"]      = df.groupby("station_name")["v_velocity"].diff().fillna(0)
    df["divergence"] = df["du_dx"] + df["dv_dy"]
    df["dv_dx"]      = df.groupby("station_name")["v_velocity"].diff().fillna(0)
    df["du_dy"]      = df.groupby("station_name")["u_velocity"].diff().fillna(0)
    df["vorticity"]  = df["dv_dx"] - df["du_dy"]

    # ── Coriolis, kinetic energy, temperature gradient ─────────────────────────
    OMEGA = 7.2921e-5                               # Earth's angular velocity (rad/s)
    df["coriolis"]       = 2 * OMEGA * np.sin(df["lat_rad"])
    df["kinetic_energy"] = 0.5 * (df["u_velocity"] ** 2 + df["v_velocity"] ** 2)
    df["temp_gradient"]  = df.groupby("station_name")["avg_temp"].diff().fillna(0)

    # ── Autoregressive rainfall lags ───────────────────────────────────────────
    for lag in [1, 3, 7, 30]:
        df[f"rain_lag_{lag}"] = (
            df.groupby("station_name")["rainfall"]
            .shift(lag)
            .fillna(0)
        )

    # ── Cyclic month encoding ──────────────────────────────────────────────────
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])
    df["month"]          = df["date_of_record"].dt.month
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)

    print(f"   Final shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — MONSOON ZONE ASSIGNMENT  (from zoning_stations.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def assign_monsoon_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each station into SW_MONSOON, NE_MONSOON, or LOW_MONSOON
    based on normalised seasonal rainfall totals.

    Logic
    -----
    1. Sum SW-season (Jun–Sep) and NE-season (Oct–Dec) rainfall per station.
    2. Normalise by number of years.
    3. LOW threshold = 20th percentile of log-total annual rainfall.
    4. If total < threshold → LOW_MONSOON.
       Elif NE ratio ≥ 0.40 → NE_MONSOON.
       Else → SW_MONSOON.
    """
    print("\n🗺️  Assigning monsoon zones...")

    df["month"] = pd.to_datetime(df["date_of_record"]).dt.month
    df["is_sw"] = df["month"].isin([6, 7, 8, 9])
    df["is_ne"] = df["month"].isin([10, 11, 12])

    sw = df[df["is_sw"]].groupby("station_name")["rainfall"].sum()
    ne = df[df["is_ne"]].groupby("station_name")["rainfall"].sum()

    station_df = pd.DataFrame({"sw_rain": sw, "ne_rain": ne}).fillna(0)
    station_df["total_rain"] = station_df["sw_rain"] + station_df["ne_rain"]

    # Normalise to annual average
    num_years = df["date_of_record"].dt.year.nunique()
    station_df[["sw_rain", "ne_rain", "total_rain"]] /= num_years

    # 20th-percentile LOW threshold (log-scale for robustness)
    station_df["log_total"] = np.log1p(station_df["total_rain"])
    threshold = np.expm1(station_df["log_total"].quantile(0.20))
    print(f"   LOW_MONSOON threshold : {threshold:.2f} mm/year")

    def _assign(row):
        if row["total_rain"] < threshold:
            return "LOW_MONSOON"
        ne_ratio = row["ne_rain"] / (row["total_rain"] + 1e-6)
        return "NE_MONSOON" if ne_ratio >= 0.40 else "SW_MONSOON"

    station_df["monsoon_zone"] = station_df.apply(_assign, axis=1)
    station_df = station_df.reset_index()

    # Merge back (drop existing column first to avoid conflicts)
    if "monsoon_zone" in df.columns:
        df = df.drop(columns=["monsoon_zone"])
    df = df.merge(
        station_df[["station_name", "monsoon_zone"]],
        on="station_name",
        how="left",
    )

    # Drop helper boolean columns
    df.drop(columns=["is_sw", "is_ne"], inplace=True, errors="ignore")

    print("\n   Row count per zone:")
    print(df["monsoon_zone"].value_counts().to_string())
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(
    raw_path:       Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    zoned_path:     Path = ZONED_DATA_PATH,
) -> pd.DataFrame:
    """Full pipeline: load → clean → feature engineer → zone → save."""

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {raw_path}\n"
            "Place dataset.csv in the dataset/ folder before running."
        )

    # ── Step 1: clean ──────────────────────────────────────────────────────────
    df = load_and_clean(raw_path)

    # ── Step 2: feature engineering ───────────────────────────────────────────
    df = engineer_features(df)

    # ── Step 3: save processed data ───────────────────────────────────────────
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\n💾 Saved processed data → {processed_path}")

    # ── Step 4: assign monsoon zones ──────────────────────────────────────────
    df = assign_monsoon_zones(df)

    # ── Step 5: save final zoned dataset ──────────────────────────────────────
    df.to_csv(zoned_path, index=False)
    print(f"💾 Saved zoned dataset  → {zoned_path}")

    print("\n✅ Preprocessing & zoning complete.")
    print(f"   Rows : {len(df):,}")
    print(f"   Cols : {df.shape[1]}")
    return df


if __name__ == "__main__":
    run()
