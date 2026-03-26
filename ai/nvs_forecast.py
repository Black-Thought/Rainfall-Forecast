import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from functools import partial, reduce
from typing import Tuple, List, Dict

# ══════════════════════════════════════════════════════════════════
#  PLOT STYLE
# ══════════════════════════════════════════════════════════════════
ACTUAL_COLOR    = '#00d4ff'
PREDICTED_COLOR = '#ff6b9d'
SIGMA           = 2

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'grid.color'       : '#cccccc',
    'figure.facecolor' : '#0f1117',
    'axes.facecolor'   : '#1a1d2e',
    'axes.labelcolor'  : '#e0e0e0',
    'xtick.color'      : '#b0b0b0',
    'ytick.color'      : '#b0b0b0',
    'text.color'       : '#e0e0e0',
    'legend.framealpha': 0.3,
    'legend.facecolor' : '#2a2d3e',
    'legend.edgecolor' : '#555555',
})


# ══════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════
def load_dataset(path: str) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .assign(date_of_record=lambda df: pd.to_datetime(df["date_of_record"]))
        .sort_values(["station_name", "date_of_record"])
        .reset_index(drop=True)
    )


def split_data(df: pd.DataFrame, train_end="2022-12-31", test_start="2023-01-01"):
    train = df[df["date_of_record"] <= train_end].copy()
    test  = df[df["date_of_record"] >= test_start].copy()
    return train, test


def get_station(df: pd.DataFrame, station: str) -> pd.DataFrame:
    return df[df["station_name"] == station].copy()


# ══════════════════════════════════════════════════════════════════
#  FINITE DIFFERENCE OPERATORS
# ══════════════════════════════════════════════════════════════════
def gradient_4th(x: np.ndarray, dx: float = 1.0) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    g[2:-2] = (-x[4:] + 8*x[3:-1] - 8*x[1:-3] + x[:-4]) / (12 * dx)
    g[1]    = (x[2]  - x[0])  / (2 * dx)
    g[-2]   = (x[-1] - x[-3]) / (2 * dx)
    g[0]    = (-3*x[0] + 4*x[1] - x[2])       / (2 * dx)
    g[-1]   = ( 3*x[-1] - 4*x[-2] + x[-3])    / (2 * dx)
    return g


def laplacian_4th(x: np.ndarray, dx: float = 1.0) -> np.ndarray:
    l = np.zeros_like(x, dtype=float)
    l[2:-2] = (-x[4:] + 16*x[3:-1] - 30*x[2:-2] + 16*x[1:-3] - x[:-4]) / (12 * dx**2)
    l[1]    = (x[2]  - 2*x[1]  + x[0])  / dx**2
    l[-2]   = (x[-1] - 2*x[-2] + x[-3]) / dx**2
    return l


# ══════════════════════════════════════════════════════════════════
#  THERMODYNAMICS
# ══════════════════════════════════════════════════════════════════
def saturation_mixing_ratio(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    T_C = T - 273.15
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    return 0.622 * e_s / np.maximum(P - e_s, 1.0)


def condensation_rate(q, T, P, rho, tau=3600):
    q_sat  = saturation_mixing_ratio(T, P)
    excess = np.maximum(q - q_sat, 0.0)
    return rho * excess / tau


def evaporation_rate(R, q, T, P, rho, tau_e=7200):
    q_sat   = saturation_mixing_ratio(T, P)
    deficit = np.maximum(q_sat - q, 0.0)
    return rho * deficit * np.maximum(R, 0.0) / tau_e


# ══════════════════════════════════════════════════════════════════
#  NAVIER-STOKES SYSTEM
# ══════════════════════════════════════════════════════════════════
def navier_stokes_full(t: float, state: np.ndarray, params: dict) -> np.ndarray:
    n   = params["n"]
    dx  = params["dx"]

    u   = state[0*n : 1*n]
    v   = state[1*n : 2*n]
    q   = state[2*n : 3*n]
    R   = state[3*n : 4*n]

    P    = params["P"]
    rho  = params["rho"]
    T    = params["T"]
    lag  = params["lag"]
    nu   = params["nu"]
    Dq   = params["Dq"]
    DR   = params["DR"]
    f    = params["coriolis"]
    beta = params["beta"]
    alpha= params["alpha"]
    gamma= params["gamma"]

    du_dx = gradient_4th(u, dx)
    dv_dy = gradient_4th(v, dx)
    dP_dx = gradient_4th(P, dx)
    dP_dy = gradient_4th(P, dx)
    dq_dx = gradient_4th(q, dx)
    dR_dx = gradient_4th(R, dx)

    lap_u = laplacian_4th(u, dx)
    lap_v = laplacian_4th(v, dx)
    lap_q = laplacian_4th(q, dx)
    lap_R = laplacian_4th(R, dx)

    C = condensation_rate(q, T, P, rho)
    E = evaporation_rate(R, q, T, P, rho)

    du_dt = (- u * du_dx
             - (1.0 / rho) * dP_dx
             + nu   * lap_u
             + f    * v
             - beta * dR_dx)

    dv_dt = (- v * dv_dy
             - (1.0 / rho) * dP_dy
             + nu  * lap_v
             - f   * u)

    dq_dt = (- u * dq_dx
             - v * dq_dx
             + Dq * lap_q
             - C + E)

    dR_dt = (- u * dR_dx
             - v * dR_dx
             + DR  * lap_R
             + C   - E
             - gamma * np.maximum(R, 0.0)
             + alpha * lag)

    return np.concatenate([du_dt, dv_dt, dq_dt, dR_dt])


# ══════════════════════════════════════════════════════════════════
#  PARAMETER ESTIMATION
# ══════════════════════════════════════════════════════════════════
def estimate_params(station_df: pd.DataFrame) -> dict:
    R_std = station_df["rainfall"].std()   + 1e-6
    u_std = station_df["u_velocity"].std() + 1e-6
    L     = len(station_df)

    nu    = float(u_std * np.sqrt(L))
    DR    = float(0.05 * R_std * np.sqrt(L))
    Dq    = float(0.08 * R_std * np.sqrt(L))
    f     = float(station_df["coriolis"].mean()) if "coriolis" in station_df.columns else 1e-4
    beta  = float(0.01 * R_std / (u_std + 1e-6))
    alpha = float(0.02 * R_std / (station_df["rain_lag_30"].std() + 1e-6))
    gamma = 1.0 / 86400.0

    return dict(nu=nu, DR=DR, Dq=Dq, coriolis=f,
                beta=beta, alpha=alpha, gamma=gamma)


def build_initial_state(snapshot: pd.DataFrame) -> np.ndarray:
    u   = snapshot["u_velocity"].values.astype(float)
    v   = snapshot["v_velocity"].values.astype(float)
    rho = snapshot["air_density"].values.astype(float)
    q   = rho / (rho.max() + 1e-9)
    R   = np.maximum(snapshot["rainfall"].values.astype(float), 0.0)
    return np.concatenate([u, v, q, R])


# ══════════════════════════════════════════════════════════════════
#  SPATIAL SNAPSHOT  (all stations on one day, sorted by longitude)
# ══════════════════════════════════════════════════════════════════
def get_spatial_snapshot(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """All stations for a single day sorted west→east by longitude."""
    return (
        df[df["date_of_record"] == date]
        .copy()
        .sort_values("longitude")
        .reset_index(drop=True)
    )


def get_prev_available_date(df: pd.DataFrame, before: pd.Timestamp) -> pd.Timestamp:
    """Return the most recent date in df strictly before `before`."""
    available = df[df["date_of_record"] < before]["date_of_record"].unique()
    if len(available) == 0:
        raise ValueError(f"No data available before {before.date()}")
    return pd.Timestamp(max(available))


# ══════════════════════════════════════════════════════════════════
#  SPATIAL FORECAST
# ══════════════════════════════════════════════════════════════════
def forecast_spatial(
    df:          pd.DataFrame,
    target_date: pd.Timestamp,
    num_days:    int = 1,
) -> pd.Series:
    """
    Use the full spatial transect of all stations on `target_date`
    as the 1D NS domain. Integrate forward `num_days` days.
    Returns a Series of predicted rainfall indexed by station_name.
    """
    snapshot = get_spatial_snapshot(df, target_date)

    if snapshot.empty:
        raise ValueError(f"No spatial data for {target_date.date()}")

    # Real physical dx from longitude spacing
    lons = snapshot["longitude"].values
    dx   = float(np.mean(np.diff(lons)) * 111000.0) if len(lons) > 1 else 100000.0
    dx   = max(dx, 1000.0)

    n      = len(snapshot)
    state  = build_initial_state(snapshot)

    # Use coriolis from data since it's already preprocessed
    params = {
        "n"   : n,
        "dx"  : dx,
        "P"   : snapshot["pressure_pa"].values.astype(float),
        "rho" : snapshot["air_density"].values.astype(float),
        "T"   : snapshot["temp_k"].values.astype(float),
        "lag" : snapshot["rain_lag_30"].values.astype(float),
        **estimate_params(snapshot),
    }

    rhs           = partial(navier_stokes_full, params=params)
    dt            = 1800.0          # 30-min timestep
    steps_per_day = int(86400 / dt) # 48 steps per day

    for day in range(num_days):
        for step in range(steps_per_day):
            t     = (day * steps_per_day + step) * dt
            deriv = rhs(t, state)
            deriv = np.clip(deriv, -1e3, 1e3)
            state = state + dt * deriv
            # Keep rainfall physically bounded
            state[3*n : 4*n] = np.clip(state[3*n : 4*n], 0.0, 500.0)

    R_final = state[3*n : 4*n]
    return pd.Series(R_final, index=snapshot["station_name"].values)


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse    = np.mean((y_true - y_pred) ** 2)
    return dict(mse=mse, rmse=np.sqrt(mse), mae=np.mean(np.abs(y_true - y_pred)))


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    keys = [k for k in metrics_list[0].keys() if k != "station"]
    return reduce(
        lambda acc, m: {k: acc[k] + m[k] / len(metrics_list) for k in keys},
        metrics_list,
        {k: 0.0 for k in keys}
    )


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
def plot_forecast(
    dates:   np.ndarray,
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    station: str,
    metrics: Dict[str, float],
) -> None:
    x_idx         = np.arange(len(y_true))
    y_true_smooth = gaussian_filter1d(y_true, sigma=SIGMA)
    y_pred_smooth = gaussian_filter1d(y_pred, sigma=SIGMA)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14)

    ax.fill_between(x_idx, y_true_smooth, alpha=0.15, color=ACTUAL_COLOR)
    ax.fill_between(x_idx, y_pred_smooth, alpha=0.15, color=PREDICTED_COLOR)

    ax.scatter(x_idx, y_true, color=ACTUAL_COLOR,    alpha=0.25, s=18, zorder=2)
    ax.scatter(x_idx, y_pred, color=PREDICTED_COLOR, alpha=0.25, s=18, zorder=2)

    ax.plot(x_idx, y_true_smooth, color=ACTUAL_COLOR,    lw=2.2,
            label="Actual Rainfall",    zorder=3)
    ax.plot(x_idx, y_pred_smooth, color=PREDICTED_COLOR, lw=2.2,
            label="Predicted Rainfall", zorder=3, ls="--")

    tick_step = max(1, len(x_idx) // 15)
    tick_pos  = x_idx[::tick_step]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"Day {i}" for i in tick_pos],
                       rotation=35, ha="right", fontsize=9)

    ax.set_xlabel("Days from Start", fontsize=11, labelpad=8)
    ax.set_ylabel("Rainfall (mm)", fontsize=11, labelpad=8)
    ax.set_title(
        f"Rainfall Forecast  ·  Navier-Stokes  ·  {station}",
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
#  PREDICT
# ══════════════════════════════════════════════════════════════════
def predict(
    df:             pd.DataFrame,
    station_name:   str,
    date_of_record: str,
    num_days:       int,
    plot:           bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    start_dt  = pd.to_datetime(date_of_record)
    end_dt    = start_dt + pd.Timedelta(days=num_days - 1)

    station_df    = df[df["station_name"] == station_name].copy()
    actual_window = station_df[
        (station_df["date_of_record"] >= start_dt) &
        (station_df["date_of_record"] <= end_dt)
    ].copy().reset_index(drop=True)

    if actual_window.empty:
        raise ValueError(f"No data for '{station_name}' from {start_dt.date()}")

    y_pred = []

    for i in range(len(actual_window)):
        current_date = actual_window["date_of_record"].iloc[i]

        try:
            prev_date = get_prev_available_date(df, current_date)
            preds     = forecast_spatial(df, prev_date, num_days=1)
            val       = preds.get(station_name, float(station_df["rainfall"].mean()))
            y_pred.append(float(val))
        except Exception as e:
            print(f"  ⚠ Day {i} failed: {e}")
            y_pred.append(float(station_df["rainfall"].mean()))

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(actual_window)} days done")

    y_true    = actual_window["rainfall"].values
    y_pred    = np.array(y_pred)
    dates     = actual_window["date_of_record"].values
    metrics   = compute_metrics(y_true, y_pred)

    result_df = pd.DataFrame({
        "date_of_record": dates,
        "actual"        : y_true,
        "predicted"     : y_pred,
    })

    print(f"\n{'─'*52}")
    print(f"  Station  : {station_name}")
    print(f"  Period   : {start_dt.date()} → {end_dt.date()}  ({len(actual_window)} days)")
    print(f"  MSE      : {metrics['mse']:.4f}")
    print(f"  RMSE     : {metrics['rmse']:.4f}")
    print(f"  MAE      : {metrics['mae']:.4f}")
    print(f"{'─'*52}")

    if plot:
        plot_forecast(dates, y_true, y_pred, station_name, metrics)

    return result_df, metrics


# ══════════════════════════════════════════════════════════════════
#  MULTI-STATION EVALUATION
# ══════════════════════════════════════════════════════════════════
def evaluate_stations(
    df:       pd.DataFrame,
    stations: List[str],
    start:    str = "2023-01-01",
    end:      str = "2025-12-31",
) -> pd.DataFrame:
    num_days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1

    rows = []
    for station in stations:
        try:
            _, m = predict(df, station, start, num_days, plot=True)
            rows.append({"station": station, **m})
        except Exception as e:
            print(f"  ⚠  Skipping {station}: {e}")

    avg = aggregate_metrics([{k: v for k, v in r.items() if k != "station"} for r in rows])

    print(f"\n{'═'*52}")
    print("  SUMMARY  —  Average Metrics (Navier-Stokes)")
    print(f"{'═'*52}")
    print(f"  MSE  : {avg['mse']:.4f}")
    print(f"  RMSE : {avg['rmse']:.4f}")
    print(f"  MAE  : {avg['mae']:.4f}")
    print(f"{'═'*52}\n")

    return pd.DataFrame(rows).set_index("station")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    df = load_dataset("processed_weather_data.csv")

    stations = df["station_name"].unique()[:4].tolist()

    # Test with 10 days first to confirm it works
    summary = evaluate_stations(df, stations, start="2023-01-01", end="2025-02-10")

    # Once confirmed, expand to full range
    # summary = evaluate_stations(df, stations, start="2023-01-01", end="2025-02-10")

    # Ad-hoc single prediction
    result, metrics = predict(
        df=df,
        station_name=stations[0],
        date_of_record="2023-06-01",
        num_days=30
    )
    print(result.head(10))