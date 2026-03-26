FEATURES: list[str] = [
    "avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure",
    "air_density", "u_velocity", "v_velocity", "pressure_gradient",
    "divergence", "vorticity", "kinetic_energy", "temp_gradient",
    "rain_lag_1", "rain_lag_3", "rain_lag_7", "rain_lag_30",
    "month_sin", "month_cos"
]

TARGET: str = "rainfall"

BEST_XGB_PARAMS = {
    "n_estimators": 731,
    "max_depth": 11,
    "learning_rate": 0.014498153031985235,
    "subsample": 0.7544991318541121,
    "colsample_bytree": 0.7352517209279018,
    "gamma": 0.8642446639034301,
    "reg_alpha": 3.5856717998369967,
    "reg_lambda": 1.982944548754793,
    "min_child_weight": 10,
    "max_delta_step": 0,
    "colsample_bylevel": 0.5751814333780054,
    "colsample_bynode": 0.6861526870547513,
}