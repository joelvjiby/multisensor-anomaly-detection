from pathlib import Path
import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "speed_kmh",
    "long_accel_mps2",
    "lat_accel_mps2",
    "yaw_rate_rads",
    "steering_angle_deg",
    "radar_range_m",
]


def generate_sensor_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * 0.1
    speed = 55 + 8 * np.sin(t / 12) + rng.normal(0, 1.2, n_samples)
    long_acc = np.gradient(speed / 3.6, 0.1) + rng.normal(0, 0.08, n_samples)
    lat_acc = 0.15 * np.sin(t / 4) + rng.normal(0, 0.03, n_samples)
    yaw = 0.04 * np.sin(t / 5) + rng.normal(0, 0.01, n_samples)
    steering = 4 * np.sin(t / 5) + rng.normal(0, 0.7, n_samples)
    radar = 38 + 3 * np.sin(t / 7) + rng.normal(0, 0.8, n_samples)

    # Deliberately injected abnormal operating intervals for demonstration.
    anomaly_idx = np.r_[500:515, 1200:1210, 1700:1715]
    anomaly_idx = anomaly_idx[anomaly_idx < n_samples]
    lat_acc[anomaly_idx] += rng.normal(1.2, 0.2, len(anomaly_idx))
    steering[anomaly_idx] += rng.normal(25, 3, len(anomaly_idx))
    radar[anomaly_idx] -= rng.normal(18, 2, len(anomaly_idx))

    return pd.DataFrame({
        "timestamp_s": t,
        "speed_kmh": speed,
        "long_accel_mps2": long_acc,
        "lat_accel_mps2": lat_acc,
        "yaw_rate_rads": yaw,
        "steering_angle_deg": steering,
        "radar_range_m": radar,
    })


def save_generated_data(path: str | Path, n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    df = generate_sensor_data(n_samples, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
