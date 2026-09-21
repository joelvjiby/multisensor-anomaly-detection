import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .data import SENSOR_COLUMNS


class MultisensorAnomalyDetector:
    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )

    @staticmethod
    def add_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        missing = set(SENSOR_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing sensor columns: {sorted(missing)}")
        out = df.copy()
        for col in SENSOR_COLUMNS:
            out[f"{col}_roll_mean"] = out[col].rolling(window, min_periods=1).mean()
            out[f"{col}_roll_std"] = out[col].rolling(window, min_periods=1).std().fillna(0.0)
        return out

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        featured = self.add_features(df)
        feature_cols = [c for c in featured.columns if c != "timestamp_s"]
        X = featured[feature_cols].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(X) != len(featured):
            raise ValueError("Input contains missing or non-finite feature values")
        X_scaled = self.scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        result = featured.copy()
        result["anomaly_score"] = scores
        result["is_anomaly"] = labels == -1
        return result
