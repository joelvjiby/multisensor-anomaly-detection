import pandas as pd
import pytest

from src.anomaly_detection.data import generate_sensor_data
from src.anomaly_detection.detector import MultisensorAnomalyDetector


def test_generated_data_has_expected_columns():
    df = generate_sensor_data(100)
    assert len(df) == 100
    assert "radar_range_m" in df.columns


def test_feature_generation():
    df = generate_sensor_data(20)
    result = MultisensorAnomalyDetector.add_features(df, window=5)
    assert "speed_kmh_roll_mean" in result.columns
    assert len(result) == 20


def test_detector_returns_labels_and_scores():
    df = generate_sensor_data(300)
    result = MultisensorAnomalyDetector(contamination=0.05).fit_predict(df)
    assert "is_anomaly" in result.columns
    assert "anomaly_score" in result.columns
    assert result["is_anomaly"].dtype == bool


def test_missing_sensor_column():
    df = pd.DataFrame({"speed_kmh": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing sensor columns"):
        MultisensorAnomalyDetector.add_features(df)
