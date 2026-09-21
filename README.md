# Multisensor Anomaly Detection with Python

A portfolio project for detecting unusual patterns in synchronized sensor streams using a data-driven Isolation Forest approach.

## What it demonstrates
- multivariate time-series preprocessing
- sensor synchronization and rolling features
- robust scaling
- unsupervised anomaly detection with Isolation Forest
- anomaly scoring and thresholding
- CSV report generation
- synthetic sensor-data generation for reproducibility
- pytest and GitHub Actions CI

> Portfolio note: the sensor data is synthetic. The project demonstrates the algorithmic workflow and software structure rather than claiming performance on real vehicle data.

## Sensor channels
The example dataset contains:
- vehicle speed
- longitudinal acceleration
- lateral acceleration
- yaw rate
- steering angle
- radar object range

Anomalies are injected into selected intervals to make the demo reproducible.

## Structure
```text
multisensor-anomaly-detection/
├── data/
├── src/anomaly_detection/
│   ├── __init__.py
│   ├── data.py
│   ├── detector.py
│   └── cli.py
├── tests/test_detector.py
├── .github/workflows/ci.yml
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Generate data and detect anomalies
```bash
python -m src.anomaly_detection.cli --generate-data data/sensor_data.csv --output data/anomaly_results.csv
```

Or run detection on an existing compatible CSV:
```bash
python -m src.anomaly_detection.cli --input data/sensor_data.csv --output data/anomaly_results.csv
```

## Method
1. Clean and sort the synchronized sensor stream.
2. Create rolling mean/std features.
3. Robust-scale the feature matrix.
4. Fit an Isolation Forest.
5. Convert model scores to anomaly labels.
6. Export a row-level report.

A production version should additionally address sensor dropout, calibration, time synchronization uncertainty, operating-domain segmentation, ground-truth labeling and false-alarm analysis.
