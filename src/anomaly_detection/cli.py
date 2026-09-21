import argparse
from pathlib import Path
import pandas as pd

from .data import save_generated_data
from .detector import MultisensorAnomalyDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect anomalies in multisensor time-series data.")
    parser.add_argument("--input", help="Existing sensor CSV")
    parser.add_argument("--generate-data", help="Generate a synthetic sensor CSV at this path")
    parser.add_argument("--output", required=True, help="Output anomaly CSV")
    parser.add_argument("--contamination", type=float, default=0.03)
    args = parser.parse_args()

    if args.generate_data:
        df = save_generated_data(args.generate_data)
    elif args.input:
        df = pd.read_csv(args.input)
    else:
        parser.error("Provide either --input or --generate-data")

    detector = MultisensorAnomalyDetector(contamination=args.contamination)
    result = detector.fit_predict(df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"Rows processed: {len(result)}")
    print(f"Anomalies detected: {int(result['is_anomaly'].sum())}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
