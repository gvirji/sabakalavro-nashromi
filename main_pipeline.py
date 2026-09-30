from __future__ import annotations

import argparse
import os

import pandas as pd

from pipeline import analyze_trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Run ADS-B trajectory clustering and anomaly detection."
    )
    parser.add_argument("--input", default="data/raw/adsb_snapshots.csv")
    parser.add_argument("--output", default="data/processed/flight_results.csv")
    parser.add_argument("--rotorcraft-only", action="store_true")
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    points, results, summary, meta = analyze_trajectories(
        raw,
        rotorcraft_only=args.rotorcraft_only,
        contamination=args.contamination,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)

    summary_path = os.path.splitext(args.output)[0] + "_cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Valid points: {len(points):,}")
    print(f"Detected trajectories: {len(results):,}")
    print(f"Anomaly candidates: {int(results['final_anomaly'].sum()) if not results.empty else 0:,}")
    print(f"Saved: {args.output}")
    print(f"Saved: {summary_path}")
    print(f"Silhouette scores: {meta.get('silhouette_scores', {})}")


if __name__ == "__main__":
    main()
