from __future__ import annotations

import pandas as pd

from config import SETTINGS
from data_sources import normalize_columns
from preprocessing import (
    add_derived_point_features,
    add_flight_ids,
    clean_adsb,
    filter_rotorcraft,
)
from features import TRAJECTORY_FEATURES, trajectory_features
from models import (
    add_rule_based_flags,
    cluster_summary,
    combine_anomaly_results,
    run_isolation_forest,
    run_kmeans,
)


def prepare_points(
    raw: pd.DataFrame,
    rotorcraft_only: bool = False,
) -> pd.DataFrame:
    df = normalize_columns(raw)
    df = clean_adsb(df)
    df = filter_rotorcraft(df, enabled=rotorcraft_only)

    if df.empty:
        return df

    df = add_flight_ids(df, gap_minutes=SETTINGS.flight_gap_minutes)
    df = add_derived_point_features(df)
    return df


def analyze_trajectories(
    raw: pd.DataFrame,
    rotorcraft_only: bool = False,
    contamination: float | None = None,
    max_cluster_k: int | None = None,
):
    points = prepare_points(raw, rotorcraft_only=rotorcraft_only)
    if points.empty:
        return points, pd.DataFrame(), pd.DataFrame(), {}

    flights = trajectory_features(points)

    if flights.empty:
        return points, flights, pd.DataFrame(), {}

    # Remove very short tracks from trajectory-level ML if timestamped.
    if "timestamp" in points.columns and points["timestamp"].notna().any():
        counts = flights["points"]
        flights = flights[counts >= SETTINGS.minimum_trajectory_points].copy()

    if flights.empty:
        return points, flights, pd.DataFrame(), {}

    available = [c for c in TRAJECTORY_FEATURES if c in flights.columns]
    flights = flights.dropna(subset=available).reset_index(drop=True)

    clustered, _, _, silhouette_scores = run_kmeans(
        flights,
        feature_cols=available,
        max_k=max_cluster_k or SETTINGS.max_cluster_k,
    )

    if clustered.empty:
        return points, flights, pd.DataFrame(), {}

    clustered = add_rule_based_flags(
        clustered,
        rapid_descent_fpm=SETTINGS.rapid_descent_fpm,
        low_altitude_ft=SETTINGS.low_altitude_ft,
        low_altitude_speed_kts=SETTINGS.low_altitude_speed_kts,
    )

    # Main ML detector is intentionally separate from clustering.
    detected, _, _ = run_isolation_forest(
        clustered,
        feature_cols=available,
        contamination=contamination or SETTINGS.anomaly_contamination,
    )

    detected = combine_anomaly_results(detected)
    summary = cluster_summary(detected)

    meta = {
        "feature_columns": available,
        "silhouette_scores": silhouette_scores,
        "n_flights": int(len(detected)),
        "n_anomalies": int(detected["final_anomaly"].sum()),
    }

    return points, detected, summary, meta
