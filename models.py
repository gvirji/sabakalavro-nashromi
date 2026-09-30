from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler


def choose_kmeans_k(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 6,
) -> tuple[int, dict[int, float]]:
    n = len(X)
    if n < 4:
        return 1, {}

    max_k = min(max_k, n - 1)
    scores = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(X, labels))

    if not scores:
        return min(2, n), {}

    best_k = max(scores, key=scores.get)
    return best_k, scores


def run_kmeans(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_k: int = 6,
):
    work = df.copy()
    work = work.dropna(subset=feature_cols).reset_index(drop=True)

    if len(work) < 4:
        work["cluster"] = 0
        return work, None, None, {}

    X_raw = work[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    k, silhouette_scores = choose_kmeans_k(X, max_k=max_k)
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    work["cluster"] = model.fit_predict(X)

    return work, model, scaler, silhouette_scores


def run_dbscan(
    df: pd.DataFrame,
    feature_cols: list[str],
    eps: float = 1.2,
    min_samples: int = 8,
):
    work = df.copy().dropna(subset=feature_cols).reset_index(drop=True)
    if len(work) < max(4, min_samples):
        work["cluster"] = -1
        return work, None, None

    X_raw = work[feature_cols].to_numpy(dtype=float)
    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    work["cluster"] = model.fit_predict(X)
    return work, model, scaler


def run_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
):
    work = df.copy()
    work = work.dropna(subset=feature_cols).reset_index(drop=True)

    if len(work) < 10:
        work["ml_anomaly"] = 0
        work["anomaly_score"] = 0.0
        return work, None, None

    X_raw = work[feature_cols].to_numpy(dtype=float)

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    pred = model.fit_predict(X)
    raw_score = model.score_samples(X)

    work["ml_anomaly"] = (pred == -1).astype(int)
    # Higher score = more anomalous
    raw_min = float(raw_score.min())
    raw_max = float(raw_score.max())
    if raw_max - raw_min > 1e-12:
        normalized = (raw_max - raw_score) / (raw_max - raw_min)
    else:
        normalized = np.zeros_like(raw_score)

    work["anomaly_score"] = normalized
    return work, model, scaler


def add_rule_based_flags(
    df: pd.DataFrame,
    rapid_descent_fpm: float = -3000,
    low_altitude_ft: float = 10000,
    low_altitude_speed_kts: float = 250,
) -> pd.DataFrame:
    work = df.copy()

    rules = {
        "rapid_descent_event": work["max_descent_fpm"] < rapid_descent_fpm,
        "low_alt_high_speed_event": (
            (work["min_altitude_ft"] < low_altitude_ft)
            & (work["max_speed_kts"] > low_altitude_speed_kts)
        ),
        "low_speed_unusual_event": (
            (work["min_altitude_ft"] > 3000)
            & (work["min_speed_kts"] < 60)
            & (work["max_speed_kts"] > 0)
        ),
    }

    for name, mask in rules.items():
        work[name] = mask.astype(int)

    work["rule_anomaly"] = (
        work[list(rules.keys())].sum(axis=1) > 0
    ).astype(int)

    reasons = []
    for _, row in work.iterrows():
        r = []
        if row["rapid_descent_event"]:
            r.append("rapid descent")
        if row["low_alt_high_speed_event"]:
            r.append("low-altitude/high-speed event")
        if row["low_speed_unusual_event"]:
            r.append("unusually low ground speed")
        reasons.append(", ".join(r) if r else "none")

    work["rule_reasons"] = reasons
    return work


def combine_anomaly_results(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["final_anomaly"] = (
        (work["ml_anomaly"] == 1) | (work["rule_anomaly"] == 1)
    ).astype(int)

    labels = []
    for _, row in work.iterrows():
        if row["ml_anomaly"] and row["rule_anomaly"]:
            labels.append("ML + Rule")
        elif row["ml_anomaly"]:
            labels.append("ML")
        elif row["rule_anomaly"]:
            labels.append("Rule")
        else:
            labels.append("Normal")

    work["detection_source"] = labels
    return work


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cluster" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("cluster", dropna=False)
        .agg(
            flights=("flight_id", "count"),
            mean_altitude_ft=("mean_altitude_ft", "mean"),
            mean_speed_kts=("mean_speed_kts", "mean"),
            mean_distance_km=("distance_km", "mean"),
            anomaly_rate=("final_anomaly", "mean") if "final_anomaly" in df.columns else ("flight_id", "count"),
        )
        .reset_index()
    )

    if "anomaly_rate" in summary:
        summary["anomaly_rate"] *= 100

    return summary.sort_values("flights", ascending=False)
