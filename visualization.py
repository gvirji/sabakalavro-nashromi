from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import plotly.express as px


def make_flight_map(
    points: pd.DataFrame,
    flight_results: pd.DataFrame,
    max_anomalies: int = 25,
):
    if points.empty:
        return folium.Map(location=[20, 0], zoom_start=2)

    center = [
        float(points["Latitude"].mean()),
        float(points["Longitude"].mean()),
    ]

    m = folium.Map(
        location=center,
        zoom_start=3,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if "flight_id" not in points.columns:
        return m

    result_cols = [
        "flight_id", "Callsign", "final_anomaly",
        "anomaly_score", "cluster", "detection_source", "rule_reasons"
    ]
    result_lookup = (
        flight_results[result_cols]
        .drop_duplicates("flight_id")
        .set_index("flight_id")
        .to_dict("index")
        if not flight_results.empty
        else {}
    )

    anomalies = (
        flight_results[flight_results["final_anomaly"] == 1]
        .sort_values("anomaly_score", ascending=False)
        .head(max_anomalies)
        if not flight_results.empty and "final_anomaly" in flight_results
        else pd.DataFrame()
    )

    selected_ids = set(anomalies["flight_id"].tolist())

    # Draw anomalous trajectories first.
    for flight_id in selected_ids:
        g = points[points["flight_id"] == flight_id].sort_values("timestamp") \
            if "timestamp" in points.columns else points[points["flight_id"] == flight_id]

        if g.empty:
            continue

        meta = result_lookup.get(flight_id, {})
        popup = (
            f"<b>Callsign:</b> {meta.get('Callsign', 'N/A')}<br>"
            f"<b>Cluster:</b> {meta.get('cluster', 'N/A')}<br>"
            f"<b>Anomaly score:</b> {float(meta.get('anomaly_score', 0)):.3f}<br>"
            f"<b>Source:</b> {meta.get('detection_source', 'N/A')}<br>"
            f"<b>Reason:</b> {meta.get('rule_reasons', 'N/A')}"
        )

        coords = g[["Latitude", "Longitude"]].dropna().values.tolist()

        if len(coords) >= 2:
            folium.PolyLine(
                coords,
                weight=5,
                opacity=0.9,
                color="red",
                tooltip=f"ANOMALY {meta.get('Callsign', flight_id)}",
            ).add_to(m)

        folium.CircleMarker(
            location=coords[-1],
            radius=7,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=320),
        ).add_to(m)

    return m


def make_cluster_scatter(df: pd.DataFrame):
    if df.empty:
        return None

    fig = px.scatter(
        df,
        x="mean_altitude_ft",
        y="mean_speed_kts",
        color=df["cluster"].astype(str),
        hover_data=[
            "Callsign",
            "distance_km",
            "anomaly_score",
            "final_anomaly",
        ],
        title="Flight Clusters: Mean Altitude vs Mean Speed",
        labels={"color": "Cluster"},
    )
    return fig


def make_anomaly_distribution(df: pd.DataFrame):
    if df.empty:
        return None

    fig = px.histogram(
        df,
        x="anomaly_score",
        nbins=30,
        title="Anomaly Score Distribution",
        labels={"anomaly_score": "Anomaly score"},
    )
    return fig


def make_cluster_size_chart(summary: pd.DataFrame):
    if summary.empty:
        return None

    fig = px.bar(
        summary,
        x="cluster",
        y="flights",
        title="Flights by Cluster",
        text_auto=True,
    )
    return fig


def make_feature_importance_proxy(df: pd.DataFrame, feature_cols: list[str]):
    """
    A transparent descriptive proxy rather than model-native feature importance:
    absolute correlation between each feature and anomaly score.
    """
    if df.empty or len(df) < 5:
        return None

    records = []
    for col in feature_cols:
        if col not in df:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df["anomaly_score"], errors="coerce")
        valid = x.notna() & y.notna()

        corr = float(x[valid].corr(y[valid])) if valid.sum() >= 3 else 0.0
        records.append({
            "feature": col,
            "abs_correlation_with_anomaly_score": abs(corr),
        })

    out = pd.DataFrame(records).sort_values(
        "abs_correlation_with_anomaly_score",
        ascending=False,
    )

    return px.bar(
        out.head(12),
        x="abs_correlation_with_anomaly_score",
        y="feature",
        orientation="h",
        title="Descriptive Association with Anomaly Score",
        labels={"abs_correlation_with_anomaly_score": "|correlation|"},
    )
