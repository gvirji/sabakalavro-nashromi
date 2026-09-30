from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from data_sources import fetch_fr24_snapshot, fetch_opensky_snapshot, load_table
from pipeline import analyze_trajectories
from sample_data import generate_demo_adsb
from visualization import (
    make_anomaly_distribution,
    make_cluster_scatter,
    make_cluster_size_chart,
    make_feature_importance_proxy,
    make_flight_map,
)


st.set_page_config(
    page_title="ADS-B Trajectory Analytics",
    page_icon="✈️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def demo_data(n_flights: int, points_per_flight: int) -> pd.DataFrame:
    return generate_demo_adsb(
        n_flights=n_flights,
        points_per_flight=points_per_flight,
    )


def show_data_quality(df: pd.DataFrame):
    st.subheader("Data quality")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Unique aircraft", f"{df['ICAO24'].nunique():,}" if "ICAO24" in df else "0")
    cols[2].metric("Valid coordinates", f"{df[['Latitude','Longitude']].notna().all(axis=1).sum():,}")
    if "Aircraft_Category" in df and df["Aircraft_Category"].notna().any():
        cols[3].metric("Rotorcraft records", f"{(df['Aircraft_Category'] == 8).sum():,}")
    else:
        cols[3].metric("Rotorcraft records", "N/A")


st.title("ADS-B Flight Trajectory Analytics")
st.caption(
    "Research prototype: trajectory reconstruction → feature engineering → clustering → "
    "Isolation Forest → rule-based baseline → interactive visualization"
)

with st.sidebar:
    st.header("1. Data source")
    source = st.radio(
        "Choose source",
        [
            "Demo data",
            "Upload CSV/Parquet",
            "Live FR24 snapshot",
            "Live OpenSky snapshot",
        ],
    )

    st.header("2. Analysis settings")
    rotorcraft_only = st.checkbox(
        "Rotorcraft only",
        value=True,
        help="Uses ADS-B emitter category 8 when the source provides it.",
    )
    contamination = st.slider(
        "Isolation Forest contamination",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )
    max_k = st.slider(
        "Maximum K for K-Means search",
        min_value=2,
        max_value=8,
        value=6,
        step=1,
    )

    st.divider()
    st.header("Threshold baseline")
    st.write("These are analytical event thresholds, not universal flight regulations.")
    st.write("Rapid descent:", -3000, "fpm")
    st.write("Low altitude:", 10000, "ft")
    st.write("Low-altitude high speed:", 250, "kt")

raw = pd.DataFrame()

try:
    if source == "Demo data":
        raw = demo_data(120, 30)

    elif source == "Upload CSV/Parquet":
        uploaded = st.file_uploader(
            "Upload historical ADS-B data",
            type=["csv", "parquet"],
        )
        if uploaded is None:
            st.info("Upload a timestamped dataset for trajectory-level analysis.")
            st.stop()
        raw = load_table(uploaded)

    elif source == "Live FR24 snapshot":
        st.warning(
            "A single snapshot is state-level data. For trajectory analysis, collect snapshots "
            "repeatedly with collector.py."
        )
        if st.button("Fetch FR24 snapshot", type="primary"):
            raw = fetch_fr24_snapshot()
            st.session_state["live_raw"] = raw
        else:
            raw = st.session_state.get("live_raw", pd.DataFrame())

    elif source == "Live OpenSky snapshot":
        st.warning(
            "A single snapshot is state-level data. For trajectory analysis, collect snapshots "
            "repeatedly with collector.py."
        )
        if st.button("Fetch OpenSky snapshot", type="primary"):
            raw = fetch_opensky_snapshot(rotorcraft_only=rotorcraft_only)
            st.session_state["live_raw"] = raw
        else:
            raw = st.session_state.get("live_raw", pd.DataFrame())

except Exception as exc:
    st.error(f"Data source error: {type(exc).__name__}: {exc}")
    st.stop()

if raw.empty:
    st.stop()

# Demo ground truth is not used by the detector itself.
demo_ground_truth = raw["DEMO_ANOMALY"].copy() if "DEMO_ANOMALY" in raw else None

points, results, cluster_summary, meta = analyze_trajectories(
    raw,
    rotorcraft_only=rotorcraft_only,
    contamination=contamination,
    max_cluster_k=max_k,
)

if results.empty:
    st.error(
        "No usable trajectories were produced. Historical trajectory analysis needs "
        "multiple timestamped points per flight/session."
    )
    st.stop()

# Reconnect optional demo truth at flight level.
if "DEMO_ANOMALY" in raw.columns and "flight_id" in points.columns:
    truth_map = (
        points.groupby("flight_id")["DEMO_ANOMALY"]
        .max()
        .to_dict()
    )
    results["demo_ground_truth"] = results["flight_id"].map(truth_map).fillna(0).astype(int)

show_data_quality(points)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trajectories", f"{len(results):,}")
c2.metric("Anomaly candidates", f"{int(results['final_anomaly'].sum()):,}")
c3.metric("Clusters", f"{results['cluster'].nunique():,}")
c4.metric("Mean anomaly score", f"{results['anomaly_score'].mean():.3f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Map", "🧩 Clustering", "🚨 Anomalies", "📊 Data & Evaluation"]
)

with tab1:
    st.subheader("Anomalous trajectories")
    map_obj = make_flight_map(points, results)
    st_folium(map_obj, width=None, height=650)

with tab2:
    st.subheader("Cluster structure")
    col1, col2 = st.columns(2)

    with col1:
        fig = make_cluster_scatter(results)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_cluster_size_chart(cluster_summary)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster summary")
    st.dataframe(
        cluster_summary.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Silhouette scores")
    scores = meta.get("silhouette_scores", {})
    if scores:
        score_df = pd.DataFrame(
            [{"K": int(k), "Silhouette": float(v)} for k, v in scores.items()]
        ).sort_values("K")
        st.line_chart(score_df.set_index("K"))
    else:
        st.info("Not enough trajectories to calculate multiple K candidates.")

with tab3:
    st.subheader("Top anomaly candidates")
    anomaly_table = (
        results[results["final_anomaly"] == 1]
        .sort_values("anomaly_score", ascending=False)
    )

    display_cols = [
        "Callsign",
        "ICAO24",
        "cluster",
        "anomaly_score",
        "detection_source",
        "rule_reasons",
        "mean_altitude_ft",
        "max_speed_kts",
        "max_descent_fpm",
        "distance_km",
    ]

    st.dataframe(
        anomaly_table[display_cols].round(3),
        use_container_width=True,
        hide_index=True,
    )

    fig = make_anomaly_distribution(results)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Feature space")
    feature_cols = meta.get("feature_columns", [])

    fig = make_feature_importance_proxy(results, feature_cols)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Processed trajectory records")
    st.dataframe(
        points.head(500),
        use_container_width=True,
        hide_index=True,
    )

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analysis results",
        data=csv,
        file_name="flight_anomaly_results.csv",
        mime="text/csv",
    )

    if "demo_ground_truth" in results.columns:
        st.subheader("Demo-only evaluation")
        tp = int(((results["demo_ground_truth"] == 1) & (results["final_anomaly"] == 1)).sum())
        fp = int(((results["demo_ground_truth"] == 0) & (results["final_anomaly"] == 1)).sum())
        fn = int(((results["demo_ground_truth"] == 1) & (results["final_anomaly"] == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        e1, e2, e3 = st.columns(3)
        e1.metric("Precision", f"{precision:.3f}")
        e2.metric("Recall", f"{recall:.3f}")
        e3.metric("F1", f"{f1:.3f}")

        st.caption(
            "These metrics are valid only for the included synthetic DEMO dataset. "
            "They must not be reported as real aviation performance."
        )

st.divider()
st.caption(
    "Important: this software is an academic research prototype. Anomaly flags are "
    "statistical/analytical candidates and are not flight-safety determinations."
)
