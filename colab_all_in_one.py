# ============================================================
# ADS-B TRAJECTORY ANALYTICS - COLAB VERSION
# Clustering + Isolation Forest + Rule-Based Baseline
# ============================================================

# Colab install cell: run this in a notebook cell before executing the file:
# !pip -q install pandas numpy scikit-learn plotly folium streamlit-folium FlightRadarAPI

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import folium
from IPython.display import display


# -----------------------------
# 1. LOAD DATA
# -----------------------------
# Option A: use your own timestamped ADS-B CSV
# df = pd.read_csv("/content/adsb_snapshots.csv")
#
# Option B: collect a live FR24 snapshot.
from FlightRadarAPI import FlightRadar24API

api = FlightRadar24API()
flights = api.get_flights()

rows = []
now = datetime.now(timezone.utc).isoformat()

for f in flights:
    rows.append({
        "timestamp": now,
        "ICAO24": getattr(f, "icao_24bit", None),
        "Callsign": getattr(f, "callsign", None),
        "Origin_Airport": getattr(f, "origin_airport_iata", None),
        "Latitude": getattr(f, "latitude", None),
        "Longitude": getattr(f, "longitude", None),
        "Altitude_ft": getattr(f, "altitude", None),
        "Speed_knots": getattr(f, "ground_speed", None),
        "Vertical_Rate_fpm": getattr(f, "vertical_speed", None),
        "Heading_deg": getattr(f, "heading", None),
        "On_Ground": 1 if getattr(f, "on_ground", 0) == 1 else 0,
    })

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No flight data received.")

print("Rows received:", len(df))


# -----------------------------
# 2. CLEANING
# -----------------------------
numeric_cols = [
    "Latitude",
    "Longitude",
    "Altitude_ft",
    "Speed_knots",
    "Vertical_Rate_fpm",
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Callsign"] = df["Callsign"].fillna("N/A").astype(str).str.strip()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

df = df.dropna(subset=numeric_cols)
df = df[
    df["Latitude"].between(-90, 90)
    & df["Longitude"].between(-180, 180)
    & df["Speed_knots"].between(0, 700)
    & df["Altitude_ft"].between(-2000, 60000)
]

df = df.sort_values(["ICAO24", "timestamp"])


# -----------------------------
# 3. POINT-LEVEL DERIVED FEATURES
# -----------------------------
df["Heading_Change_deg"] = (
    df.groupby("ICAO24")["Heading_deg"]
      .diff()
      .pipe(lambda s: (s + 180) % 360 - 180)
      .abs()
)

df["dt_seconds"] = (
    df.groupby("ICAO24")["timestamp"]
      .diff()
      .dt.total_seconds()
)

df["Speed_Change_kts"] = df.groupby("ICAO24")["Speed_knots"].diff()
df["Altitude_Change_ft"] = df.groupby("ICAO24")["Altitude_ft"].diff()


# -----------------------------
# 4. HELPER: HAVERSINE
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arctan2(np.sqrt(a), np.sqrt(1-a))


# -----------------------------
# 5. SESSION / TRAJECTORY ID
# -----------------------------
gap = (
    df.groupby("ICAO24")["timestamp"]
      .diff()
      .dt.total_seconds()
      .div(60)
      .fillna(0)
)

new_session = gap > 30
session_no = new_session.groupby(df["ICAO24"]).cumsum()

df["flight_id"] = (
    df["ICAO24"].astype(str) + "_" + session_no.astype(int).astype(str)
)

print("Sessions:", df["flight_id"].nunique())


# -----------------------------
# 6. TRAJECTORY-LEVEL FEATURES
# -----------------------------
feature_rows = []

for flight_id, g in df.groupby("flight_id", sort=False):
    g = g.sort_values("timestamp").copy()

    if len(g) < 2:
        continue

    lat = g["Latitude"].to_numpy()
    lon = g["Longitude"].to_numpy()

    seg = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    total_distance = float(np.nansum(seg))
    endpoint_distance = float(
        haversine_km(lat[0], lon[0], lat[-1], lon[-1])
    )

    duration_min = (
        g["timestamp"].max() - g["timestamp"].min()
    ).total_seconds() / 60

    feature_rows.append({
        "flight_id": flight_id,
        "ICAO24": g["ICAO24"].iloc[0],
        "Callsign": g["Callsign"].iloc[0],
        "points": len(g),
        "duration_min": duration_min,
        "distance_km": total_distance,
        "trajectory_straightness": (
            endpoint_distance / total_distance
            if total_distance > 0 else 1
        ),
        "mean_altitude_ft": g["Altitude_ft"].mean(),
        "max_altitude_ft": g["Altitude_ft"].max(),
        "min_altitude_ft": g["Altitude_ft"].min(),
        "altitude_std_ft": g["Altitude_ft"].std(ddof=0),
        "mean_speed_kts": g["Speed_knots"].mean(),
        "max_speed_kts": g["Speed_knots"].max(),
        "min_speed_kts": g["Speed_knots"].min(),
        "speed_std_kts": g["Speed_knots"].std(ddof=0),
        "mean_vertical_rate_fpm": g["Vertical_Rate_fpm"].mean(),
        "max_climb_fpm": g["Vertical_Rate_fpm"].max(),
        "max_descent_fpm": g["Vertical_Rate_fpm"].min(),
        "vertical_rate_std_fpm": g["Vertical_Rate_fpm"].std(ddof=0),
        "mean_heading_change_deg": g["Heading_Change_deg"].mean(),
        "max_heading_change_deg": g["Heading_Change_deg"].max(),
        "low_altitude_ratio": (g["Altitude_ft"] < 10000).mean(),
        "high_speed_ratio": (g["Speed_knots"] > 250).mean(),
        "rapid_descent_ratio": (g["Vertical_Rate_fpm"] < -3000).mean(),
    })

features = pd.DataFrame(feature_rows)

print("Usable trajectories:", len(features))


# -----------------------------
# 7. CLUSTERING
# -----------------------------
feature_cols = [
    "points",
    "duration_min",
    "distance_km",
    "trajectory_straightness",
    "mean_altitude_ft",
    "max_altitude_ft",
    "min_altitude_ft",
    "altitude_std_ft",
    "mean_speed_kts",
    "max_speed_kts",
    "min_speed_kts",
    "speed_std_kts",
    "mean_vertical_rate_fpm",
    "max_climb_fpm",
    "max_descent_fpm",
    "vertical_rate_std_fpm",
    "mean_heading_change_deg",
    "max_heading_change_deg",
    "low_altitude_ratio",
    "high_speed_ratio",
    "rapid_descent_ratio",
]

features = features.dropna(subset=feature_cols).reset_index(drop=True)

scaler = StandardScaler()
X = scaler.fit_transform(features[feature_cols])

scores = {}

for k in range(2, min(7, len(features)-1)):
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(X)
    if len(set(labels)) >= 2:
        scores[k] = silhouette_score(X, labels)

best_k = max(scores, key=scores.get)

kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
features["cluster"] = kmeans.fit_predict(X)

print("Silhouette scores:", scores)
print("Selected K:", best_k)


# -----------------------------
# 8. ISOLATION FOREST
# -----------------------------
iso = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)

pred = iso.fit_predict(X)
raw_score = iso.score_samples(X)

features["ml_anomaly"] = (pred == -1).astype(int)

mn = raw_score.min()
mx = raw_score.max()

features["anomaly_score"] = (
    (mx - raw_score) / (mx - mn)
    if mx > mn else 0
)


# -----------------------------
# 9. RULE-BASED BASELINE
# -----------------------------
features["rapid_descent_event"] = (
    features["max_descent_fpm"] < -3000
).astype(int)

features["low_alt_high_speed_event"] = (
    (features["min_altitude_ft"] < 10000)
    & (features["max_speed_kts"] > 250)
).astype(int)

features["rule_anomaly"] = (
    (
        features["rapid_descent_event"]
        + features["low_alt_high_speed_event"]
    ) > 0
).astype(int)

features["final_anomaly"] = (
    (features["ml_anomaly"] == 1)
    | (features["rule_anomaly"] == 1)
).astype(int)


# -----------------------------
# 10. RESULTS
# -----------------------------
anomalies = (
    features[features["final_anomaly"] == 1]
    .sort_values("anomaly_score", ascending=False)
)

print("===================================")
print("TRAJECTORY ANALYSIS RESULTS")
print("===================================")
print("Trajectories:", len(features))
print("Clusters:", features["cluster"].nunique())
print("Anomaly candidates:", len(anomalies))
print()

display(
    anomalies[
        [
            "Callsign",
            "ICAO24",
            "cluster",
            "anomaly_score",
            "max_descent_fpm",
            "min_altitude_ft",
            "max_speed_kts",
            "distance_km",
        ]
    ].head(20)
)


# -----------------------------
# 11. MAP
# -----------------------------
m = folium.Map(
    location=[
        float(df["Latitude"].mean()),
        float(df["Longitude"].mean()),
    ],
    zoom_start=3,
    tiles="CartoDB positron",
)

anomaly_ids = set(anomalies.head(20)["flight_id"])

for flight_id in anomaly_ids:
    g = df[df["flight_id"] == flight_id].sort_values("timestamp")
    coords = g[["Latitude", "Longitude"]].dropna().values.tolist()

    if len(coords) < 2:
        continue

    folium.PolyLine(
        coords,
        color="red",
        weight=5,
        tooltip=f"ANOMALY {g['Callsign'].iloc[0]}",
    ).add_to(m)

    folium.Marker(
        coords[-1],
        popup=(
            f"Callsign: {g['Callsign'].iloc[0]}<br>"
            f"ICAO24: {g['ICAO24'].iloc[0]}"
        ),
    ).add_to(m)

display(m)


# -----------------------------
# 12. EXPORT
# -----------------------------
features.to_csv(
    "/content/flight_anomaly_results.csv",
    index=False,
)

print("Saved: /content/flight_anomaly_results.csv")
