from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


def generate_demo_adsb(
    n_flights: int = 120,
    points_per_flight: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic ADS-B-like data for demonstrating the complete pipeline.
    It is intentionally marked as DEMO data and must not be presented as real telemetry.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime.now(timezone.utc) - timedelta(days=1)

    for i in range(n_flights):
        category = 8  # rotorcraft
        icao = f"DEMO{i:06d}"
        callsign = f"HELI{i:03d}"

        lat0 = 41.2 + rng.normal(0, 1.0)
        lon0 = 44.8 + rng.normal(0, 1.5)

        mode = rng.choice(["low", "medium", "high"], p=[0.45, 0.4, 0.15])
        speed_base = {"low": 70, "medium": 105, "high": 145}[mode]
        altitude_base = {"low": 2500, "medium": 6500, "high": 12000}[mode]
        climb_base = rng.normal(0, 250)

        is_anomaly = rng.random() < 0.12
        anomaly_type = rng.choice(["descent", "speed", "position"]) if is_anomaly else None

        flight_start = start + timedelta(minutes=int(i * 7))

        headings = np.cumsum(rng.normal(0, 4, points_per_flight)) + rng.uniform(0, 360)
        speeds = np.clip(rng.normal(speed_base, 10, points_per_flight), 25, 220)
        altitudes = np.clip(
            altitude_base
            + np.cumsum(rng.normal(climb_base / 20, 70, points_per_flight)),
            300,
            18000,
        )
        vertical = rng.normal(climb_base, 120, points_per_flight)

        for j in range(points_per_flight):
            angle = np.radians(headings[j])
            step = 0.015 + 0.002 * speeds[j]
            lat = lat0 + np.cos(angle) * step * j / 100
            lon = lon0 + np.sin(angle) * step * j / 100

            if anomaly_type == "descent" and j == points_per_flight // 2:
                vertical[j] = -4200
                altitudes[j:] -= 2500

            if anomaly_type == "speed" and j == points_per_flight // 2:
                speeds[j] = 285

            if anomaly_type == "position" and j == points_per_flight // 2:
                lat += 1.5
                lon += 1.5

            rows.append({
                "timestamp": flight_start + timedelta(seconds=60 * j),
                "ICAO24": icao,
                "Callsign": callsign,
                "Origin_Country": "DEMO",
                "Origin_Airport": None,
                "Latitude": lat,
                "Longitude": lon,
                "Altitude_ft": float(altitudes[j]),
                "Speed_knots": float(speeds[j]),
                "Vertical_Rate_fpm": float(vertical[j]),
                "Heading_deg": float(headings[j] % 360),
                "On_Ground": 0,
                "Aircraft_Category": category,
                "DEMO_ANOMALY": int(is_anomaly),
                "DEMO_ANOMALY_TYPE": anomaly_type or "none",
            })

    return pd.DataFrame(rows)
