from __future__ import annotations

import math
import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _safe_std(series):
    return float(series.std(ddof=0)) if len(series) else 0.0


def _safe_mean(series):
    return float(series.mean()) if len(series) else 0.0


def trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert point-level trajectory records to one feature row per flight/session.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    if "flight_id" not in work:
        work["flight_id"] = work["ICAO24"].astype(str)

    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
        work = work.sort_values(["flight_id", "timestamp"], kind="stable")

    rows = []

    for flight_id, g in work.groupby("flight_id", sort=False):
        g = g.dropna(subset=["Latitude", "Longitude", "Altitude_ft", "Speed_knots"]).copy()
        if g.empty:
            continue

        lat = g["Latitude"].to_numpy(dtype=float)
        lon = g["Longitude"].to_numpy(dtype=float)

        if len(g) > 1:
            segment_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            total_distance = float(np.nansum(segment_km))
            endpoint_distance = float(haversine_km(lat[0], lon[0], lat[-1], lon[-1]))
            max_segment = float(np.nanmax(segment_km))
        else:
            total_distance = endpoint_distance = max_segment = 0.0

        duration_minutes = 0.0
        if "timestamp" in g.columns and g["timestamp"].notna().sum() >= 2:
            duration_minutes = float(
                (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 60
            )

        altitude = g["Altitude_ft"].astype(float)
        speed = g["Speed_knots"].astype(float)
        vr = g["Vertical_Rate_fpm"].astype(float)

        altitude_change = float(altitude.iloc[-1] - altitude.iloc[0])
        speed_change = float(speed.iloc[-1] - speed.iloc[0])

        turn_mean = 0.0
        turn_max = 0.0
        if "Heading_Change_deg" in g:
            turn_values = g["Heading_Change_deg"].dropna()
            if not turn_values.empty:
                turn_mean = float(turn_values.mean())
                turn_max = float(turn_values.max())

        if "dt_seconds" in g:
            valid_dt = g["dt_seconds"].where(g["dt_seconds"] > 0)
            speed_delta = g["Speed_Change_kts"].abs() if "Speed_Change_kts" in g else pd.Series(dtype=float)
            altitude_delta = g["Altitude_Change_ft"].abs() if "Altitude_Change_ft" in g else pd.Series(dtype=float)
        else:
            valid_dt = pd.Series(dtype=float)
            speed_delta = pd.Series(dtype=float)
            altitude_delta = pd.Series(dtype=float)

        rows.append({
            "flight_id": flight_id,
            "ICAO24": g["ICAO24"].iloc[0],
            "Callsign": g["Callsign"].iloc[0],
            "Origin_Country": g["Origin_Country"].iloc[0] if "Origin_Country" in g else None,
            "Aircraft_Category": g["Aircraft_Category"].iloc[0] if "Aircraft_Category" in g else None,
            "points": len(g),
            "duration_min": duration_minutes,
            "distance_km": total_distance,
            "endpoint_distance_km": endpoint_distance,
            "trajectory_straightness": (endpoint_distance / total_distance) if total_distance > 0 else 1.0,
            "max_segment_km": max_segment,
            "mean_altitude_ft": _safe_mean(altitude),
            "max_altitude_ft": float(altitude.max()),
            "min_altitude_ft": float(altitude.min()),
            "altitude_std_ft": _safe_std(altitude),
            "altitude_change_ft": altitude_change,
            "max_altitude_range_ft": float(altitude.max() - altitude.min()),
            "mean_speed_kts": _safe_mean(speed),
            "max_speed_kts": float(speed.max()),
            "min_speed_kts": float(speed.min()),
            "speed_std_kts": _safe_std(speed),
            "speed_change_kts": speed_change,
            "mean_vertical_rate_fpm": _safe_mean(vr),
            "max_climb_fpm": float(vr.max()),
            "max_descent_fpm": float(vr.min()),
            "vertical_rate_std_fpm": _safe_std(vr),
            "mean_heading_change_deg": turn_mean,
            "max_heading_change_deg": turn_max,
            "low_altitude_ratio": float((altitude < 10000).mean()),
            "high_speed_ratio": float((speed > 250).mean()),
            "rapid_descent_ratio": float((vr < -3000).mean()),
            "mean_speed_delta_kts": float(speed_delta.mean()) if len(speed_delta.dropna()) else 0.0,
            "mean_altitude_delta_ft": float(altitude_delta.mean()) if len(altitude_delta.dropna()) else 0.0,
        })

    return pd.DataFrame(rows)


STATE_FEATURES = [
    "Altitude_ft",
    "Speed_knots",
    "Vertical_Rate_fpm",
]


TRAJECTORY_FEATURES = [
    "points",
    "duration_min",
    "distance_km",
    "trajectory_straightness",
    "max_segment_km",
    "mean_altitude_ft",
    "max_altitude_ft",
    "min_altitude_ft",
    "altitude_std_ft",
    "altitude_change_ft",
    "max_altitude_range_ft",
    "mean_speed_kts",
    "max_speed_kts",
    "min_speed_kts",
    "speed_std_kts",
    "speed_change_kts",
    "mean_vertical_rate_fpm",
    "max_climb_fpm",
    "max_descent_fpm",
    "vertical_rate_std_fpm",
    "mean_heading_change_deg",
    "max_heading_change_deg",
    "low_altitude_ratio",
    "high_speed_ratio",
    "rapid_descent_ratio",
    "mean_speed_delta_kts",
    "mean_altitude_delta_ft",
]
