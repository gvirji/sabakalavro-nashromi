from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_NUMERIC = [
    "Latitude",
    "Longitude",
    "Altitude_ft",
    "Speed_knots",
    "Vertical_Rate_fpm",
]


def clean_adsb(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if out.empty:
        return out

    if "Callsign" not in out:
        out["Callsign"] = "N/A"
    if "On_Ground" not in out:
        out["On_Ground"] = 0
    if "Heading_deg" not in out:
        out["Heading_deg"] = np.nan
    if "Aircraft_Category" not in out:
        out["Aircraft_Category"] = np.nan

    for c in REQUIRED_NUMERIC:
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["On_Ground"] = pd.to_numeric(out["On_Ground"], errors="coerce").fillna(0).astype(int)
    out["Callsign"] = out["Callsign"].fillna("N/A").astype(str).str.strip()

    if "timestamp" in out:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    out = out.dropna(subset=["Latitude", "Longitude", "Altitude_ft", "Speed_knots"])
    out = out[
        out["Latitude"].between(-90, 90)
        & out["Longitude"].between(-180, 180)
    ]

    # Conservative sensor-quality bounds. These are data-cleaning limits,
    # not claims about aviation regulations.
    out = out[out["Speed_knots"].between(0, 700)]
    out = out[out["Altitude_ft"].between(-2000, 60000)]
    out = out[out["Vertical_Rate_fpm"].abs().fillna(0).lt(10000)]

    if "timestamp" in out and out["timestamp"].notna().any():
        out = out.sort_values(["ICAO24", "timestamp"], kind="stable")

    return out.reset_index(drop=True)


def filter_rotorcraft(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    if not enabled or "Aircraft_Category" not in df.columns:
        return df.copy()

    if df["Aircraft_Category"].notna().any():
        return df[df["Aircraft_Category"] == 8].copy()

    # If the source doesn't provide aircraft category, don't silently guess.
    return df.copy()


def add_flight_ids(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """
    Create a flight/session identifier using ICAO24 and timestamp gaps.
    A new session starts when:
      - ICAO24 changes
      - time gap exceeds gap_minutes
      - an airborne transition follows a ground state
    """
    out = df.copy()
    if out.empty:
        return out

    if "timestamp" not in out.columns or out["timestamp"].isna().all():
        out["flight_id"] = out["ICAO24"].astype(str)
        return out

    out = out.sort_values(["ICAO24", "timestamp"], kind="stable").reset_index(drop=True)
    gap = out.groupby("ICAO24")["timestamp"].diff().dt.total_seconds().div(60).fillna(0)

    previous_ground = out.groupby("ICAO24")["On_Ground"].shift(1).fillna(1)
    airborne_start = (previous_ground == 1) & (out["On_Ground"] == 0)

    new_session = (gap > gap_minutes) | airborne_start
    session_number = new_session.groupby(out["ICAO24"]).cumsum()

    out["flight_id"] = (
        out["ICAO24"].astype(str) + "_" + session_number.astype(int).astype(str)
    )
    return out


def add_derived_point_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out = out.sort_values(["ICAO24", "flight_id", "timestamp"], kind="stable") \
             if "timestamp" in out.columns else out

    if "Heading_deg" in out.columns:
        heading_diff = (
            out.groupby("flight_id")["Heading_deg"]
            .diff()
            .pipe(lambda s: (s + 180) % 360 - 180)
        )
        out["Heading_Change_deg"] = heading_diff.abs()

    if "timestamp" in out.columns:
        dt = out.groupby("flight_id")["timestamp"].diff().dt.total_seconds()
        out["dt_seconds"] = dt

        speed_delta = out.groupby("flight_id")["Speed_knots"].diff()
        out["Speed_Change_kts"] = speed_delta

        altitude_delta = out.groupby("flight_id")["Altitude_ft"].diff()
        out["Altitude_Change_ft"] = altitude_delta

    return out
