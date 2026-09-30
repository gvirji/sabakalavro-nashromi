from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import requests


FR24_COLUMNS = [
    "timestamp", "ICAO24", "Callsign", "Origin_Country", "Origin_Airport",
    "Latitude", "Longitude", "Altitude_ft", "Speed_knots", "Vertical_Rate_fpm",
    "Heading_deg", "On_Ground", "Aircraft_Category"
]


def _utc_now():
    return datetime.now(timezone.utc)


def fetch_fr24_snapshot() -> pd.DataFrame:
    """
    Educational/live snapshot adapter for the unofficial FlightRadarAPI package.
    A snapshot contains one row per currently returned flight.
    For trajectory analysis, collect snapshots repeatedly and append them to a CSV.
    """
    from FlightRadarAPI import FlightRadar24API

    api = FlightRadar24API()
    flights = api.get_flights()
    now = _utc_now().isoformat()

    rows = []
    for f in flights:
        rows.append({
            "timestamp": now,
            "ICAO24": getattr(f, "icao_24bit", None),
            "Callsign": getattr(f, "callsign", None),
            "Origin_Country": getattr(f, "origin_airport_iata", None),
            "Origin_Airport": getattr(f, "origin_airport_iata", None),
            "Latitude": getattr(f, "latitude", None),
            "Longitude": getattr(f, "longitude", None),
            "Altitude_ft": getattr(f, "altitude", None),
            "Speed_knots": getattr(f, "ground_speed", None),
            "Vertical_Rate_fpm": getattr(f, "vertical_speed", None),
            "Heading_deg": getattr(f, "heading", None),
            "On_Ground": 1 if getattr(f, "on_ground", 0) == 1 else 0,
            "Aircraft_Category": None,
        })

    return pd.DataFrame(rows, columns=FR24_COLUMNS)


def _opensky_token(client_id: str, client_secret: str) -> str:
    token_url = (
        "https://auth.opensky-network.org/auth/realms/"
        "opensky-network/protocol/openid-connect/token"
    )
    response = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def fetch_opensky_snapshot(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    rotorcraft_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch live OpenSky state vectors.
    If credentials are missing, anonymous access is attempted with reduced limits.
    OpenSky category 8 is Rotorcraft.
    """
    client_id = client_id or os.getenv("OPENSKY_CLIENT_ID")
    client_secret = client_secret or os.getenv("OPENSKY_CLIENT_SECRET")

    headers = {}
    if client_id and client_secret:
        headers["Authorization"] = f"Bearer {_opensky_token(client_id, client_secret)}"

    response = requests.get(
        "https://opensky-network.org/api/states/all",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    states = payload.get("states", [])
    rows = []

    for s in states:
        if len(s) < 17:
            continue

        category = s[17] if len(s) > 17 else None
        if rotorcraft_only and category != 8:
            continue

        rows.append({
            "timestamp": datetime.fromtimestamp(
                payload.get("time", int(_utc_now().timestamp())),
                tz=timezone.utc
            ).isoformat(),
            "ICAO24": s[0],
            "Callsign": str(s[1]).strip() if s[1] else None,
            "Origin_Country": s[2],
            "Origin_Airport": None,
            "Latitude": s[6],
            "Longitude": s[5],
            "Altitude_ft": (s[7] * 3.280839895) if s[7] is not None else None,
            "Speed_knots": (s[9] * 1.943844492) if s[9] is not None else None,
            "Vertical_Rate_fpm": (s[11] * 196.8503937) if s[11] is not None else None,
            "Heading_deg": s[10],
            "On_Ground": 1 if bool(s[8]) else 0,
            "Aircraft_Category": category,
        })

    return pd.DataFrame(rows, columns=FR24_COLUMNS)


def append_snapshot(df: pd.DataFrame, path: str = "data/raw/adsb_snapshots.csv") -> str:
    if df.empty:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)

    df.to_csv(path, mode="a", header=not exists, index=False)
    return path


def load_table(uploaded_file_or_path) -> pd.DataFrame:
    """
    Accept a Streamlit uploaded file, CSV/Parquet path, or pandas-readable file-like object.
    """
    if uploaded_file_or_path is None:
        return pd.DataFrame()

    name = getattr(uploaded_file_or_path, "name", str(uploaded_file_or_path)).lower()

    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file_or_path)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file_or_path)

    raise ValueError("Unsupported file type. Use CSV or Parquet.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common ADS-B column spellings to the internal schema.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    aliases = {
        "icao24": "ICAO24",
        "icao_24bit": "ICAO24",
        "callsign": "Callsign",
        "lat": "Latitude",
        "latitude": "Latitude",
        "lon": "Longitude",
        "lng": "Longitude",
        "longitude": "Longitude",
        "altitude": "Altitude_ft",
        "baro_altitude": "Altitude_ft",
        "ground_speed": "Speed_knots",
        "velocity": "Speed_knots",
        "vertical_speed": "Vertical_Rate_fpm",
        "vertical_rate": "Vertical_Rate_fpm",
        "heading": "Heading_deg",
        "true_track": "Heading_deg",
        "on_ground": "On_Ground",
        "time": "timestamp",
        "timestamp_utc": "timestamp",
        "category": "Aircraft_Category",
        "origin_country": "Origin_Country",
    }

    rename_map = {}
    for c in out.columns:
        key = str(c).strip()
        low = key.lower()
        if low in aliases:
            rename_map[c] = aliases[low]

    out = out.rename(columns=rename_map)
    return out
