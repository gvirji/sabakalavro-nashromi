# ADS-B Trajectory Analytics: Clustering + Anomaly Detection

Academic prototype for a bachelor's thesis on:

**ADS-B მონაცემების გამოყენებით საჰაერო ხომალდების ფრენის ტრაექტორიების ანალიზი და ანომალიების გამოვლენა.**

## What this project does

1. Loads ADS-B point data from CSV/Parquet, FR24 live snapshots, or OpenSky live state vectors.
2. Cleans invalid coordinates and telemetry values.
3. Reconstructs aircraft sessions/trajectories from timestamped observations.
4. Builds trajectory-level features.
5. Searches K-Means cluster count using silhouette score.
6. Detects unusual trajectories with Isolation Forest.
7. Runs a transparent rule-based baseline.
8. Combines the detectors into an anomaly-candidate table.
9. Displays trajectories, clusters, and anomaly candidates in a Streamlit dashboard.
10. Exports processed results to CSV.

## Important research design choice

A single live snapshot is **not** a trajectory. For trajectory-level analysis, collect repeated observations using:

```bash
python collector.py --source fr24 --interval 10 --duration 1800
```

or, for OpenSky:

```bash
python collector.py --source opensky --interval 10 --duration 1800 --rotorcraft-only
```

Then run:

```bash
python main_pipeline.py --input data/raw/adsb_snapshots.csv --rotorcraft-only
```

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## OpenSky authentication

Current OpenSky API access uses OAuth2 client credentials. Set:

```bash
set OPENSKY_CLIENT_ID=...
set OPENSKY_CLIENT_SECRET=...
```

Windows PowerShell:

```powershell
$env:OPENSKY_CLIENT_ID="..."
$env:OPENSKY_CLIENT_SECRET="..."
streamlit run app.py
```

## FR24 note

`FlightRadarAPI` is an unofficial FlightRadar24 SDK. Use it only in accordance with the provider's terms. For academic work, keep the source and licensing/usage conditions documented in the thesis.

## Suggested thesis experiments

### Baseline
Rule-based:
- rapid descent
- low-altitude/high-speed event
- unusually low ground speed

### ML
- K-Means clustering
- Isolation Forest anomaly detection

### Evaluation
Because real anomaly labels may be unavailable, use:
- real ADS-B trajectories as the normal corpus
- synthetic anomaly injection for controlled tests
- sensitivity analysis over model contamination
- cluster-count comparison with silhouette score
- qualitative inspection on the map

## Data schema

Minimum useful columns:

- timestamp
- ICAO24
- Callsign
- Latitude
- Longitude
- Altitude_ft
- Speed_knots
- Vertical_Rate_fpm
- Heading_deg
- On_Ground
- Aircraft_Category (optional but recommended for OpenSky rotorcraft filtering)

## Academic caution

The anomaly detector produces **candidate anomalies**. It is not a certified aviation safety system and should not be described as one.
