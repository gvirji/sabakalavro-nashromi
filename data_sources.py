import pandas as pd
import requests

def fetch_fr24_snapshot():
    # Placeholder for Flightradar24 snapshot fetching logic
    return pd.DataFrame()

def fetch_opensky_snapshot():
    # Placeholder for OpenSky Network snapshot fetching logic
    return pd.DataFrame()

def load_table(uploaded_file):
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    # სვეტების დასახელებების ავტომატური გადარქმევა
    column_mapping = {
        'lat': 'Latitude',
        'long': 'Longitude',
        'tail_number': 'ICAO24',
        'flight': 'Callsign',
        'alt': 'Altitude',
        'spotted': 'timestamp'
    }
    df = df.rename(columns=column_mapping)
    
    # თარიღის ფორმატის სწორი გარდაქმნა (NaT შეცდომის თავიდან ასაცილებლად)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')

    return df
