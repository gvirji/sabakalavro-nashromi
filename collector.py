from __future__ import annotations

import argparse
import time

from data_sources import append_snapshot, fetch_fr24_snapshot, fetch_opensky_snapshot


def main():
    parser = argparse.ArgumentParser(
        description="Collect repeated ADS-B snapshots for trajectory reconstruction."
    )
    parser.add_argument(
        "--source",
        choices=["fr24", "opensky"],
        default="fr24",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Seconds between snapshots.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Collection duration in seconds.",
    )
    parser.add_argument(
        "--rotorcraft-only",
        action="store_true",
        help="For OpenSky, keep category 8 rotorcraft only.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/adsb_snapshots.csv",
    )
    args = parser.parse_args()

    if args.source == "fr24":
        print("Using FlightRadarAPI live snapshots.")
    else:
        print("Using OpenSky live state vectors.")

    start = time.time()
    total = 0

    while time.time() - start < args.duration:
        try:
            if args.source == "fr24":
                df = fetch_fr24_snapshot()
            else:
                df = fetch_opensky_snapshot(rotorcraft_only=args.rotorcraft_only)

            if not df.empty:
                append_snapshot(df, args.output)
                total += len(df)
                print(
                    f"[OK] rows={len(df):5d} | "
                    f"appended={total:7d} | output={args.output}"
                )
            else:
                print("[WARN] Empty snapshot.")

        except Exception as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}")

        time.sleep(max(1, args.interval))

    print(f"Done. Collected approximately {total} rows.")


if __name__ == "__main__":
    main()
