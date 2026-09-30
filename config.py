from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    rapid_descent_fpm: float = float(os.getenv("RAPID_DESCENT_FPM", "-3000"))
    low_altitude_ft: float = float(os.getenv("LOW_ALTITUDE_FT", "10000"))
    low_altitude_speed_kts: float = float(os.getenv("LOW_ALT_SPEED_KTS", "250"))
    minimum_trajectory_points: int = int(os.getenv("MIN_TRAJECTORY_POINTS", "8"))
    flight_gap_minutes: int = int(os.getenv("FLIGHT_GAP_MINUTES", "30"))
    anomaly_contamination: float = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
    max_cluster_k: int = int(os.getenv("MAX_CLUSTER_K", "6"))


SETTINGS = Settings()
