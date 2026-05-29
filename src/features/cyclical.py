"""
Cyclical Time Encoding — sin/cos encoding for temporal features.

Cyclical encoding preserves the continuity of time:
  - hour 23 is close to hour 0 (sin/cos captures this)
  - One-hot or linear encoding would treat them as distant

This is particularly important for Kill Zones that span midnight (Asian KZ).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CYCLICAL_COLUMNS = ["hour_sin", "hour_cos"]
DAY_COLUMNS = ["is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday", "is_sunday"]


def compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Contract Violation: Timestamp column not found.")
        
    if df['timestamp'].dt.tz is None or df['timestamp'].dt.tz.zone != 'America/New_York':
        raise ValueError("Contract Violation: Timestamp column is not timezone-aware.")
        
    hour = df['timestamp'].dt.hour
    minute = df['timestamp'].dt.minute

    continuous_hour = hour + (minute / 60.0)
    df["hour_sin"] = np.sin(2 * np.pi * continuous_hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * continuous_hour / 24).astype(np.float32)

    # ── One-Hot Encoding pe ziua săptămânii ──────────────────
    day_of_week = df['timestamp'].dt.dayofweek
    df["is_monday"] = (day_of_week == 0).astype(np.float32).values
    df["is_tuesday"] = (day_of_week == 1).astype(np.float32).values
    df["is_wednesday"] = (day_of_week == 2).astype(np.float32).values
    df["is_thursday"] = (day_of_week == 3).astype(np.float32).values
    df["is_friday"] = (day_of_week == 4).astype(np.float32).values
    df["is_sunday"] = (day_of_week == 6).astype(np.float32).values

    logger.info(f"  Cyclical features computed: {CYCLICAL_COLUMNS}")
    logger.info(f"  Day features computed: {DAY_COLUMNS}")
    return df
