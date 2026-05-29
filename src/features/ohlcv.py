"""
OHLCV Feature Engineering — Price-level invariant features.

These features are normalized relative to the close price of each candle,
making them invariant to the absolute price level. This is critical for
non-stationary financial time series where the mean shifts over time.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_ohlcv_log_features(df: pd.DataFrame, volume_sma_window: int = 60) -> pd.DataFrame:

    df = df.copy()

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).astype(np.float32)
    df["log_open"]   = np.log(df["open"]  / df["close"]).astype(np.float32)
    df["log_high"]   = np.log(df["high"]  / df["close"]).astype(np.float32)
    df["log_low"]    = np.log(df["low"]   / df["close"]).astype(np.float32)
    df["log_range"]  = np.log(df["high"]  / df["low"]).astype(np.float32)

    #min_periods = 1 => for the first 60 values it will compute the mean of the existing values
    #eg for the first 59 values it will compute the mean of the first 59 values
    # vol_ma = df["volume"].rolling(volume_sma_window, min_periods=1).mean()

    #.rolling() - groups data into windows
    vol_ma = df["volume"].rolling(volume_sma_window).mean()
    df["log_volume"] = np.log((df["volume"] + 1.0) / (vol_ma + 1.0)).astype(np.float32)

    # ── Order Flow Imbalance (Volume Delta) ───────────────────
    # Normalizăm delta_volume raportat la volumul total al lumânării.
    # Ne dă o valoare strict între -1.0 (100% vânzători) și +1.0 (100% cumpărători).
    if "delta_volume" in df.columns:
        df["volume_imbalance"] = (df["delta_volume"] / (df["volume"] + 1.0)).astype(np.float32)
    elif "ask_volume" in df.columns and "bid_volume" in df.columns:
        df["volume_imbalance"] = ((df["ask_volume"] - df["bid_volume"]) / (df["volume"] + 1.0)).astype(np.float32)
    df = df.dropna(subset=["log_return", "log_volume"]).reset_index(drop=True)
    return df
