from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bars_per_year: int = 24000,
) -> dict[str, float]:

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # ── MLOps Defensive Guardrails ────────────────────────────
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        raise ValueError("Evaluator Error: Predictions contain NaN or Inf values! Check model output.")
    if np.isnan(y_true).any() or np.isinf(y_true).any():
        raise ValueError("Evaluator Error: Ground truth targets contain NaN or Inf values! Check Formatter.")
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError(f"Evaluator Error: Shape mismatch or empty arrays! y_true={len(y_true)}, y_pred={len(y_pred)}")

    # ── ML Standard ───────────────────────────────────────────
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # R² — cât din varianța target-ului e explicată de model
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # ── Information Coefficient (IC) ──────────────────────────
    if np.std(y_pred) > 1e-12:
        from scipy.stats import pearsonr as _pr, spearmanr as _sr
        ic_pearson = float(_pr(y_pred, y_true).statistic)
        ic_spearman = float(_sr(y_pred, y_true).statistic)
    else:
        ic_pearson, ic_spearman = 0.0, 0.0

    # ── Financial Metrics ─────────────────────────────────────
    # Direction Accuracy
    signs_match = np.sign(y_pred) == np.sign(y_true)
    direction_pct = float(signs_match.mean() * 100)

    # Strategy Returns
    strategy_returns = np.sign(y_pred) * y_true

    # Sharpe Ratio
    sr_mu = float(np.mean(strategy_returns))
    sr_sd = float(np.std(strategy_returns, ddof=1))
    sharpe = (sr_mu / sr_sd * np.sqrt(bars_per_year)) if sr_sd > 1e-12 else 0.0

    # Profit Factor (annualized)
    wins = float(np.sum(strategy_returns[strategy_returns > 0]))
    losses = float(np.abs(np.sum(strategy_returns[strategy_returns <= 0])))
    pf = (wins / losses) if losses > 1e-12 else 0.0

    # Max Drawdown
    cum = np.cumsum(strategy_returns)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd) * 100) if len(dd) > 0 else 0.0

    return {
        "MSE": round(mse, 12),
        "MAE": round(mae, 10),
        "RMSE": round(rmse, 10),
        "R2": round(float(r2), 6),
        "IC_Pearson": round(float(ic_pearson), 4),
        "IC_Spearman": round(float(ic_spearman), 4),
        "Direction_%": round(direction_pct, 2),
        "Sharpe": round(float(sharpe), 4),
        "ProfitFactor": round(float(pf), 4),
        "MaxDD_%": round(float(max_dd), 3),
    }


def print_metrics(model_name: str, metrics: dict) -> None:
    
    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    print(f"  ── ML Standard ──")
    print(f"  MSE           = {metrics['MSE']:.4e}")
    print(f"  MAE           = {metrics['MAE']:.4e}")
    print(f"  R²            = {metrics['R2']:.6f}")
    print(f"  ── Signal Quality ──")
    print(f"  IC Pearson    = {metrics['IC_Pearson']:+.4f}")
    print(f"  IC Spearman   = {metrics['IC_Spearman']:+.4f}")
    print(f"  ── Financial ──")
    print(f"  Direction     = {metrics['Direction_%']:.2f}%")
    print(f"  Sharpe        = {metrics['Sharpe']:.4f}")
    print(f"  Profit Factor = {metrics['ProfitFactor']:.4f}")
    print(f"  Max Drawdown  = {metrics['MaxDD_%']:.3f}%")
    print(f"{'=' * 50}")


def build_comparison_table(all_results: list[dict]) -> pd.DataFrame:

    df = pd.DataFrame(all_results)
    if "Direction_%" in df.columns:
        df = df.sort_values("Direction_%", ascending=False).reset_index(drop=True)
    return df
