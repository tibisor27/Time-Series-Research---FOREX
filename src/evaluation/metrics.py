"""
Evaluation Metrics — all metric calculations in one testable module.

(B) Return-space metrics (the ones a quant actually trusts):
    - IC               : Spearman correlation between predicted and actual log-returns
    - IR               : Information Ratio (mean IC / std IC across sub-windows)
    - Sharpe_net       : Annualized Sharpe of a sign(prediction)-position strategy net of costs
    - Sortino_net      : Sharpe variant penalizing only downside volatility
    - MaxDD_pct        : Maximum drawdown of the equity curve in percentage
    - ProfitFactor     : Gross wins / gross losses
    - HitRate          : Fraction of profitable windows

The H-window protocol assumed by the return-space metrics:
    - The rolling forecast emits H predictions per window.
    - For each window k, anchor = actual[k*H - 1] (last known truth before forecast).
    - End-of-window prediction = predicted[k*H + H - 1].
    - End-of-window actual     = actual[k*H + H - 1].
    - predicted_log_return     = log(end_pred / anchor)
    - actual_log_return        = log(end_actual / anchor)
    - position                 = sign(predicted_log_return)
    - realized PnL             = position * actual_log_return - cost_log
"""

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
    # Direction Accuracy: modelul a prezis semnul corect?
    signs_match = np.sign(y_pred) == np.sign(y_true)
    direction_pct = float(signs_match.mean() * 100)

    # Strategy Returns: tranzacționăm bazat pe semnul predicției
    strategy_returns = np.sign(y_pred) * y_true

    # Sharpe Ratio (anualizat)
    sr_mu = float(np.mean(strategy_returns))
    sr_sd = float(np.std(strategy_returns, ddof=1))
    sharpe = (sr_mu / sr_sd * np.sqrt(bars_per_year)) if sr_sd > 1e-12 else 0.0

    # Profit Factor: câștiguri / pierderi
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
    """Afișează metricile într-un format ușor de citit."""
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
    """
    Construiește un tabel Pandas sortabil din lista de rezultate.

    Usage (în Notebook):
        all_results = []
        all_results.append({"Model": "OLS", **compute_metrics_returns(y_test, pred_ols)})
        all_results.append({"Model": "Ridge", **compute_metrics_returns(y_test, pred_ridge)})
        table = build_comparison_table(all_results)
        display(table)
    """
    df = pd.DataFrame(all_results)
    if "Direction_%" in df.columns:
        df = df.sort_values("Direction_%", ascending=False).reset_index(drop=True)
    return df
