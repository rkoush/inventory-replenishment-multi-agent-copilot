"""Deterministic demand forecast (no LLM math).

Day-of-week seasonal indices + simple exponential smoothing on the
de-seasonalised series. Small, explainable, and good enough for a weekly
replenishment demo; swap in Prophet/ETS/LightGBM later behind the same interface.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ALPHA = 0.2  # smoothing factor for the level
SEASON_WEEKS = 8  # weeks of history used for day-of-week indices


def _fit(sales: pd.Series) -> tuple[float, np.ndarray]:
    recent = sales.iloc[-SEASON_WEEKS * 7 :]
    dow_mean = recent.groupby(recent.index.day_of_week).mean().reindex(range(7)).fillna(recent.mean())
    dow_index = (dow_mean / recent.mean()).to_numpy() if recent.mean() > 0 else np.ones(7)
    deseason = sales / dow_index[sales.index.day_of_week]
    level = float(deseason.ewm(alpha=ALPHA, adjust=False).mean().iloc[-1])
    return level, dow_index


def _project(level: float, dow_index: np.ndarray, start: pd.Timestamp, days: int) -> pd.Series:
    idx = pd.date_range(start, periods=days, freq="D")
    return pd.Series(level * dow_index[idx.day_of_week], index=idx)


def forecast_sku(history: list[dict[str, Any]], lead_time_days: int, horizon_days: int = 7) -> dict[str, Any]:
    """history = [{"date": "YYYY-MM-DD", "units_sold": int}, ...] (oldest first)."""
    s = pd.Series(
        [h["units_sold"] for h in history], index=pd.to_datetime([h["date"] for h in history]), dtype=float
    ).asfreq("D", fill_value=0.0)
    if len(s) < 28:
        raise ValueError("Need at least 28 days of history to forecast.")

    # Backtest on the last 7 days (fit on everything before) -> WAPE for transparency
    train, test = s.iloc[:-7], s.iloc[-7:]
    lvl, dow = _fit(train)
    pred = _project(lvl, dow, test.index[0], 7)
    wape = float(np.abs(test.values - pred.values).sum() / max(test.sum(), 1.0))

    level, dow_index = _fit(s)
    days = lead_time_days + horizon_days
    fc = _project(level, dow_index, s.index[-1] + pd.Timedelta(days=1), days)
    return {
        "method": "dow-seasonal exponential smoothing",
        "avg_daily": round(float(fc.mean()), 2),
        "demand_lead": round(float(fc.iloc[:lead_time_days].sum()), 1),
        "demand_horizon": round(float(fc.iloc[lead_time_days:].sum()), 1),
        "forecast_start": str(fc.index[0].date()),
        "backtest_wape_7d": round(wape, 3),
    }
