"""Turning weights into returns, honestly.

Two things here decide whether any result in this repo means anything.

**The shift.** A weight formed from data up to the close of day t earns the
return from t to t+1. Every position series is shifted exactly once, in one
place, so that a sleeve cannot accidentally shift again and produce a result
that is merely conservative rather than correct.

**The costs.** Turnover is charged where it happens, on the day the book
changes, at a rate the caller has to supply. The default is not zero. A
zero-cost default is how a high-turnover sleeve gets published looking good --
short-term reversal in this book earns roughly its whole gross return back in
trading, and at zero cost that is invisible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    returns: pd.Series          # net daily returns
    gross_returns: pd.Series
    turnover: pd.Series         # per-day sum |dw|
    weights: pd.DataFrame       # as held (already shifted)
    cost_bps: float
    name: str = ""

    @property
    def net_of_cost_drag(self) -> float:
        return float((self.gross_returns - self.returns).sum())

    def metrics(self) -> dict:
        return performance_metrics(self.returns, self.turnover, name=self.name)


def run(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 5.0,
        name: str = "") -> BacktestResult:
    """Apply a weight book to asset returns, charging costs on turnover.

    `weights` is indexed by the day the position is DECIDED. The shift to the
    day it is held happens here, once.
    """
    aligned = weights.reindex(returns.index).fillna(0.0)
    held = aligned.shift(1).fillna(0.0)

    gross = (held * returns).sum(axis=1)

    # Turnover is measured on the decision series: the trade happens at the
    # close of the day the weight changes. The first row is the cost of putting
    # the book on from flat, which is small over twenty years but is a real
    # trade and is not free.
    turnover = aligned.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(aligned.iloc[0].abs().sum())

    # Shifted to match: a trade at the close of day t is paid out of the return
    # earned from t to t+1, which is the same return the new position earns.
    costs = turnover.shift(1).fillna(0.0) * (cost_bps / 10_000.0)

    return BacktestResult(
        returns=(gross - costs).rename(name or "net"),
        gross_returns=gross.rename(name or "gross"),
        turnover=turnover.rename("turnover"),
        weights=held,
        cost_bps=cost_bps,
        name=name,
    )


def performance_metrics(returns: pd.Series, turnover: pd.Series | None = None,
                        periods_per_year: int = TRADING_DAYS, name: str = "") -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {"name": name, "n_days": len(r)}

    total = float((1.0 + r).prod() - 1.0)
    years = len(r) / periods_per_year
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else np.nan

    out = {
        "name": name,
        "n_days": int(len(r)),
        "years": round(years, 2),
        "total_return": round(total, 4),
        "annual_return": round(float((1.0 + total) ** (1.0 / years) - 1.0), 4) if years > 0 else np.nan,
        "annual_vol": round(vol, 4),
        "sharpe": round(float(r.mean() * periods_per_year / vol), 3) if vol > 0 else np.nan,
        "sortino": round(float(r.mean() * periods_per_year / downside_vol), 3)
        if downside_vol and np.isfinite(downside_vol) and downside_vol > 0 else np.nan,
        "max_drawdown": round(float(drawdown.min()), 4),
        "hit_rate": round(float((r > 0).mean()), 4),
        "worst_day": round(float(r.min()), 4),
        "best_day": round(float(r.max()), 4),
        # Fat tails are the thing a Sharpe ratio hides, and a multi-strategy
        # book is sold on smoothness -- so the shape of the distribution is
        # reported next to the ratio rather than in an appendix.
        "skew": round(float(r.skew()), 3),
        "excess_kurtosis": round(float(r.kurtosis()), 3),
    }
    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0.0)
        out["turnover_per_day"] = round(float(t.mean()), 4)
        out["turnover_annual"] = round(float(t.mean() * periods_per_year), 2)
    return out


def sharpe_of(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 3:
        return float("nan")
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(r.mean() * periods_per_year / (sd * np.sqrt(periods_per_year)))


def newey_west_sharpe_tstat(returns: pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean daily return, corrected for autocorrelation.

    Daily strategy returns are not independent -- a monthly-rebalanced book
    holds the same position for twenty days, so consecutive returns share a
    driver. The plain t-statistic overstates the evidence for exactly the
    reason it does with overlapping forward returns.
    """
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 10:
        return float("nan")
    if lags is None:
        # Newey-West's rule of thumb.
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))

    e = (r - r.mean()).to_numpy()
    variance = float(e @ e / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(e[lag:] @ e[:-lag] / n)
    if not np.isfinite(variance) or variance <= 0:
        return float("nan")
    return float(r.mean() / np.sqrt(variance / n))
