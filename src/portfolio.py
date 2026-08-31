"""The multi-strategy book: sleeves in, one return stream out.

The walk-forward loop is the load-bearing part. On each rebalance date the
allocator sees a trailing window of sleeve returns ending the previous day,
produces sleeve weights, and those weights are held until the next rebalance.
Nothing that happens after the decision date can reach back into it.

Why the book is combined at the ASSET level, not the return level
------------------------------------------------------------------
There are two ways to run a multi-strategy portfolio. The naive one is to
compute each sleeve's return stream separately and average them. The real one is
to combine the sleeves' *positions* into a single book and trade that.

They are not the same, and the difference is the whole reason a desk nets its
sleeves. When trend is long SPY and reversal is short it, the netted book trades
neither -- the internal crossing is free. Averaging the return streams charges
both sleeves' costs on positions that cancelled. With one sleeve turning over
67x a year, that difference is large enough to change the conclusion, and this
module measures it explicitly (:func:`netting_benefit`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import backtest
from src.allocators import Allocator, EqualWeight
from src.risk import (
    diversification_ratio,
    drawdown_throttle,
    effective_bets,
    ledoit_wolf_covariance,
    realised_volatility,
    volatility_scalar,
)
from src.sleeves.base import Sleeve, SleeveContext, rebalance_dates

TRADING_DAYS = 252


@dataclass
class PortfolioConfig:
    lookback_days: int = 504          # two years of trailing sleeve returns
    rebalance: str = "monthly"
    cost_bps: float = 5.0
    target_vol: float | None = 0.08   # None disables volatility targeting
    max_leverage: float = 3.0
    vol_window: int = 63
    use_drawdown_throttle: bool = False
    throttle_start: float = 0.10
    throttle_stop: float = 0.20
    throttle_floor: float = 0.25


@dataclass
class PortfolioResult:
    name: str
    result: backtest.BacktestResult
    sleeve_weights: pd.DataFrame        # dates x sleeves, as decided
    asset_weights: pd.DataFrame         # dates x assets, netted, before overlay
    leverage: pd.Series
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def metrics(self) -> dict:
        out = self.result.metrics()
        out["name"] = self.name
        out["avg_leverage"] = round(float(self.leverage.replace(0.0, np.nan).mean()), 3)
        if not self.diagnostics.empty:
            for column in ("diversification_ratio", "effective_bets"):
                if column in self.diagnostics:
                    out[column] = round(float(self.diagnostics[column].mean()), 3)
        return out


def sleeve_books(sleeves: list[Sleeve], context: SleeveContext) -> dict[str, pd.DataFrame]:
    """Each sleeve's unit-gross asset weights, keyed by sleeve name."""
    return {sleeve.name: sleeve.weights(context) for sleeve in sleeves}


def sleeve_return_streams(books: dict[str, pd.DataFrame], returns: pd.DataFrame,
                          cost_bps: float) -> pd.DataFrame:
    """Each sleeve's standalone net return, for the allocator to learn from.

    Charged at the sleeve's own turnover, which is the right input for an
    allocator deciding how much to fund: it should be comparing what each sleeve
    delivers after its own trading, not before.
    """
    streams = {
        name: backtest.run(book, returns, cost_bps=cost_bps, name=name).returns
        for name, book in books.items()
    }
    return pd.DataFrame(streams)


def run_portfolio(books: dict[str, pd.DataFrame], sleeve_returns: pd.DataFrame,
                  asset_returns: pd.DataFrame, allocator: Allocator,
                  config: PortfolioConfig, name: str | None = None) -> PortfolioResult:
    """Walk forward, allocating across sleeves and netting them into one book."""
    sleeve_names = list(books.keys())
    dates = asset_returns.index
    decision_dates = rebalance_dates(dates, config.rebalance)

    sleeve_weight_rows: dict[pd.Timestamp, np.ndarray] = {}
    diagnostic_rows: list[dict] = []
    fallback = EqualWeight()

    for date in decision_dates:
        # Strictly trailing: everything up to and including the decision date.
        # The sleeve return on `date` itself is already knowable at its close,
        # which is when the decision is made.
        window = sleeve_returns.loc[:date].tail(config.lookback_days).dropna()

        if len(window) < max(allocator.min_observations, 2):
            weights = fallback.allocate(sleeve_returns.iloc[:1])
        else:
            weights = allocator.allocate(window)

        sleeve_weight_rows[date] = weights

        if len(window) >= 60:
            covariance, shrinkage = ledoit_wolf_covariance(window)
            diagnostic_rows.append({
                "date": date,
                "diversification_ratio": diversification_ratio(weights, covariance),
                "effective_bets": effective_bets(weights, covariance),
                "shrinkage": shrinkage,
                "mean_pairwise_corr": _mean_pairwise_correlation(window),
            })

    sleeve_weights = pd.DataFrame(sleeve_weight_rows, index=sleeve_names).T
    sleeve_weights = sleeve_weights.reindex(dates).ffill().fillna(0.0)

    # Net the sleeves into one asset book. This is where offsetting positions
    # cancel before they are ever traded.
    asset_weights = pd.DataFrame(0.0, index=dates, columns=asset_returns.columns)
    for sleeve_name in sleeve_names:
        book = books[sleeve_name].reindex(dates).fillna(0.0)
        asset_weights += book.mul(sleeve_weights[sleeve_name], axis=0)

    leverage = pd.Series(1.0, index=dates, name="leverage")

    if config.target_vol is not None:
        # Volatility of the unlevered netted book, estimated on trailing data
        # and lagged one day so the scalar applied today uses only yesterday's
        # information.
        unlevered = backtest.run(asset_weights, asset_returns, cost_bps=0.0).returns
        trailing_vol = realised_volatility(unlevered, config.vol_window).shift(1)
        leverage = trailing_vol.apply(
            lambda v: volatility_scalar(v, config.target_vol, config.max_leverage)
        ).fillna(0.0).rename("leverage")
        # Hold the scalar between rebalances too, or the book re-levers daily
        # and pays for it.
        leverage = leverage.reindex(decision_dates).reindex(dates).ffill().fillna(0.0)

    if config.use_drawdown_throttle:
        unlevered = backtest.run(asset_weights.mul(leverage, axis=0),
                                 asset_returns, cost_bps=config.cost_bps).returns
        equity = (1.0 + unlevered).cumprod()
        throttle = drawdown_throttle(
            equity, config.throttle_start, config.throttle_stop, config.throttle_floor
        ).shift(1).fillna(1.0)
        throttle = throttle.reindex(decision_dates).reindex(dates).ffill().fillna(1.0)
        leverage = leverage * throttle

    final_weights = asset_weights.mul(leverage, axis=0)
    result = backtest.run(final_weights, asset_returns, cost_bps=config.cost_bps,
                          name=name or allocator.name)

    diagnostics = pd.DataFrame(diagnostic_rows)
    if not diagnostics.empty:
        diagnostics = diagnostics.set_index("date")

    return PortfolioResult(
        name=name or allocator.name,
        result=result,
        sleeve_weights=sleeve_weights,
        asset_weights=asset_weights,
        leverage=leverage,
        diagnostics=diagnostics,
    )


def _mean_pairwise_correlation(window: pd.DataFrame) -> float:
    if window.shape[1] < 2:
        return float("nan")
    correlations = window.corr().to_numpy()
    off_diagonal = correlations[~np.eye(correlations.shape[0], dtype=bool)]
    # A constant sleeve (still inside its warmup) yields an all-NaN column, and
    # nanmean over an empty slice warns rather than simply returning NaN.
    finite = off_diagonal[np.isfinite(off_diagonal)]
    return float(finite.mean()) if finite.size else float("nan")


def netting_benefit(portfolio: PortfolioResult, books: dict[str, pd.DataFrame],
                    sleeve_returns_gross: pd.DataFrame, asset_returns: pd.DataFrame,
                    config: PortfolioConfig) -> dict:
    """Cost saved by netting sleeves into one book instead of averaging streams.

    The naive arm re-runs each sleeve separately at the same allocation and adds
    the return streams, so every sleeve pays its own full turnover. The netted
    arm is the portfolio actually run. The gap is what the desk saves purely by
    crossing offsetting orders internally, and it is a genuine operational edge
    that has nothing to do with signal quality.
    """
    dates = asset_returns.index
    naive = pd.Series(0.0, index=dates)
    for sleeve_name, book in books.items():
        allocation = portfolio.sleeve_weights[sleeve_name]
        scaled = book.reindex(dates).fillna(0.0).mul(allocation * portfolio.leverage, axis=0)
        naive = naive.add(
            backtest.run(scaled, asset_returns, cost_bps=config.cost_bps).returns,
            fill_value=0.0,
        )

    netted_turnover = float(portfolio.result.turnover.mean() * TRADING_DAYS)
    naive_turnover = 0.0
    for sleeve_name, book in books.items():
        allocation = portfolio.sleeve_weights[sleeve_name]
        scaled = book.reindex(dates).fillna(0.0).mul(allocation * portfolio.leverage, axis=0)
        naive_turnover += float(
            backtest.run(scaled, asset_returns, cost_bps=0.0).turnover.mean() * TRADING_DAYS
        )

    return {
        "netted_sharpe": round(backtest.sharpe_of(portfolio.result.returns), 3),
        "naive_sum_sharpe": round(backtest.sharpe_of(naive), 3),
        "netted_turnover_annual": round(netted_turnover, 2),
        "naive_turnover_annual": round(naive_turnover, 2),
        "turnover_saved_pct": round(
            100.0 * (1.0 - netted_turnover / naive_turnover), 1
        ) if naive_turnover > 0 else float("nan"),
    }
