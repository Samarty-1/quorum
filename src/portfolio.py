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
    adaptive_throttle,
    asymmetric_volatility,
    diversification_ratio,
    effective_bets,
    ledoit_wolf_covariance,
    volatility_scalar,
)
from src.sleeves.base import (
    Sleeve,
    SleeveContext,
    hold_between_rebalances,
    rebalance_dates,
)

TRADING_DAYS = 252


@dataclass
class PortfolioConfig:
    lookback_days: int = 504          # two years of trailing sleeve returns
    rebalance: str = "monthly"
    cost_bps: float | pd.Series = 5.0
    target_vol: float | None = 0.08   # None disables volatility targeting
    max_leverage: float = 3.0

    #: Scale each sleeve to its own volatility target BEFORE allocating.
    #:
    #: This is the fix for the layer that was inoperative. With unit-gross
    #: sleeves averaged at 1/N, roughly half the gross cancels on netting, so
    #: the combined book ran at 2.40% median volatility and needed 4.15x
    #: leverage to reach a 10% target -- which meant the leverage cap bound on
    #: 95.3% of days and the "dynamic scalar" was a constant.
    #:
    #: Targeting each sleeve individually puts the leverage where the risk is
    #: actually understood, at the level where a sleeve's own volatility is
    #: estimated from its own returns, instead of asking one portfolio-level
    #: scalar to undo the netting after the fact.
    sleeve_target_vol: float | None = 0.10
    sleeve_vol_halflife: int = 40
    sleeve_max_leverage: float = 4.0

    #: Hard cap on summed absolute weight, independent of the vol target. A
    #: volatility estimate can be wrong; a gross exposure limit cannot be.
    max_gross: float | None = 3.0

    # --- volatility estimator ------------------------------------------------
    #: EWMA half-lives for the portfolio scalar. The fast one de-levers, the
    #: slow one gates re-levering (see risk.asymmetric_volatility).
    vol_fast_halflife: int = 20
    vol_slow_halflife: int = 60

    # --- drawdown throttle ---------------------------------------------------
    use_drawdown_throttle: bool = False
    #: Thresholds are in STANDARD DEVIATIONS of drawdown, not percent, and the
    #: response is continuous. See risk.adaptive_throttle.
    throttle_start_sigma: float = 1.5
    throttle_stop_sigma: float = 3.0
    throttle_floor: float = 0.30


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


def vol_target_sleeves(books: dict[str, pd.DataFrame], asset_returns: pd.DataFrame,
                       config: PortfolioConfig,
                       sleeves: list[Sleeve] | None = None) -> dict[str, pd.DataFrame]:
    """Scale each sleeve to a common volatility before any capital is allocated.

    Two things this fixes at once.

    **The allocator finally compares like with like.** Unit *gross* exposure is
    not unit *risk*: on this book the sleeve volatilities ranged from 5.7% to
    10.4%, so an equal-weight allocation was already an implicit risk bet. Every
    allocator downstream now decides how much risk to give a sleeve, not how
    much notional.

    **The portfolio scalar stops being a constant.** With sleeves at a known
    volatility, the combined book lands near its target by construction and the
    portfolio-level scalar has something small to do, instead of being pinned at
    a cap it can never leave.

    The per-sleeve scalar is lagged one day and capped, for the same reasons the
    portfolio one is: the estimate is trailing, and 1/vol is unbounded as vol
    goes to zero.

    It is also HELD BETWEEN THE SLEEVE'S OWN REBALANCES, which is not a detail.
    A volatility estimate moves every day, so multiplying a book by a daily
    scalar re-trades the whole position daily even when not a single signal has
    changed. Applied naively that alone took this book's turnover from 3.4 to
    8.3 round trips a year on the trend sleeve and 0.9 to 5.9 on carry -- pure
    cost, no information. The scalar steps on the same schedule as the signals
    it is scaling.
    """
    if config.sleeve_target_vol is None:
        return books

    frequencies = {s.name: s.rebalance for s in sleeves} if sleeves else {}

    scaled: dict[str, pd.DataFrame] = {}
    for name, book in books.items():
        raw = backtest.run(book, asset_returns, cost_bps=0.0).returns
        vol = asymmetric_volatility(raw, config.sleeve_vol_halflife,
                                    config.sleeve_vol_halflife * 3)
        scalar = (config.sleeve_target_vol / vol).replace([np.inf, -np.inf], np.nan)
        scalar = scalar.clip(upper=config.sleeve_max_leverage).shift(1)
        scalar = hold_between_rebalances(
            scalar.to_frame("s"), frequencies.get(name, config.rebalance)
        )["s"].fillna(0.0)
        scaled[name] = book.mul(scalar.reindex(book.index).fillna(0.0), axis=0)
    return scaled


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
    unlevered = backtest.run(asset_weights, asset_returns, cost_bps=0.0).returns

    if config.target_vol is not None:
        # Asymmetric EWMA rather than a fixed rolling window: de-lever on the
        # fast estimate, re-lever only when the slow one agrees. On a 126-day
        # window this book took 40 days to register COVID and never registered
        # Volmageddon at all.
        trailing_vol = asymmetric_volatility(
            unlevered, config.vol_fast_halflife, config.vol_slow_halflife
        ).shift(1)
        leverage = trailing_vol.apply(
            lambda v: volatility_scalar(v, config.target_vol, config.max_leverage)
        ).fillna(0.0).rename("leverage")
        # Hold the scalar between rebalances too, or the book re-levers daily
        # and pays for it.
        leverage = leverage.reindex(decision_dates).reindex(dates).ffill().fillna(0.0)

    if config.use_drawdown_throttle:
        # The SHADOW book: unthrottled, so the throttle cannot ratchet itself.
        # Reading its own throttled equity curve is what makes a de-risking rule
        # self-trapping -- cutting risk slows the recovery, which keeps the
        # drawdown deep, which keeps the book cut.
        shadow = backtest.run(asset_weights.mul(leverage, axis=0),
                              asset_returns, cost_bps=config.cost_bps).returns
        shadow_vol = asymmetric_volatility(shadow, config.vol_fast_halflife,
                                           config.vol_slow_halflife)
        throttle = adaptive_throttle(
            shadow, shadow_vol,
            start_sigma=config.throttle_start_sigma,
            stop_sigma=config.throttle_stop_sigma,
            floor=config.throttle_floor,
        ).shift(1).fillna(1.0)
        throttle = throttle.reindex(decision_dates).reindex(dates).ffill().fillna(1.0)
        leverage = leverage * throttle

    final_weights = asset_weights.mul(leverage, axis=0)

    if config.max_gross is not None:
        # A gross cap the volatility estimate cannot override. Vol targeting
        # asks how much risk the book should take given an estimate; this asks
        # how much notional it may hold if that estimate is wrong.
        gross = final_weights.abs().sum(axis=1)
        excess = (config.max_gross / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        final_weights = final_weights.mul(excess, axis=0)
        leverage = leverage * excess
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
