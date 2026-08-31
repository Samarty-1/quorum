"""What a sleeve is, and the rules every one of them obeys.

A sleeve is one strategy. It sees prices and distributions, and returns a target
weight per asset per day. It knows nothing about the other sleeves, about how
much capital it will be given, or about the risk overlay -- those are the
allocator's job, and keeping the boundary clean is what makes it possible to ask
"is this sleeve adding anything" as a separate question from "is the allocator
any good".

Three rules, enforced here rather than trusted to each sleeve
-------------------------------------------------------------
**Weights are formed from trailing data only.** Every signal is computed from
data up to and including day t, and :func:`Sleeve.weights` returns the position
to *hold into* day t+1. The backtester shifts once and only once; a sleeve that
shifted internally as well would be flat wrong in the safe direction and
undetectable in the results.

**Weights are normalised to unit gross exposure.** Every sleeve returns a book
with sum |w| = 1, so no sleeve can quietly out-lever another. The allocator
decides scale; the sleeve decides shape. Without this the "capital allocation"
question is meaningless because the sleeves have already allocated themselves.

**Rebalancing is explicit and periodic.** A signal recomputed daily and traded
daily is a signal that pays daily costs. Each sleeve declares how often it
actually re-trades, and the weights are held constant in between. Short-term
reversal is the honest test of this: it is the highest-turnover idea here and
the one most likely to be destroyed by costs, which is a finding rather than a
problem to hide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class SleeveContext:
    """Everything a sleeve is allowed to see."""

    prices: pd.DataFrame          # total-return index, dates x assets
    dividends: pd.DataFrame       # long: ticker, date, amount
    cash_daily: pd.Series         # daily risk-free rate on the price calendar
    #: ticker -> asset class. Needed by any sleeve that must neutralise the
    #: structural ordering of asset classes before ranking across them.
    asset_class: dict[str, str] | None = None


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Trading days on which a sleeve is allowed to re-trade.

    Anchored to the calendar rather than to a rolling day count, so a sleeve's
    rebalance dates do not drift with the sample start -- two runs over
    different windows rebalance on the same days where they overlap.
    """
    if frequency == "daily":
        return index
    # Period aliases, not offset aliases: `to_period` wants "M", not "ME".
    period = {"weekly": "W", "monthly": "M", "quarterly": "Q"}.get(frequency)
    if period is None:
        raise ValueError(f"unknown rebalance frequency: {frequency}")
    marker = pd.Series(index, index=index).groupby(index.to_period(period)).last()
    return pd.DatetimeIndex(marker.values)


def hold_between_rebalances(signal: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Sample a daily signal on its rebalance dates and hold in between."""
    dates = rebalance_dates(signal.index, frequency)
    held = signal.reindex(dates).reindex(signal.index).ffill()
    return held


def normalise_gross(weights: pd.DataFrame, target_gross: float = 1.0) -> pd.DataFrame:
    """Scale each row so sum |w| equals `target_gross`.

    Rows that are entirely zero -- a sleeve with no opinion, or still inside its
    warmup -- stay zero rather than becoming NaN. A NaN here would silently
    remove the day from the portfolio instead of holding cash on it.
    """
    gross = weights.abs().sum(axis=1)
    scale = pd.Series(0.0, index=weights.index)
    live = gross > 0
    scale[live] = target_gross / gross[live]
    return weights.mul(scale, axis=0).fillna(0.0)


def demean_cross_section(scores: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional mean, making the book dollar-neutral.

    A cross-sectional sleeve should express a view on relative performance, not
    smuggle in a market bet. Without demeaning, a momentum sleeve in a bull
    market is mostly just long -- and would look diversifying against trend
    following only because both are long the same thing.
    """
    return scores.sub(scores.mean(axis=1), axis=0)


def rank_scores(values: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank, mapped to [-1, 1] and demeaned.

    Ranks rather than raw values, because the inputs are on wildly different
    scales -- a 5-day return, a 12-month return and a dividend yield cannot be
    compared as numbers. Ranking also stops one extreme observation from
    dominating a fifteen-asset cross-section.
    """
    ranked = values.rank(axis=1, pct=True)
    centred = 2.0 * (ranked - 0.5)
    return demean_cross_section(centred)


def neutralise_within_class(scores: pd.DataFrame,
                            asset_class: dict[str, str] | None) -> pd.DataFrame:
    """Demean scores inside each asset class before ranking across them.

    Without this, a cross-sectional rank on any level-based signal is dominated
    by the structural ordering of asset classes rather than by relative
    cheapness. Measured on the income sleeve before this was applied: mean net
    exposure of Credit +0.223, Real assets +0.116, Equity **-0.279**, with HYG,
    VNQ and LQD as the largest persistent longs and QQQ, IWM, SPY the largest
    persistent shorts.

    That is not an income strategy. It is a permanent long-credit,
    short-US-equity macro position wearing a yield label -- and the value
    sleeve, ranking on a different signal, ended up expressing the same bet
    (Equity -0.205), which is why the two correlated at +0.34 and why both lost
    money across a US-equity-dominated sample.

    Demeaning within class asks the question the sleeve is supposed to ask: is
    this REIT cheap against other REITs, is this credit ETF cheap against other
    credit. The asset-class bet, if wanted, belongs in a separate sleeve where
    it can be sized deliberately.

    NaN in `scores` means "this asset has no reading", not "zero", and the
    distinction matters. Passing a NaN-filled frame in drags the class mean
    toward zero by however many non-observations the class contains: in a
    commodity class where one ETF distributes and two do not, the payer would be
    measured against a mean of one third its own yield and look enormously rich.
    The mean here skips NaN and NaN members stay NaN, so a caller can mask them
    out afterwards without having polluted anything.
    """
    if not asset_class:
        return scores

    out = scores.copy()
    classes: dict[str, list[str]] = {}
    for ticker in scores.columns:
        classes.setdefault(asset_class.get(ticker, "_unknown"), []).append(ticker)

    for members in classes.values():
        block = scores[members]
        # Count only members that actually have a reading on each date.
        observed = block.notna().sum(axis=1)
        if len(members) < 2:
            # A single-member class carries no within-class information. Zero it
            # rather than leaving its raw level to dominate the cross-section.
            out[members] = block.where(block.isna(), 0.0)
            continue
        centred = block.sub(block.mean(axis=1, skipna=True), axis=0)
        # A date on which only one member of the class has a reading is the
        # single-member case again, per date rather than per universe.
        out[members] = centred.where(observed.gt(1), other=centred.where(block.isna(), 0.0))
    return out


def apply_no_trade_band(weights: pd.DataFrame, band: float) -> pd.DataFrame:
    """Hold the existing position until the target moves more than `band`.

    Turnover is the tax on a signal that changes faster than it decays. The
    reversal sleeve here turns over 67 times a year and converts a +0.285 gross
    Sharpe into -0.033 net: the entire effect is smaller than the cost of
    harvesting it. A band trades only when the target has moved enough to be
    worth the spread, which is what a desk does and what a naive
    rebalance-to-target loop does not.

    Applied per asset on the L1 distance of the whole book, so the band cannot
    let the position drift arbitrarily far in aggregate.
    """
    if band <= 0.0:
        return weights

    values = weights.to_numpy(dtype=float)
    held = np.zeros_like(values)
    current = np.zeros(values.shape[1])
    opened = False

    for i in range(values.shape[0]):
        target = values[i]
        # Opening the book is not subject to the band. A unit-gross target sits
        # exactly 1.0 from flat, so any band above 1.0 would otherwise leave the
        # sleeve permanently at zero -- silently, with no error and a turnover
        # of exactly nothing to give it away.
        if not opened:
            if np.abs(target).sum() > 0.0:
                current = target.copy()
                opened = True
        elif np.abs(target - current).sum() > band:
            current = target.copy()
        held[i] = current
    return pd.DataFrame(held, index=weights.index, columns=weights.columns)


def trailing_volatility(returns: pd.DataFrame, span: int = 60,
                        min_periods: int = 30) -> pd.DataFrame:
    """EWMA volatility, annualised, using data up to and including each day."""
    return returns.ewm(span=span, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS)


def inverse_vol_scale(weights: pd.DataFrame, returns: pd.DataFrame,
                      span: int = 60, floor: float = 0.02) -> pd.DataFrame:
    """Scale positions by inverse trailing volatility.

    Without this a sleeve's risk is dominated by whichever asset happens to be
    most volatile -- in this universe, silver and emerging markets -- regardless
    of how strong the signal on it is. The floor stops a very quiet asset (short
    Treasuries in 2013) from being handed unbounded leverage.
    """
    vol = trailing_volatility(returns, span).reindex(weights.index).clip(lower=floor)
    return (weights / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def trailing_dividend_yield(context: SleeveContext, window_days: int = 365) -> pd.DataFrame:
    """Sum of distributions paid in the trailing year, over price.

    Uses only payments with an ex-date on or before each observation date. The
    usual way this goes wrong is a rolling sum over a forward-filled series,
    which lets a payment count on days before it happened.
    """
    prices = context.prices
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if context.dividends.empty:
        return out

    payments = context.dividends.copy()
    payments["date"] = pd.to_datetime(payments["date"])

    for ticker, group in payments.groupby("ticker"):
        if ticker not in out.columns:
            continue
        amounts = group.set_index("date")["amount"].sort_index()
        # searchsorted on both ends of the window, so each date sums exactly the
        # payments inside it.
        ex_dates = amounts.index
        cumulative = amounts.cumsum()
        upper = ex_dates.searchsorted(prices.index, side="right")
        lower = ex_dates.searchsorted(prices.index - pd.Timedelta(window_days, "D"), side="right")
        totals = np.where(upper > lower,
                          cumulative.values[np.clip(upper - 1, 0, len(cumulative) - 1)] -
                          np.where(lower > 0,
                                   cumulative.values[np.clip(lower - 1, 0, len(cumulative) - 1)],
                                   0.0),
                          0.0)
        out[ticker] = totals

    return (out / prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)


class Sleeve(ABC):
    """One strategy, producing a unit-gross book."""

    name: str = "sleeve"
    rebalance: str = "monthly"
    warmup_days: int = 252
    #: L1 distance the target book must move before the sleeve re-trades.
    #: Zero means rebalance to target every period.
    no_trade_band: float = 0.0

    @abstractmethod
    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        """Target weights before holding, normalisation and warmup masking."""

    def weights(self, context: SleeveContext) -> pd.DataFrame:
        """The book to hold into the next day.

        The template method is where the rules get applied uniformly, so a new
        sleeve only has to express its idea and cannot forget the plumbing.
        """
        raw = self.raw_weights(context)
        held = hold_between_rebalances(raw, self.rebalance)

        # Blank the warmup rather than trusting each signal to be NaN there. A
        # rolling mean with min_periods set produces NaN, but a rank over a row
        # of NaNs produces a perfectly plausible-looking zero.
        if self.warmup_days > 0:
            held.iloc[: self.warmup_days] = 0.0

        normalised = normalise_gross(held.fillna(0.0))
        # The band goes last, on the final book: it is a trading decision, not
        # a signal one, and applying it before normalisation would let the held
        # position drift away from unit gross.
        return apply_no_trade_band(normalised, self.no_trade_band)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, rebalance={self.rebalance!r})"
