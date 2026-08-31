"""The five sleeves.

Chosen to be genuinely different from one another, which is the only thing that
makes a multi-strategy book worth building. Two of them -- cross-sectional
momentum and short-term reversal -- are close to mechanical opposites, and that
is deliberate: if the allocator cannot exploit a pair that anticorrelated, it
will not exploit anything.

Each is a standard, published idea rather than something tuned on this sample.
That is a deliberate constraint. With fifteen assets and one history, any
parameter chosen by looking at the results would be fitted to it, and the
combined book would inherit every one of those choices. Literature defaults are
not optimal, but they are not fitted either, and the honest comparison is
between a book of unfitted sleeves and a single unfitted sleeve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sleeves.base import (
    Sleeve,
    SleeveContext,
    inverse_vol_scale,
    rank_scores,
    trailing_dividend_yield,
)


class TrendFollowing(Sleeve):
    """Time-series momentum: hold each asset long or short on its own trend.

    The one sleeve here that is directional. It takes a view on each asset
    against zero rather than against its peers, so in a sustained selloff it can
    be net short everything -- which is precisely the payoff that makes trend
    following worth owning next to four dollar-neutral sleeves. It is the
    convexity in the book.

    Signal is the sign of the trailing 12-month excess return, the Moskowitz-
    Ooi-Pedersen formulation, sized inverse to trailing volatility so a
    position in silver does not carry six times the risk of one in Treasuries.
    """

    name = "trend"
    rebalance = "monthly"
    warmup_days = 252 + 60

    def __init__(self, lookback_days: int = 252, vol_span: int = 60):
        self.lookback_days = lookback_days
        self.vol_span = vol_span

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        prices = context.prices
        returns = prices.pct_change()

        total_return = prices / prices.shift(self.lookback_days) - 1.0
        # Against cash, not against zero: in 2007 a 4% return was underperforming
        # the bill, and calling that an uptrend would have been wrong.
        cash_over_window = (
            context.cash_daily.reindex(prices.index).fillna(0.0)
            .rolling(self.lookback_days, min_periods=self.lookback_days // 2).sum()
        )
        excess = total_return.sub(cash_over_window, axis=0)

        direction = np.sign(excess)
        return inverse_vol_scale(direction, returns, self.vol_span)


class CrossSectionalMomentum(Sleeve):
    """Rank assets on 12-month return skipping the last month; long winners.

    The one-month skip is not decoration. Short-horizon returns reverse, so a
    12-month window that includes the most recent month blends momentum with its
    own opposite and dilutes both. Skipping it is standard for exactly that
    reason -- and this book runs that reversal as a separate sleeve, where the
    allocator can size it on its own merits.
    """

    name = "xs_momentum"
    rebalance = "monthly"
    warmup_days = 252 + 21

    def __init__(self, lookback_days: int = 252, skip_days: int = 21):
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        prices = context.prices
        recent = prices.shift(self.skip_days)
        past = prices.shift(self.lookback_days)
        performance = recent / past - 1.0
        return rank_scores(performance)


class ShortTermReversal(Sleeve):
    """Long recent losers, short recent winners, over a one-week window.

    The highest-turnover sleeve here by a wide margin, and the one whose fate is
    decided by transaction costs rather than by whether the effect exists. It is
    included partly because it is the sharpest test of whether the cost model is
    being taken seriously: at zero cost it looks excellent, and the interesting
    question is what survives.
    """

    name = "reversal"
    rebalance = "weekly"
    warmup_days = 60

    def __init__(self, lookback_days: int = 5):
        self.lookback_days = lookback_days

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        prices = context.prices
        recent_return = prices / prices.shift(self.lookback_days) - 1.0
        # Negated: the loser gets the long.
        return rank_scores(-recent_return)


class Carry(Sleeve):
    """Long high-yielding assets, short low-yielding ones.

    The income sleeve, expressed cross-sectionally rather than as a buy-and-hold
    in dividend payers -- which would be a disguised long-credit, long-REIT bet
    rather than a strategy. Yield is the clearest signal available on free data
    that is not a price transformation, so it is the one sleeve here whose input
    is genuinely independent of the other four.

    The risk it carries is worth naming: high carry is compensation for
    something, and in this universe that something is credit and duration. The
    sleeve should be expected to lose money in exactly the weeks trend following
    makes it, and the diagnostics check whether it does.
    """

    name = "carry"
    rebalance = "monthly"
    warmup_days = 365

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        yields = trailing_dividend_yield(context)
        # Assets that pay nothing (GLD, SLV, DBC hold no income) are excluded
        # rather than ranked bottom: shorting gold because it has no coupon is
        # a statement about commodities, not about carry.
        yields = yields.where(yields > 0)
        return rank_scores(yields).fillna(0.0)


class Value(Sleeve):
    """Long-horizon reversal: long what has lagged for years, short what has run.

    The value sleeve, and the substitution needs stating plainly. Real value
    needs fundamentals, and point-in-time fundamentals are not available free --
    using restated ones inflates a backtest measurably. So this uses the
    standard cross-asset proxy instead: the five-year-to-one-year return,
    inverted. It is the De Bondt-Thaler long-horizon reversal, and it is what
    cross-asset value factors use when a book value does not exist for gold.

    It is a proxy, not the thing. An asset can be cheap on this measure purely
    because it deserved to fall.
    """

    name = "value"
    rebalance = "quarterly"
    warmup_days = 252 * 5

    def __init__(self, long_days: int = 252 * 5, skip_days: int = 252):
        self.long_days = long_days
        self.skip_days = skip_days

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        prices = context.prices
        recent = prices.shift(self.skip_days)
        distant = prices.shift(self.long_days)
        long_run = recent / distant - 1.0
        return rank_scores(-long_run)


def default_sleeves() -> list[Sleeve]:
    """The book. Order is fixed so reports and tests line up."""
    return [TrendFollowing(), CrossSectionalMomentum(), ShortTermReversal(), Carry(), Value()]
