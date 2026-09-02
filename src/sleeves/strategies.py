"""The sleeves.

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
    neutralise_within_class,
    rank_scores,
    roll_yield,
    trailing_dividend_yield,
)
from src.universe import CARRY_SOURCE, ROLL_PROXY_FRONT, ROLL_PROXY_LADDER


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

    NOT IN THE DEFAULT BOOK. Kept because the measurement is worth keeping, and
    because it is the sharpest test of whether the cost model is being taken
    seriously -- at zero cost it looks like the third-best idea here.

    Why it was cut
    --------------
    Short-horizon reversal is a single-name microstructure effect -- liquidity
    provision, bid-ask bounce -- and it does not transfer to broad ETFs. The
    signal's IC decays from +0.026 fresh to +0.001 after five days, and neither
    the weekly (+0.026, t=1.76) nor the monthly-sampled (-0.054, t=-1.79)
    estimate clears two standard errors. The sleeve's own weight
    autocorrelation is -0.018 at five days: the position wanted next week is
    uncorrelated with the one held.

    Twelve repair variants were searched on the selection half -- lookbacks of
    3, 5 and 10 days, full cross-section against terciles, with and without a
    no-trade band (`scripts/reversal_search.py`). The best scored +0.153 there
    and **-0.132** on the confirmation half. The search did not even find a
    better configuration than the one already in use.

    A sleeve whose best-of-twelve variant loses money out of sample does not
    belong in a book, and keeping it because a variant looked good on the half
    that chose it is the exact error this repo exists to avoid.
    """

    name = "reversal"
    rebalance = "weekly"
    warmup_days = 60
    # Trade only when the target book has moved materially. Rebalancing to
    # target every week cost this sleeve its entire gross edge: +0.285 gross
    # Sharpe against -0.033 net at 67 round trips a year. The band is set from
    # the signal's own decay -- the IC is +0.026 fresh, +0.001 after five days
    # -- so there is no case for paying to chase small changes in a number that
    # is already noise by the time the next rebalance arrives.
    #
    # Calibrated, not guessed. The L1 distance between consecutive weekly target
    # books has a median of 1.32 and a 10th percentile of 0.75, so a band below
    # ~0.75 never binds at all -- an earlier 0.60 changed nothing. At 1.0 the
    # sleeve trades only when the book has substantially reshuffled: turnover
    # falls 67 -> 58 and net Sharpe goes -0.028 -> +0.085. It is still not a
    # significant result, and the README says so.
    no_trade_band = 1.0

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
        # A zero distribution yield is not zero carry, it is a missing reading.
        # Ranking gold bottom because it pays no coupon is a statement about the
        # data field, not about carry.
        yields = yields.where(yields > 0)

        # Each asset gets the carry measure that actually applies to it.
        # Futures-based funds earn (or pay) roll yield, which is a real number
        # and a real risk premium; physically backed bullion earns neither a
        # coupon nor a roll and is left with no reading rather than a fake one.
        roll = roll_yield(context, ROLL_PROXY_FRONT, ROLL_PROXY_LADDER)
        for ticker, source in CARRY_SOURCE.items():
            if ticker not in yields.columns:
                continue
            if source == "roll" and len(roll):
                yields[ticker] = roll.reindex(yields.index)
            else:
                yields[ticker] = np.nan
        # Neutralised within asset class before ranking across it. Without this
        # the sleeve is a permanent long-credit, short-equity position rather
        # than a carry signal -- see neutralise_within_class.
        #
        # The NaN frame goes in unfilled, deliberately. Filling non-payers with
        # zero first would drag each class mean toward zero by however many
        # non-payers it holds, so a distributing commodity ETF would be measured
        # against a mean a third of its own yield and rank as wildly cheap.
        neutral = neutralise_within_class(yields, context.asset_class)
        return rank_scores(neutral).fillna(0.0)


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
        # Same neutralisation as the carry sleeve, and for the same reason: on
        # raw long-horizon returns this sleeve was structurally short US equity
        # (mean net Equity exposure -0.205) and therefore expressing the same
        # macro bet as carry rather than an independent one.
        neutral = neutralise_within_class(-long_run, context.asset_class)
        return rank_scores(neutral)


def default_sleeves() -> list[Sleeve]:
    """The book. Order is fixed so reports and tests line up.

    Four sleeves, not five. ShortTermReversal was cut after its best-of-twelve
    repair variant scored -0.132 on the confirmation half -- see its docstring
    and scripts/reversal_search.py. It remains importable, and
    `all_sleeves()` still returns it, so the measurement stays reproducible.
    """
    return [TrendFollowing(), CrossSectionalMomentum(), Carry(), Value()]


def all_sleeves() -> list[Sleeve]:
    """Every sleeve including the cut one, for diagnostics and comparison."""
    return [TrendFollowing(), CrossSectionalMomentum(), ShortTermReversal(), Carry(), Value()]


# ---------------------------------------------------------------------------
# Candidates chosen from the external literature, not from this sample.
#
# Every parameter below comes from a published specification. That is the only
# defence available when adding strategies to a dataset that has already been
# looked at: if the parameters were picked here, the results would be fitted
# here, and the deflation would have to account for a search this repo could not
# honestly bound.
# ---------------------------------------------------------------------------


class MultiHorizonTrend(Sleeve):
    """Trend across three horizons at once, rather than twelve months alone.

    The single-horizon trend sleeve is the only thing in this study that worked
    (0.450 net, t = 2.13), and the literature's consistent finding is that a
    blend of short, medium and long lookbacks is more robust than any one of
    them -- Hurst, Ooi and Pedersen's "A Century of Evidence on Trend-Following
    Investing" and AQR's "Trends Everywhere" both build on 1-, 3- and 12-month
    signals combined, and Baltas-Kosowski find the blend's advantage is
    primarily a reduction in turnover-adjusted drawdown rather than a higher
    raw mean.

    Measured here, it did not help, and the reason is worth recording because
    it contradicts what this docstring originally claimed.

    The expected benefit was lower turnover -- three lookbacks whipsaw at
    different times, so the blend should change position less often than any
    component. The opposite happened: **2.7x the turnover of single-horizon
    trend**, and a lower Sharpe over 35 years (0.413 against 0.611).

    The cause is the construction. Averaging three `sign()` signals produces a
    four-state signal (-1, -1/3, +1/3, +1) rather than a two-state one, so it
    moves whenever ANY horizon flips, and each flip retrades a third of the
    book. The literature's turnover benefit comes from a *continuous* signal --
    AQR and Hurst-Ooi-Pedersen use z-scored or tanh-squashed momentum, where
    blending genuinely smooths magnitude rather than adding intermediate states.

    A continuous-signal version would be the fair test of the published claim
    and is NOT implemented here, deliberately: it would be another trial on a
    sample that has already been searched hard, and the deflation cannot absorb
    many more. Recorded as untested rather than quietly attempted.

    Equal weight across horizons, no optimisation over the blend.
    """

    name = "trend_multi"
    rebalance = "monthly"
    warmup_days = 252 + 60

    def __init__(self, lookbacks: tuple[int, ...] = (21, 63, 252), vol_span: int = 60):
        self.lookbacks = lookbacks
        self.vol_span = vol_span

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        prices = context.prices
        returns = prices.pct_change()
        cash = context.cash_daily.reindex(prices.index).fillna(0.0)

        signal = None
        for lookback in self.lookbacks:
            total = prices / prices.shift(lookback) - 1.0
            # Against cash over the same window, not against zero.
            hurdle = cash.rolling(lookback, min_periods=lookback // 2).sum()
            direction = np.sign(total.sub(hurdle, axis=0))
            signal = direction if signal is None else signal + direction

        signal = signal / float(len(self.lookbacks))
        return inverse_vol_scale(signal, returns, self.vol_span)


class BettingAgainstBeta(Sleeve):
    """Long low-beta assets, short high-beta ones, with the legs beta-matched.

    Frazzini and Pedersen (2014). The premise is that leverage-constrained
    investors bid up high-beta assets to get exposure they cannot borrow to
    obtain, so high beta is persistently overpriced per unit of risk. It is one
    of the few anomalies documented across equities, bonds, credit and futures
    simultaneously, and the 2024 crowding literature classes it with the
    "judgment" factors that decay more slowly than mechanical ones like
    momentum.

    The beta-matching is what makes it BAB rather than a disguised short on
    market risk: the long leg is levered up and the short leg levered down until
    both have unit beta, so the book is beta-neutral by construction and its
    return is not a market bet. Skipping that step -- ranking on beta and
    equal-weighting the legs -- produces a permanently net-short-beta portfolio
    that loses money in every bull market, which is a different strategy and a
    worse one.
    """

    name = "bab"
    rebalance = "monthly"
    warmup_days = 252 + 60

    def __init__(self, beta_window: int = 252, min_periods: int = 126,
                 max_leg_leverage: float = 3.0):
        self.beta_window = beta_window
        self.min_periods = min_periods
        self.max_leg_leverage = max_leg_leverage

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        prices = context.prices
        returns = prices.pct_change()
        # Equal-weight own universe as the market proxy: using an external index
        # would import a different investable set than the one being ranked.
        market = returns.mean(axis=1)

        covariance = returns.rolling(self.beta_window, min_periods=self.min_periods).cov(market)
        variance = market.rolling(self.beta_window, min_periods=self.min_periods).var()
        beta = covariance.div(variance, axis=0).replace([np.inf, -np.inf], np.nan)
        # Shrink toward one, as Frazzini-Pedersen do: rolling betas are noisy
        # and the raw estimate overstates dispersion.
        beta = 0.6 * beta + 0.4 * 1.0

        ranks = beta.rank(axis=1, pct=True)
        low = (ranks <= 0.5).astype(float)
        high = (ranks > 0.5).astype(float)
        low = low.div(low.sum(axis=1).replace(0, np.nan), axis=0)
        high = high.div(high.sum(axis=1).replace(0, np.nan), axis=0)

        # Beta-match the legs: scale each by the reciprocal of its own beta.
        beta_low = (low * beta).sum(axis=1)
        beta_high = (high * beta).sum(axis=1)
        scale_low = (1.0 / beta_low).clip(upper=self.max_leg_leverage)
        scale_high = (1.0 / beta_high).clip(upper=self.max_leg_leverage)

        weights = low.mul(scale_low, axis=0) - high.mul(scale_high, axis=0)
        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class TurnOfMonth(Sleeve):
    """Hold the market around the month boundary; hold nothing the rest of the time.

    Ogden (1990) and a long replication literature since: equity returns cluster
    into the last day of the month and the first three of the next, with the
    remaining ~16 trading days contributing close to nothing. The usual
    explanation is a liquidity/flow effect -- salary, pension and coupon flows
    concentrate at month end -- which is a mechanism that does not require
    anyone to be wrong, and so has less reason to be arbitraged away than a
    mispricing would.

    Included here mainly because it is a genuinely DIFFERENT kind of signal from
    everything else in the book: it is a calendar rule, not a price
    transformation, so it cannot be correlated with trend or momentum by
    construction. If the multi-strategy premise has anything left in it, this is
    where the diversification would come from.

    It is in the market only about a fifth of the time, so its standalone
    volatility is low and its Sharpe should be read alongside how little risk it
    takes.
    """

    name = "turn_of_month"
    rebalance = "daily"
    warmup_days = 21

    def __init__(self, days_before: int = 1, days_after: int = 3):
        self.days_before = days_before
        self.days_after = days_after

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        index = context.prices.index
        month = pd.Series(index, index=index).dt.to_period("M")

        # Position within the month, counted from each end.
        from_start = month.groupby(month).cumcount()
        from_end = month.groupby(month).cumcount(ascending=False)

        in_window = (from_end < self.days_before) | (from_start < self.days_after)

        weights = pd.DataFrame(0.0, index=index, columns=context.prices.columns)
        weights.loc[in_window.to_numpy(), :] = 1.0
        return weights


def candidate_sleeves() -> list[Sleeve]:
    """The literature-sourced candidates, for the extended-history study."""
    return [MultiHorizonTrend(), BettingAgainstBeta(), TurnOfMonth()]
