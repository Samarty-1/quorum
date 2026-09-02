"""How much capital each sleeve gets.

This is the layer the whole repo exists to test. The sleeves are fixed,
unfitted, published ideas; the only thing that varies between the books below is
how capital is split across them. That isolation is the point -- if a
sophisticated allocator beats equal weight, it is the allocator doing it.

Every allocator sees only trailing data
---------------------------------------
Each is handed a window of sleeve returns ending the day before it allocates,
and returns weights to hold from that day forward. None of them ever sees the
returns they are about to earn. This is the single most important property here,
because an allocator fitted on the full sample is guaranteed to look brilliant
and mean nothing.

The benchmark that usually wins
-------------------------------
Equal weight is not a straw man. DeMiguel, Garlappi and Uppal (2009) found that
across a wide range of datasets, no optimising allocator reliably beat 1/N out
of sample -- the estimation error in the inputs costs more than the optimisation
gains. Every allocator below has to clear that bar, and the study reports
whether it does rather than assuming it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.risk import ledoit_wolf_covariance

TRADING_DAYS = 252


class Allocator(ABC):
    """Maps a trailing window of sleeve returns to sleeve weights."""

    name: str = "allocator"
    #: Minimum observations before the allocator is trusted; below this the
    #: caller falls back to equal weight rather than allocating on noise.
    min_observations: int = 126

    @abstractmethod
    def allocate(self, window: pd.DataFrame) -> np.ndarray:
        """Non-negative weights summing to one, in `window`'s column order."""

    def check_window(self, lookback_days: int) -> None:
        """Refuse a lookback this allocator can never satisfy.

        An allocator whose `min_observations` exceeds the window it will be
        handed falls back on every rebalance date and never runs -- producing
        results identical to the fallback, with no error and no way to tell from
        the output that anything is wrong. It happened once here and cost a
        whole study run to notice.
        """
        if self.min_observations > lookback_days:
            raise ValueError(
                f"{self.name}: min_observations={self.min_observations} exceeds "
                f"lookback_days={lookback_days}, so it would fall back on every "
                f"date and never run. Lower the minimum or lengthen the lookback."
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def _normalise(weights: np.ndarray) -> np.ndarray:
    """Clip negatives and scale to sum one.

    Sleeve weights are long-only by construction: a negative allocation means
    running a strategy backwards, which is a different strategy and would need
    its own justification. Short-term reversal run in reverse is momentum, and
    this book already has two of those.
    """
    w = np.clip(np.nan_to_num(weights, nan=0.0), 0.0, None)
    total = w.sum()
    if total <= 1e-12:
        return np.full_like(w, 1.0 / len(w))
    return w / total


class EqualWeight(Allocator):
    """1/N. The benchmark everything else has to beat."""

    name = "equal_weight"
    min_observations = 0

    def allocate(self, window: pd.DataFrame) -> np.ndarray:
        return np.full(window.shape[1], 1.0 / window.shape[1])


class InverseVolatility(Allocator):
    """Weight inversely to each sleeve's own trailing volatility.

    Uses only the diagonal of the covariance matrix, which is the reason it is
    robust: variances are estimated far more reliably than correlations, so this
    captures most of the available risk balancing while barely exposing itself
    to estimation error. It is the natural first step past equal weight.
    """

    name = "inverse_vol"

    def allocate(self, window: pd.DataFrame) -> np.ndarray:
        vol = window.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS)
        vol = np.where(np.isfinite(vol) & (vol > 1e-8), vol, np.nan)
        if np.all(np.isnan(vol)):
            return EqualWeight().allocate(window)
        return _normalise(np.nan_to_num(1.0 / vol, nan=0.0))


class RiskParity(Allocator):
    """Equalise each sleeve's contribution to portfolio risk.

    Unlike inverse volatility, this accounts for correlations: a sleeve that is
    correlated with the rest gets less than its standalone volatility would
    suggest, because it adds more risk than it appears to. On this book that
    should matter -- trend and cross-sectional momentum are correlated at 0.74,
    and an allocator that cannot see it will double-count them.

    Solved by the standard multiplicative fixed point rather than a general
    optimiser: it preserves non-negativity for free, converges in tens of
    iterations on a problem this size, and cannot wander off into a corner.
    """

    name = "risk_parity"

    def __init__(self, max_iterations: int = 500, tolerance: float = 1e-10):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def allocate(self, window: pd.DataFrame) -> np.ndarray:
        covariance, _ = ledoit_wolf_covariance(window)
        n = covariance.shape[0]
        target = np.full(n, 1.0 / n)

        w = np.full(n, 1.0 / n)
        for _ in range(self.max_iterations):
            marginal = covariance @ w
            if not np.all(np.isfinite(marginal)) or np.any(marginal <= 0):
                return InverseVolatility().allocate(window)
            portfolio_variance = float(w @ marginal)
            if portfolio_variance <= 1e-16:
                return EqualWeight().allocate(window)
            updated = target * portfolio_variance / marginal
            updated = updated / updated.sum()
            if np.max(np.abs(updated - w)) < self.tolerance:
                w = updated
                break
            w = updated
        return _normalise(w)


class MinimumVariance(Allocator):
    """The lowest-variance long-only combination.

    Included as the aggressive end of the spectrum. It uses the full covariance
    matrix, including its most error-prone parts, so it is where estimation
    error should do the most damage -- and where a shrunk covariance should earn
    its keep. If shrinkage is not enough, this allocator will show it.

    Long-only is enforced by projection rather than by a QP solver: solve the
    unconstrained problem, zero the negatives, and re-solve on the survivors.
    Crude, but it terminates, and on a handful of sleeves it reaches the same answer a
    proper active-set method would.
    """

    name = "min_variance"

    def allocate(self, window: pd.DataFrame) -> np.ndarray:
        covariance, _ = ledoit_wolf_covariance(window)
        n = covariance.shape[0]
        active = np.ones(n, dtype=bool)

        for _ in range(n):
            sub = covariance[np.ix_(active, active)]
            ones = np.ones(sub.shape[0])
            try:
                # Ridge term guards a covariance that is singular because two
                # sleeves were nearly identical over the window.
                solution = np.linalg.solve(sub + np.eye(sub.shape[0]) * 1e-10, ones)
            except np.linalg.LinAlgError:
                return InverseVolatility().allocate(window)
            if solution.sum() <= 0:
                return InverseVolatility().allocate(window)
            solution = solution / solution.sum()

            if np.all(solution >= -1e-10):
                weights = np.zeros(n)
                weights[active] = np.clip(solution, 0.0, None)
                return _normalise(weights)

            # Drop the worst violator and re-solve on the rest.
            indices = np.flatnonzero(active)
            active[indices[int(np.argmin(solution))]] = False
            if active.sum() == 0:
                break

        return InverseVolatility().allocate(window)


class TrailingSharpeTilt(Allocator):
    """Fund sleeves in proportion to trailing risk-adjusted performance.

    This is what multi-strategy desks actually do -- cut the pods that are
    losing, add to the ones that are working -- and it is the allocator most
    likely to be wrong. Nothing here establishes that sleeve performance
    persists at this horizon; if it does not, the tilt is chasing noise, and it
    will buy each sleeve after its good run and sell it after its bad one.

    Negative-Sharpe sleeves go to zero rather than negative, so the book stops
    funding a losing sleeve but never runs it backwards. The floor keeps a
    little capital in every sleeve, because a sleeve cut to exactly zero can
    never earn its way back in.
    """

    name = "sharpe_tilt"
    #: Three years, not six months.
    #:
    #: A Sharpe ratio estimated from 126 observations has a standard error of
    #: roughly 1/sqrt(0.5) = 1.41 Sharpe units -- an order of magnitude larger
    #: than the differences between these sleeves. Measured on the six-month
    #: version, allocated weight correlated +0.13 with a sleeve's trailing
    #: returns and -0.10 with its NEXT month: it was buying sleeves after their
    #: good runs, and those runs then reverted. Most sleeves showed a
    #: negative correlation between weight and subsequent return.
    #:
    #: Three years cuts the standard error to about 0.58 and makes the tilt a
    #: slow structural lean rather than a performance chase. It is still the
    #: least defensible allocator here, and the default is now RiskParity.
    min_observations = 504

    def __init__(self, floor: float = 0.05, lookback_days: int = 756):
        self.floor = floor
        self.lookback_days = lookback_days

    def allocate(self, window: pd.DataFrame) -> np.ndarray:
        window = window.tail(self.lookback_days)
        mean = window.mean().to_numpy() * TRADING_DAYS
        vol = window.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(vol > 1e-8, mean / vol, 0.0)

        positive = np.clip(np.nan_to_num(sharpe, nan=0.0), 0.0, None)
        if positive.sum() <= 1e-12:
            return EqualWeight().allocate(window)

        tilted = positive / positive.sum()
        n = len(tilted)
        # Blend toward equal weight so no sleeve is fully defunded.
        return _normalise((1.0 - self.floor * n) * tilted + self.floor)


#: The allocator to reach for absent a reason not to.
#:
#: ERC over minimum variance on stability grounds, measured: p95 monthly weight
#: change 0.091 against 0.173, and mean Herfindahl 0.251 against 0.455 (roughly
#: 4.0 effective sleeves against 2.2). That is structural rather than
#: sample-specific -- ERC depends on the covariance through marginal risk
#: contributions, which are well conditioned, while minimum variance depends on
#: its inverse, which loads on the smallest eigenvalues where estimation error
#: is worst.
#:
#: Read alongside the finding that no optimiser beat 1/N out of sample. The
#: recommendation is "use ERC if you are going to optimise at all", not "ERC
#: adds value over equal weight".
DEFAULT_ALLOCATOR = RiskParity


def default_allocators() -> list[Allocator]:
    """Order is fixed so reports and tests line up."""
    return [EqualWeight(), InverseVolatility(), RiskParity(), MinimumVariance(),
            TrailingSharpeTilt(), EdgeGated(RiskParity())]


class EdgeGated(Allocator):
    """Fund only the sleeves that have shown edge, then allocate over those.

    This is the recommendation the audit ended on, made into a component rather
    than left as a one-off analysis: *establish that a sleeve has edge before
    deciding how much capital it gets*.

    The measurements it exists to answer:

    * Most sleeves lost money out of sample, and no allocation scheme
      rescues a book of negative-edge components. Diversification reduces
      variance; it does not manufacture expected return.
    * Which sleeves were funded moved the confirmation-half Sharpe more than the
      choice of allocator did. The gate operates on the decision that matters.
    * Volatility targeting equalises risk, not quality, and will happily lever a
      losing sleeve toward its target. Something has to decide quality.

    How it differs from :class:`TrailingSharpeTilt`, which chases
    -------------------------------------------------------------
    The tilt allocates *proportionally* to trailing Sharpe, so it responds to
    every wiggle in a statistic whose standard error dwarfs the differences
    between sleeves -- measured at +0.13 correlation with a sleeve's past and
    -0.10 with its future.

    This is a **binary gate on a long window**, not a proportional response on a
    short one. A sleeve is either in or out, the bar is a t-statistic rather than
    a point estimate, and the window is long enough that the statistic means
    something. Crossing the bar changes nothing about how much a sleeve gets --
    that is still the inner allocator's job.

    The t-statistic is Newey-West corrected: a monthly-rebalanced sleeve holds
    the same position for twenty days, so its daily returns are autocorrelated
    and the plain statistic overstates the evidence.

    `min_observations` defaults to 504 to match the engine's default lookback.
    It must not exceed it: an allocator whose minimum is larger than the window
    it is handed falls back on every single date and never runs at all. That
    happened here -- a 756 default against a 504 lookback made this allocator
    silently identical to equal weight, with no error and entirely plausible
    output. :func:`Allocator.check_window` now refuses that configuration.

    Two years is short for a t-statistic on a Sharpe ratio (standard error
    around 0.7 Sharpe units), which is exactly why the default bar is zero
    rather than two: at this window length the gate can only be asked to
    separate "has lost money persistently" from "has not".
    """

    name = "edge_gated"

    def __init__(self, inner: Allocator | None = None, min_t_stat: float = 0.0,
                 min_observations: int = 504):
        self.inner = inner if inner is not None else EqualWeight()
        self.min_t_stat = min_t_stat
        self.min_observations = min_observations
        self.name = f"gated/{self.inner.name}"

    @staticmethod
    def _newey_west_t(series: np.ndarray) -> float:
        n = len(series)
        if n < 30:
            return float("nan")
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        e = series - series.mean()
        variance = float(e @ e / n)
        for lag in range(1, min(lags, n - 1) + 1):
            weight = 1.0 - lag / (lags + 1.0)
            variance += 2.0 * weight * float(e[lag:] @ e[:-lag] / n)
        if not np.isfinite(variance) or variance <= 0:
            return float("nan")
        return float(series.mean() / np.sqrt(variance / n))

    def allocate(self, window: pd.DataFrame) -> np.ndarray:
        n = window.shape[1]
        scores = np.array([self._newey_west_t(window[c].dropna().to_numpy(dtype=float))
                           for c in window.columns])
        keep = np.isfinite(scores) & (scores > self.min_t_stat)

        # Every sleeve rejected. Falling back to funding all of them would be
        # perverse -- the gate has just said none of them has shown edge -- but
        # this allocator cannot hold cash, because its contract is weights that
        # sum to one. So it returns the least-bad book it can and the caller's
        # volatility target does the de-risking. Recorded here because it is a
        # real limitation rather than a design choice.
        if not keep.any():
            return EqualWeight().allocate(window)

        survivors = window.loc[:, keep]
        inner_weights = self.inner.allocate(survivors)

        weights = np.zeros(n)
        weights[keep] = inner_weights
        return _normalise(weights)
