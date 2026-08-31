"""The risk layer -- the item at the top of the list, and the reason the rest works.

Position sizing is not a strategy, it is the constraint every strategy runs
inside. Three pieces here, in the order they matter.

**Covariance estimation.** Every allocator below the equal-weight one needs a
covariance matrix, and the sample covariance of five sleeves over a two-year
window is estimated from about 500 observations for 15 free parameters. That is
enough to be badly wrong in a specific way: the sample matrix systematically
overstates the spread of its own eigenvalues, so the smallest-variance direction
looks smaller than it is, and any optimiser that minimises variance piles into
it. Ledoit-Wolf shrinkage pulls the matrix toward a structured target by an
amount derived from the data rather than chosen, which is why it is here rather
than a hand-set shrinkage constant.

**Volatility targeting.** Scale the book so its forecast volatility hits a
target. This is what makes results comparable -- two allocators that differ only
in leverage are not different allocators -- and it is what stops a book from
quietly tripling its risk when correlations fall.

**Drawdown throttle.** Cut exposure after losses. Included because it is
standard practice and widely believed to help, and because this repo is in a
position to measure whether it does. It is not obviously a good idea: a throttle
that de-risks into a trough and re-risks after the recovery converts a drawdown
into a permanent loss.

Two estimators, on purpose
--------------------------
The covariance matrix and the volatility scalar are estimated on deliberately
different windows, because they are answering different questions and have
different error properties.

* **Correlation structure** comes from :func:`ledoit_wolf_covariance` on the
  allocator's long window (two years by default). Correlations are more stable
  than variances and genuinely benefit from the extra data; a short window
  produces a correlation matrix dominated by noise, and every allocator past
  equal weight inverts or iterates on that matrix.
* **The scaling decision** comes from :func:`asymmetric_volatility`, a blend of
  20- and 60-day EWMAs. Measured on this book, a 126-day rolling window took 22
  trading days to register the GFC, 40 to register COVID, and never registered
  Volmageddon at all -- because one observation is 1/126 of the estimate, so a
  single 5-sigma day moves it under 4%.

Using one window for both is the mistake: long enough for the correlations is
far too slow for the leverage, and short enough for the leverage makes the
correlations unusable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward a constant-correlation target.

    Returns the shrunk matrix and the shrinkage intensity actually applied, so a
    caller can see how much the data was trusted. An intensity near 1 means the
    sample matrix was judged nearly worthless, which on a short window it often
    is.

    Target is constant-correlation (Ledoit-Wolf 2004) rather than the identity:
    sleeve returns genuinely are correlated, and shrinking toward a matrix that
    says otherwise would trade one bias for a worse one.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n, p = x.shape
    if n < 3 or p < 2:
        return np.cov(x, rowvar=False, ddof=1).reshape(p, p), 1.0

    x = x - x.mean(axis=0)
    sample = (x.T @ x) / n

    variances = np.diag(sample).copy()
    std = np.sqrt(np.maximum(variances, 1e-300))
    outer_std = np.outer(std, std)
    correlations = sample / np.where(outer_std > 0, outer_std, 1.0)
    off_diagonal = correlations[~np.eye(p, dtype=bool)]
    mean_correlation = float(off_diagonal.mean()) if off_diagonal.size else 0.0

    target = mean_correlation * outer_std
    np.fill_diagonal(target, variances)

    # pi: sum of asymptotic variances of the sample covariance entries.
    x_squared = x ** 2
    pi_matrix = (x_squared.T @ x_squared) / n - sample ** 2
    pi = float(pi_matrix.sum())

    # rho: covariance between the sample entries and the target's estimation
    # error. The diagonal contributes directly; the off-diagonal through the
    # constant-correlation term.
    term = ((x ** 3).T @ x) / n - variances[:, None] * sample
    np.fill_diagonal(term, 0.0)
    rho = float(np.diag(pi_matrix).sum()) + mean_correlation * float(
        ((std[None, :] / np.where(std[:, None] > 0, std[:, None], 1.0)) * term).sum()
    )

    gamma = float(((target - sample) ** 2).sum())
    if gamma <= 0:
        return sample, 0.0

    kappa = (pi - rho) / gamma
    shrinkage = float(np.clip(kappa / n, 0.0, 1.0))
    shrunk = shrinkage * target + (1.0 - shrinkage) * sample

    # Annualise: the inputs are daily.
    return shrunk * TRADING_DAYS, shrinkage


def forecast_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Annualised volatility of a weighted combination, from an annualised cov."""
    variance = float(weights @ covariance @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def volatility_scalar(realised_vol: float, target_vol: float,
                      max_leverage: float = 3.0) -> float:
    """How much to scale a book to hit `target_vol`.

    Capped, because the scalar is 1/vol and a quiet estimation window sends it
    to infinity. The cap is a real constraint a desk would have, and leaving it
    out is how a backtest ends up 30x levered into a volatility regime change.
    """
    if not np.isfinite(realised_vol) or realised_vol <= 1e-8:
        return 0.0
    return float(min(target_vol / realised_vol, max_leverage))


def realised_volatility(returns: pd.Series, window: int = 63,
                        min_periods: int = 21) -> pd.Series:
    """Trailing annualised volatility, using data up to and including each day."""
    return returns.rolling(window, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS)


def ewma_volatility(returns: pd.Series, halflife: int = 20,
                    min_periods: int = 20) -> pd.Series:
    """Exponentially weighted annualised volatility.

    A rolling window weights a 126-day-old observation exactly as much as
    yesterday's and then drops it off a cliff. For the *scaling* decision that
    is the wrong shape: a 126-day window moves less than 4% on a single 5-sigma
    day, because one observation is 1/126 of the estimate.

    Measured on this book, days for the estimate to register a 50% rise in
    volatility: GFC 22 days on a 126-day window against 11 on a 21-day one;
    COVID 40 against 14; Volmageddon never registered at all on 126 days. The
    window length was not a detail, it was the difference between reacting
    inside a crisis and reacting after it.
    """
    return returns.ewm(halflife=halflife, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS)


def asymmetric_volatility(returns: pd.Series, fast_halflife: int = 20,
                          slow_halflife: int = 60, min_periods: int = 20) -> pd.Series:
    """De-lever fast, re-lever slow.

    The maximum of a fast and a slow EWMA. When volatility spikes the fast
    estimate rises first and the max follows it immediately, so the book
    de-levers inside days. When volatility subsides the fast estimate falls
    first but the slow one has not yet, so the max stays high and the book
    re-levers only once both agree.

    The asymmetry is deliberate and not merely conservative. Volatility
    clusters: a quiet week after a crisis is far more likely to be followed by
    another violent one than the fast estimate alone implies. Re-levering on
    the fast estimate is what turns one drawdown into two.
    """
    fast = ewma_volatility(returns, fast_halflife, min_periods)
    slow = ewma_volatility(returns, slow_halflife, min_periods)
    return pd.concat([fast, slow], axis=1).max(axis=1).rename("vol")


def volatility_adjusted_drawdown(equity: pd.Series, volatility: pd.Series,
                                 horizon_days: int = 63,
                                 floor_vol: float = 0.02) -> pd.Series:
    """Drawdown depth expressed in standard deviations, not percent.

    A 15% drawdown in a 20%-vol regime is a 0.75-sigma event and carries almost
    no information; the same 15% at 8% vol is a 1.9-sigma event and does. A
    ladder on raw percentage depth cannot tell those apart, so it de-risks on
    ordinary noise in a volatile regime and fails to de-risk on a genuine
    breakdown in a calm one.

    On this book the raw ladder spent 39.6% of days de-risked and cost 3.9
    percentage points through 2022 -- a year the book finished up 13%. Almost
    all of that was drawdown that was unremarkable relative to the volatility
    of the moment.
    """
    depth = -(equity / equity.cummax() - 1.0).clip(upper=0.0)
    scale = volatility.reindex(depth.index).clip(lower=floor_vol) * np.sqrt(
        horizon_days / TRADING_DAYS
    )
    return (depth / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("dd_sigma")


def adaptive_throttle(shadow_returns: pd.Series, volatility: pd.Series,
                      start_sigma: float = 1.5, stop_sigma: float = 3.0,
                      floor: float = 0.30, horizon_days: int = 63) -> pd.Series:
    """Continuous de-risking on volatility-adjusted drawdown.

    Four changes from a stepped ladder on raw drawdown, each fixing a specific
    failure found by measuring the original:

    **Measured on the SHADOW book.** `shadow_returns` is the unthrottled
    strategy. A ladder that reads its own throttled equity curve is a ratchet:
    cutting risk slows the recovery, which keeps the drawdown deep, which keeps
    the book cut. Reading the shadow book means the trigger stays responsive
    while de-risked, and re-entry needs no timer -- when the shadow recovers,
    depth falls and exposure rises on its own.

    **Volatility-adjusted depth**, per :func:`volatility_adjusted_drawdown`.

    **Continuous, not stepped.** Rungs at 10/15/20% guarantee that a book
    oscillating around a threshold trades on every crossing, paying costs for
    nothing. A smooth ramp has no boundary to oscillate across.

    **A floor, never zero.** The audited spec cut to cash with an undefined
    20-day recovery trigger. That rung never fired in nineteen years -- because
    the shallower rungs had already cut risk -- so the most dangerous rule in
    the system was untested rather than proven safe. A book at zero exposure
    cannot earn its way back, and a state-dependent re-entry that never fires
    keeps it there indefinitely.
    """
    equity = (1.0 + shadow_returns.fillna(0.0)).cumprod()
    depth_sigma = volatility_adjusted_drawdown(equity, volatility, horizon_days)

    span = max(stop_sigma - start_sigma, 1e-9)
    fraction = ((depth_sigma - start_sigma) / span).clip(lower=0.0, upper=1.0)
    return (1.0 - (1.0 - floor) * fraction).rename("throttle")


def diversification_ratio(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Weighted average of component vols, over the vol of the combination.

    One if everything moves together, larger the more the components offset.
    This is the number a multi-strategy book is sold on, and the point of
    measuring it separately from the Sharpe ratio is that it can be high while
    the book still loses money -- diversification is not edge.
    """
    component_vols = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    weighted = float(np.abs(weights) @ component_vols)
    portfolio_vol = forecast_volatility(weights, covariance)
    if portfolio_vol <= 1e-12:
        return float("nan")
    return weighted / portfolio_vol


def effective_bets(weights: np.ndarray, covariance: np.ndarray) -> float:
    """How many genuinely independent bets the book holds (Meucci).

    This has to be measured in a basis where the components are uncorrelated,
    and getting that wrong is easy. The obvious implementation -- entropy of
    each sleeve's marginal risk contribution, in the original basis -- answers a
    different question: how evenly is risk spread across the *names* on the
    book. Five perfectly correlated sleeves at equal weight each contribute a
    fifth of the risk, so that version scores 5. They are one bet.

    So the covariance is diagonalised first, the portfolio is expressed as
    exposures to the resulting uncorrelated factors, and the entropy is taken
    over how much variance each factor supplies:

        Sigma = E L E'      v = E'w      p_k = v_k^2 * l_k / (w' Sigma w)
        ENB   = exp(-sum p_k log p_k)

    Perfectly correlated components give one non-zero eigenvalue and ENB = 1;
    uncorrelated equal-variance components at equal weight give ENB = n. The
    principal-component basis is one of several defensible choices (Meucci's
    minimum-torsion basis is another); it is used here because it needs no
    optimisation and the ranking it produces is what the study relies on.
    """
    portfolio_variance = float(weights @ covariance @ weights)
    if portfolio_variance <= 1e-16 or not np.isfinite(portfolio_variance):
        return float("nan")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    exposures = eigenvectors.T @ weights
    contributions = (exposures ** 2) * np.clip(eigenvalues, 0.0, None) / portfolio_variance

    positive = contributions[contributions > 1e-12]
    if positive.size == 0:
        return float("nan")
    positive = positive / positive.sum()
    return float(np.exp(-(positive * np.log(positive)).sum()))
