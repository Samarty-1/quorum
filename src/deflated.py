"""What a Sharpe ratio is worth after you account for having looked 25 times.

The study tests 5 sleeves x 5 allocators, plus risk-overlay variants. Reporting
the best cell's Sharpe ratio is reporting the maximum of a couple of dozen
correlated draws, and the expected maximum of N draws from a zero-mean
distribution is emphatically not zero. At N = 25 and this sample length the
inflation is on the order of half a Sharpe unit -- comfortably larger than every
difference the study is trying to detect.

Two corrections, which answer different questions:

* :func:`deflated_sharpe_ratio` (Bailey-Lopez de Prado) -- the probability that
  the observed Sharpe exceeds what the best of N trials would produce under the
  null of no skill. Accounts for the number of trials, and for the skew and
  kurtosis of the returns, which matter because a Sharpe ratio assumes neither.
* :func:`haircut_sharpe` (Harvey-Liu) -- how much of the Sharpe survives a
  multiple-testing adjustment to its p-value. More conservative, and easier to
  explain to someone who does not want a probability.

Both need an honest N. The temptation is to count only the configurations that
made it into the final table; the correct N includes every variant tried and
discarded, which is why the caller has to supply it rather than have it inferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """Expected maximum Sharpe across `n_trials` independent zero-skill trials.

    The standard extreme-value approximation. Note it grows with the SPREAD of
    outcomes across trials, not just their count: twenty-five near-identical
    strategies produce a much lower expected maximum than twenty-five genuinely
    different ones, which is why the trial variance has to be measured rather
    than assumed.
    """
    if n_trials < 2 or trial_variance <= 0:
        return 0.0
    sd = np.sqrt(trial_variance)
    a = (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * (a + b))


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float = 0.0,
                               periods_per_year: int = TRADING_DAYS) -> float:
    """P(true Sharpe > benchmark), adjusting for skew and kurtosis.

    The adjustment is not cosmetic. A Sharpe ratio is a mean over a standard
    deviation, and its own standard error depends on the third and fourth
    moments of the return distribution. Strategies with negative skew and fat
    tails -- which is most systematic strategies, and all short-volatility ones
    -- have a *higher* standard error than the naive formula admits, so their
    Sharpe ratios are less trustworthy than they look.
    """
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 10:
        return float("nan")

    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")

    # Per-period Sharpe, to keep the moment adjustment on the same footing.
    observed = float(r.mean() / sd)
    benchmark = benchmark_sharpe / np.sqrt(periods_per_year)
    skew = float(stats.skew(r))
    kurtosis = float(stats.kurtosis(r, fisher=False))

    denominator = np.sqrt(
        max(1.0 - skew * observed + 0.25 * (kurtosis - 1.0) * observed ** 2, 1e-12)
    )
    return float(stats.norm.cdf((observed - benchmark) * np.sqrt(n - 1) / denominator))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          trial_sharpes: list[float] | np.ndarray | None = None,
                          periods_per_year: int = TRADING_DAYS) -> dict:
    """Bailey-Lopez de Prado deflated Sharpe.

    `trial_sharpes` should be the ANNUALISED Sharpe of every configuration
    tried. Their variance is what sets the bar the observed result has to clear;
    passing them is strongly preferred to letting the function assume a spread.
    """
    r = returns.dropna().astype(float)
    if len(r) < 10:
        return {"n_obs": len(r), "deflated_sharpe": float("nan")}

    annual_sharpe = float(r.mean() * periods_per_year / (r.std(ddof=1) * np.sqrt(periods_per_year)))

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
        variance_assumed = False
    else:
        # Fallback: assume trials are spread about as much as a single Sharpe's
        # own standard error. Reported so nobody mistakes it for measurement.
        variance = 1.0 / (len(r) / periods_per_year)
        variance_assumed = True

    threshold = expected_max_sharpe(n_trials, variance)
    deflated = probabilistic_sharpe_ratio(r, threshold, periods_per_year)

    return {
        "n_obs": int(len(r)),
        "years": round(len(r) / periods_per_year, 2),
        "annual_sharpe": round(annual_sharpe, 4),
        "n_trials": int(n_trials),
        "trial_sharpe_variance": round(variance, 6),
        "trial_variance_assumed": variance_assumed,
        "expected_max_sharpe_under_null": round(threshold, 4),
        "psr_vs_zero": round(probabilistic_sharpe_ratio(r, 0.0, periods_per_year), 4),
        "deflated_sharpe": round(deflated, 4),
        "verdict": (
            "survives deflation" if deflated > 0.95 else
            "does not survive deflation -- consistent with selection from noise"
        ),
    }


def haircut_sharpe(annual_sharpe: float, n_years: float, n_trials: int,
                   method: str = "bhy") -> dict:
    """Harvey-Liu haircut: what fraction of the Sharpe survives the adjustment.

    Converts the Sharpe to a t-statistic, adjusts its p-value for `n_trials`,
    and converts back. `bonferroni` controls the family-wise error rate and is
    brutal; `bhy` (Benjamini-Hochberg-Yekutieli) controls the false discovery
    rate under arbitrary dependence and is the one to quote for correlated
    strategy variants, which these are.
    """
    if n_years <= 0 or annual_sharpe == 0:
        return {"haircut_sharpe": 0.0, "haircut_pct": 100.0}

    t_stat = annual_sharpe * np.sqrt(n_years)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

    if method == "bonferroni":
        adjusted = min(p_value * n_trials, 1.0)
    elif method == "bhy":
        # Benjamini-Yekutieli multiplier c(N) = sum_{i=1..N} 1/i, which is what
        # makes the procedure valid under arbitrary dependence between trials.
        # These trials ARE dependent -- five allocators over the same five
        # sleeves -- so the independence-assuming version would be too lenient.
        # c(25) is about 3.8, against Bonferroni's factor of 25.
        c = float(np.sum(1.0 / np.arange(1, n_trials + 1)))
        adjusted = min(p_value * c, 1.0)
    else:
        raise ValueError(f"unknown method: {method}")

    if adjusted >= 1.0:
        return {"method": method, "observed_p": round(p_value, 6),
                "adjusted_p": 1.0, "haircut_sharpe": 0.0, "haircut_pct": 100.0,
                "significant_at_5pct": False}

    adjusted_t = float(stats.norm.ppf(1.0 - adjusted / 2.0))
    haircut = float(adjusted_t / np.sqrt(n_years)) * np.sign(annual_sharpe)
    return {
        "method": method,
        "observed_t": round(float(t_stat), 4),
        "observed_p": round(p_value, 6),
        "adjusted_p": round(float(adjusted), 6),
        "haircut_sharpe": round(haircut, 4),
        "haircut_pct": round(100.0 * (1.0 - abs(haircut) / abs(annual_sharpe)), 2),
        "significant_at_5pct": bool(adjusted < 0.05),
    }
