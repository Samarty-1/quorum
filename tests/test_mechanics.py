"""Backtest arithmetic, allocators, and the risk layer.

These are the pieces where a quiet error produces a plausible number rather
than an exception -- an off-by-one shift, a cost charged on the wrong day, a
risk-parity solver that returns something that merely looks balanced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest
from src.allocators import (
    EqualWeight,
    InverseVolatility,
    MinimumVariance,
    RiskParity,
    TrailingSharpeTilt,
    default_allocators,
)
from src.risk import (
    diversification_ratio,
    effective_bets,
    ledoit_wolf_covariance,
    volatility_scalar,
)


class TestBacktestArithmetic:
    def test_weight_earns_the_next_days_return(self):
        """The shift, tested directly rather than assumed.

        Fully invested in one asset from day 0. The book earns day 1's return,
        not day 0's -- a weight decided at the close of day 0 cannot capture a
        move that already happened.
        """
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"X": [0.10, 0.05, -0.02, 0.03]}, index=dates)
        weights = pd.DataFrame({"X": [1.0, 1.0, 1.0, 1.0]}, index=dates)

        result = backtest.run(weights, returns, cost_bps=0.0)
        assert result.returns.iloc[0] == pytest.approx(0.0), "day 0's return is not earned"
        assert result.returns.iloc[1] == pytest.approx(0.05)
        assert result.returns.iloc[2] == pytest.approx(-0.02)

    def test_costs_are_charged_on_turnover_at_the_right_time(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"X": [0.0, 0.0, 0.0, 0.0]}, index=dates)
        # Flat, then fully long from day 1: one unit of turnover on day 1.
        weights = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0]}, index=dates)

        result = backtest.run(weights, returns, cost_bps=10.0)
        assert result.turnover.iloc[1] == pytest.approx(1.0)
        # Paid out of the return earned over day 1 -> day 2.
        assert result.returns.iloc[2] == pytest.approx(-10.0 / 10_000.0)
        assert float(result.returns.sum()) == pytest.approx(-10.0 / 10_000.0)

    def test_initial_position_is_not_free(self):
        """Putting the book on from flat is a real trade."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        returns = pd.DataFrame({"X": [0.0, 0.0, 0.0]}, index=dates)
        weights = pd.DataFrame({"X": [1.0, 1.0, 1.0]}, index=dates)
        result = backtest.run(weights, returns, cost_bps=10.0)
        assert result.turnover.iloc[0] == pytest.approx(1.0)

    def test_long_short_book_nets_before_it_trades(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        returns = pd.DataFrame({"X": [0.0, 0.10, 0.0], "Y": [0.0, 0.10, 0.0]}, index=dates)
        weights = pd.DataFrame({"X": [1.0] * 3, "Y": [-1.0] * 3}, index=dates)
        result = backtest.run(weights, returns, cost_bps=0.0)
        assert result.returns.iloc[1] == pytest.approx(0.0), "identical moves must cancel"

    def test_metrics_on_a_known_series(self):
        # Constant daily return: zero volatility, so Sharpe is undefined rather
        # than infinite, and the drawdown is exactly zero.
        dates = pd.bdate_range("2020-01-01", periods=252)
        returns = pd.Series(0.001, index=dates)
        metrics = backtest.performance_metrics(returns)
        assert metrics["max_drawdown"] == pytest.approx(0.0)
        assert metrics["hit_rate"] == pytest.approx(1.0)
        # performance_metrics rounds to 4dp, so compare at that resolution.
        assert metrics["annual_return"] == pytest.approx(round((1.001 ** 252) - 1, 4))


class TestNeweyWest:
    def test_deflates_a_positively_autocorrelated_series(self):
        rng = np.random.default_rng(4)
        # A held position produces exactly this structure: overlapping windows.
        overlapping = pd.Series(rng.normal(0, 1, 3000)).rolling(20).mean().dropna() + 0.05
        naive = overlapping.mean() / (overlapping.std(ddof=1) / np.sqrt(len(overlapping)))
        corrected = backtest.newey_west_sharpe_tstat(overlapping)
        assert overlapping.autocorr(1) > 0.8
        assert abs(corrected) < abs(naive)

    def test_leaves_independent_returns_alone(self):
        rng = np.random.default_rng(5)
        independent = pd.Series(rng.normal(0.0005, 0.01, 3000))
        naive = independent.mean() / (independent.std(ddof=1) / np.sqrt(len(independent)))
        corrected = backtest.newey_west_sharpe_tstat(independent)
        assert corrected == pytest.approx(naive, rel=0.35)


class TestAllocators:
    @pytest.fixture
    def window(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=504)
        data = rng.normal(0.0002, 0.01, size=(len(dates), 4))
        data[:, 1] *= 3.0                       # a much more volatile sleeve
        data[:, 3] = 0.9 * data[:, 0] + 0.1 * data[:, 3]   # nearly a duplicate of sleeve 0
        return pd.DataFrame(data, index=dates, columns=["a", "b", "c", "d"])

    @pytest.mark.parametrize("allocator", default_allocators(), ids=lambda a: a.name)
    def test_weights_are_a_valid_allocation(self, allocator, window):
        weights = allocator.allocate(window)
        assert len(weights) == window.shape[1]
        assert np.all(weights >= -1e-12), "no sleeve may be run backwards"
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(np.isfinite(weights))

    def test_equal_weight_is_equal(self, window):
        assert EqualWeight().allocate(window) == pytest.approx(np.full(4, 0.25))

    def test_inverse_vol_underweights_the_volatile_sleeve(self, window):
        weights = InverseVolatility().allocate(window)
        assert weights[1] < weights[0], "3x volatility should get less capital"
        assert weights[1] < 0.25

    def test_risk_parity_equalises_risk_contributions(self, window):
        weights = RiskParity().allocate(window)
        covariance, _ = ledoit_wolf_covariance(window)
        marginal = covariance @ weights
        contributions = weights * marginal / float(weights @ marginal)
        # The property that defines the allocator, checked directly.
        assert contributions.max() - contributions.min() < 0.02

    def test_risk_parity_sees_correlation_where_inverse_vol_cannot(self, window):
        """Sleeve d is 0.9 correlated with a. Risk parity should fund the pair
        less than inverse volatility does, because together they are one bet."""
        parity = RiskParity().allocate(window)
        inverse = InverseVolatility().allocate(window)
        assert parity[0] + parity[3] < inverse[0] + inverse[3]

    def test_min_variance_prefers_the_quiet_sleeve(self, window):
        weights = MinimumVariance().allocate(window)
        assert weights[1] < 0.15, "the 3x-vol sleeve should be small"

    def test_sharpe_tilt_defunds_a_losing_sleeve(self):
        dates = pd.bdate_range("2020-01-01", periods=504)
        rng = np.random.default_rng(11)
        window = pd.DataFrame({
            "winner": rng.normal(0.0008, 0.008, len(dates)),
            "loser": rng.normal(-0.0008, 0.008, len(dates)),
        }, index=dates)
        weights = TrailingSharpeTilt().allocate(window)
        assert weights[0] > weights[1]
        assert weights[1] > 0.0, "a defunded sleeve must keep a seat to earn back in"

    def test_all_negative_sharpe_falls_back_to_equal_weight(self):
        dates = pd.bdate_range("2020-01-01", periods=504)
        rng = np.random.default_rng(13)
        window = pd.DataFrame({
            "a": rng.normal(-0.001, 0.008, len(dates)),
            "b": rng.normal(-0.001, 0.008, len(dates)),
        }, index=dates)
        weights = TrailingSharpeTilt().allocate(window)
        assert weights == pytest.approx(np.full(2, 0.5))


class TestRiskLayer:
    def test_shrinkage_is_a_valid_intensity_and_matrix(self):
        rng = np.random.default_rng(17)
        window = pd.DataFrame(rng.normal(0, 0.01, size=(300, 5)))
        covariance, shrinkage = ledoit_wolf_covariance(window)
        assert 0.0 <= shrinkage <= 1.0
        assert covariance.shape == (5, 5)
        assert np.allclose(covariance, covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert eigenvalues.min() > -1e-12, "shrunk covariance must stay PSD"

    def test_shrinkage_rises_when_data_is_scarce(self):
        """Fewer observations relative to dimension means trust the sample less.

        The data must have HETEROGENEOUS correlation -- two blocks here -- or
        the test is degenerate. With genuinely independent columns the
        constant-correlation target is already the right answer, so shrinkage
        saturates at 1.0 for every sample size and the comparison is vacuous.
        An earlier version of this test used independent columns and failed for
        exactly that reason, which is a fact about the test, not the estimator.
        """
        rng = np.random.default_rng(23)

        def two_blocks(n):
            first = rng.normal(0, 0.01, size=(n, 1))
            second = rng.normal(0, 0.01, size=(n, 1))
            a = 0.8 * first + 0.6 * rng.normal(0, 0.01, size=(n, 3))
            b = 0.8 * second + 0.6 * rng.normal(0, 0.01, size=(n, 3))
            return pd.DataFrame(np.hstack([a, b]))

        # Averaged over draws: a single 60-observation sample is noisy enough
        # to invert the ordering by chance.
        def mean_shrinkage(n, draws=8):
            return float(np.mean([ledoit_wolf_covariance(two_blocks(n))[1] for _ in range(draws)]))

        assert mean_shrinkage(120) > mean_shrinkage(2000)

    def test_volatility_scalar_hits_the_target_and_respects_the_cap(self):
        assert volatility_scalar(0.16, 0.08) == pytest.approx(0.5)
        assert volatility_scalar(0.04, 0.08) == pytest.approx(2.0)
        assert volatility_scalar(0.001, 0.08, max_leverage=3.0) == pytest.approx(3.0)
        assert volatility_scalar(0.0, 0.08) == pytest.approx(0.0)

    def test_effective_bets_counts_independent_risk(self):
        # Three uncorrelated, equal-variance sleeves are genuinely three bets.
        independent = np.eye(3) * 0.01
        assert effective_bets(np.full(3, 1 / 3), independent) == pytest.approx(3.0, rel=1e-6)

        # Perfectly correlated, they are one -- however evenly the weight is
        # spread. This is the case a marginal-risk-contribution version of the
        # metric gets wrong, scoring 3.
        identical = np.full((3, 3), 0.01)
        assert effective_bets(np.full(3, 1 / 3), identical) == pytest.approx(1.0, rel=0.05)

        # And the in-between case: two of the three move together, so the book
        # holds fewer than three bets but more than one.
        partial = np.array([[0.01, 0.0095, 0.0], [0.0095, 0.01, 0.0], [0.0, 0.0, 0.01]])
        assert 1.5 < effective_bets(np.full(3, 1 / 3), partial) < 2.6

    def test_diversification_ratio_is_one_when_everything_moves_together(self):
        identical = np.full((3, 3), 0.01)
        assert diversification_ratio(np.full(3, 1 / 3), identical) == pytest.approx(1.0, rel=1e-6)
        independent = np.eye(3) * 0.01
        assert diversification_ratio(np.full(3, 1 / 3), independent) > 1.7
