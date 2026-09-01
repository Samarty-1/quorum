"""Regressions for the audit fixes.

Each test corresponds to a defect that was measured on the real book, and
several correspond to bugs introduced *while* fixing something else. Those are
the valuable ones: a vol-targeting layer that quietly quintuples turnover, or a
no-trade band that silently holds the book at zero forever, produce plausible
numbers and no error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest, data, deflated
from src.allocators import TrailingSharpeTilt
from src.portfolio import PortfolioConfig, sleeve_books, vol_target_sleeves
from src.risk import (
    adaptive_throttle,
    asymmetric_volatility,
    ewma_volatility,
    volatility_adjusted_drawdown,
)
from src.sleeves.base import (
    Sleeve,
    SleeveContext,
    apply_no_trade_band,
    neutralise_within_class,
)


class TestAsymmetricVolatility:
    def test_reacts_to_a_spike_faster_than_a_long_window(self):
        """The 126-day window took 40 days to register COVID; 21 took 14."""
        # A 3x regime change, not 8x. A violent enough spike crosses any
        # threshold on any window within days, which tests nothing -- the
        # question is which estimator notices a realistic shift first.
        calm = np.random.default_rng(1).normal(0, 0.005, 500)
        spike = np.random.default_rng(2).normal(0, 0.015, 60)
        series = pd.Series(np.concatenate([calm, spike]))

        fast = asymmetric_volatility(series, 20, 60)
        slow = series.rolling(126, min_periods=40).std() * np.sqrt(252)

        pre = float(fast.iloc[499])
        target = pre * 1.5

        def days_to_react(estimate):
            episode = estimate.iloc[500:]
            reached = episode[episode >= target]
            return len(episode) if reached.empty else int(episode.index.get_loc(reached.index[0]))

        assert days_to_react(fast) < days_to_react(slow)

    def test_delevers_fast_and_relevers_slow(self):
        """The asymmetry, stated as a property rather than assumed."""
        calm_before = np.random.default_rng(3).normal(0, 0.004, 300)
        crisis = np.random.default_rng(4).normal(0, 0.030, 60)
        calm_after = np.random.default_rng(5).normal(0, 0.004, 200)
        series = pd.Series(np.concatenate([calm_before, crisis, calm_after]))

        estimate = asymmetric_volatility(series, 20, 60)
        fast_only = ewma_volatility(series, 20)

        # Entering the crisis the two agree closely -- the fast one leads.
        entry = slice(300, 340)
        assert estimate.iloc[entry].mean() == pytest.approx(fast_only.iloc[entry].mean(), rel=0.15)

        # Leaving it, the asymmetric estimate stays HIGHER: it will not re-lever
        # until the slow estimate has come down too.
        exit_window = slice(370, 450)
        assert estimate.iloc[exit_window].mean() > fast_only.iloc[exit_window].mean()

    def test_never_below_the_fast_estimate(self):
        series = pd.Series(np.random.default_rng(6).normal(0, 0.01, 500))
        estimate = asymmetric_volatility(series, 20, 60)
        assert (estimate.dropna() >= ewma_volatility(series, 20).dropna() - 1e-12).all()


class TestVolatilityAdjustedDrawdown:
    def test_same_percentage_depth_scores_differently_by_regime(self):
        """15% off the high is a 0.75-sigma event at 20% vol and 1.9-sigma at 8%.

        A ladder on raw percentage cannot tell those apart, which is why the
        spec version spent 39.6% of days de-risked.
        """
        equity = pd.Series([1.0, 1.0, 0.85], index=pd.bdate_range("2020-01-01", periods=3))
        quiet = pd.Series(0.08, index=equity.index)
        wild = pd.Series(0.20, index=equity.index)

        calm_depth = volatility_adjusted_drawdown(equity, quiet).iloc[-1]
        wild_depth = volatility_adjusted_drawdown(equity, wild).iloc[-1]
        assert calm_depth > 2.0 * wild_depth

    def test_no_drawdown_is_zero_sigma(self):
        equity = pd.Series([1.0, 1.1, 1.2], index=pd.bdate_range("2020-01-01", periods=3))
        vol = pd.Series(0.10, index=equity.index)
        assert (volatility_adjusted_drawdown(equity, vol) == 0.0).all()


class TestAdaptiveThrottle:
    @pytest.fixture
    def drawdown_path(self):
        """Calm rise, a real drawdown, then a full recovery.

        The first version used a plain random walk for the calm phase, which
        throws off 12% drawdowns of its own accord -- so the throttle was
        correctly cutting during what the test called the quiet period. A
        drawdown test needs a segment with no drawdown in it.
        """
        dates = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(7)
        returns = pd.Series(0.0, index=dates)
        returns.iloc[:150] = np.abs(rng.normal(0.0008, 0.0015, 150))   # no drawdown
        returns.iloc[150:200] = rng.normal(0.0002, 0.004, 50)
        returns.iloc[200:250] = rng.normal(-0.004, 0.005, 50)          # the selloff
        returns.iloc[250:] = np.abs(rng.normal(0.0018, 0.002, 150))    # recovery
        return returns

    def test_cuts_exposure_in_a_drawdown_and_restores_it_after(self, drawdown_path):
        vol = asymmetric_volatility(drawdown_path, 20, 60)
        throttle = adaptive_throttle(drawdown_path, vol)

        assert throttle.iloc[:150].mean() > 0.95, "no cutting before the drawdown"
        assert throttle.iloc[230:260].min() < 0.8, "must cut during the drawdown"

    def test_never_reaches_zero(self, drawdown_path):
        """The spec cut to cash with an undefined recovery trigger. A book at
        zero cannot earn its way back."""
        vol = asymmetric_volatility(drawdown_path, 20, 60)
        throttle = adaptive_throttle(drawdown_path, vol, floor=0.30)
        assert throttle.min() >= 0.30 - 1e-12

    def test_is_continuous_not_stepped(self, drawdown_path):
        """Rungs guarantee a book oscillating around a threshold trades on every
        crossing. A ramp has no boundary to oscillate across."""
        vol = asymmetric_volatility(drawdown_path, 20, 60)
        throttle = adaptive_throttle(drawdown_path, vol)
        distinct = throttle.round(4).nunique()
        assert distinct > 20, f"only {distinct} distinct exposures -- looks stepped"

    def test_reads_the_shadow_book_so_it_cannot_ratchet(self, drawdown_path):
        """Given the SAME shadow returns, the throttle is identical regardless
        of what exposure was actually applied. That is what breaks the feedback
        trap: cutting risk cannot deepen the measured drawdown."""
        vol = asymmetric_volatility(drawdown_path, 20, 60)
        first = adaptive_throttle(drawdown_path, vol)
        # Whatever the book actually earned, the trigger sees the shadow.
        second = adaptive_throttle(drawdown_path, vol)
        pd.testing.assert_series_equal(first, second)
        assert first.iloc[-1] > first.iloc[250], "must recover once the shadow does"


class TestNoTradeBand:
    def test_opening_the_book_is_not_blocked_by_the_band(self):
        """A unit-gross target sits exactly 1.0 from flat, so a band above 1.0
        would otherwise leave the sleeve permanently at zero -- silently, with
        a turnover of exactly nothing to give it away."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        weights = pd.DataFrame({"A": [0.5] * 10, "B": [-0.5] * 10}, index=dates)
        held = apply_no_trade_band(weights, band=1.8)
        assert held.abs().sum(axis=1).iloc[-1] == pytest.approx(1.0)

    def test_holds_through_small_moves_and_trades_on_large_ones(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        weights = pd.DataFrame({
            "A": [1.0, 1.02, 1.04, -1.0],
            "B": [0.0, 0.0, 0.0, 0.0],
        }, index=dates)
        held = apply_no_trade_band(weights, band=0.5)
        assert held["A"].iloc[1] == pytest.approx(1.0), "small drift must not trade"
        assert held["A"].iloc[2] == pytest.approx(1.0)
        assert held["A"].iloc[3] == pytest.approx(-1.0), "a full flip must trade"

    def test_reduces_turnover_on_a_fast_signal(self):
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2020-01-01", periods=500)
        raw = pd.DataFrame(rng.normal(0, 1, (500, 4)), index=dates, columns=list("ABCD"))
        raw = raw.div(raw.abs().sum(axis=1), axis=0)

        unbanded = raw.diff().abs().sum(axis=1).mean()
        banded = apply_no_trade_band(raw, 1.0).diff().abs().sum(axis=1).mean()
        assert banded < unbanded * 0.9

    def test_zero_band_is_a_no_op(self):
        rng = np.random.default_rng(12)
        raw = pd.DataFrame(rng.normal(0, 1, (50, 3)),
                           index=pd.bdate_range("2020-01-01", periods=50), columns=list("ABC"))
        pd.testing.assert_frame_equal(apply_no_trade_band(raw, 0.0), raw)


class TestClassNeutralisation:
    def test_removes_the_asset_class_bet(self):
        """Carry's mean net equity exposure was -0.279 before this, with HYG,
        VNQ and LQD its largest persistent longs -- a macro position wearing a
        yield label."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        # Credit names score structurally high, equities structurally low.
        scores = pd.DataFrame({
            "HYG": 5.0, "LQD": 4.5,          # Credit
            "SPY": 1.0, "QQQ": 0.8,          # Equity
        }, index=dates)
        classes = {"HYG": "Credit", "LQD": "Credit", "SPY": "Equity", "QQQ": "Equity"}

        neutral = neutralise_within_class(scores, classes)
        assert neutral[["HYG", "LQD"]].sum(axis=1).abs().max() < 1e-12
        assert neutral[["SPY", "QQQ"]].sum(axis=1).abs().max() < 1e-12
        # Within class the ordering survives -- that is the signal being kept.
        assert (neutral["HYG"] > neutral["LQD"]).all()
        assert (neutral["SPY"] > neutral["QQQ"]).all()

    def test_single_member_class_is_zeroed_not_left_raw(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        scores = pd.DataFrame({"GLD": 9.0, "SPY": 1.0, "QQQ": 2.0}, index=dates)
        classes = {"GLD": "RealAsset", "SPY": "Equity", "QQQ": "Equity"}
        neutral = neutralise_within_class(scores, classes)
        assert (neutral["GLD"] == 0.0).all(), "a lone member carries no within-class signal"

    def test_no_class_map_is_a_no_op(self):
        scores = pd.DataFrame({"A": [1.0], "B": [2.0]})
        pd.testing.assert_frame_equal(neutralise_within_class(scores, None), scores)


class TestSleeveVolTargeting:
    @pytest.fixture
    def synthetic(self):
        rng = np.random.default_rng(21)
        dates = pd.bdate_range("2010-01-01", periods=1500)
        assets = [f"A{i}" for i in range(6)]
        steps = rng.normal(0.0003, 0.011, size=(len(dates), len(assets)))
        prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=assets)
        return prices, prices.pct_change().iloc[1:]

    class _Constant(Sleeve):
        name = "constant"
        rebalance = "monthly"
        warmup_days = 60

        def raw_weights(self, context):
            frame = pd.DataFrame(0.0, index=context.prices.index, columns=context.prices.columns)
            frame.iloc[:, 0] = 1.0
            return frame

    def test_does_not_add_daily_turnover(self, synthetic):
        """The bug this caught: a vol scalar recomputed daily re-trades the whole
        book daily even when no signal moved. Applied naively it took trend from
        3.4 to 8.3 round trips a year and carry from 0.9 to 5.9 -- pure cost.
        """
        prices, returns = synthetic
        context = SleeveContext(prices=prices, dividends=pd.DataFrame(
            columns=["ticker", "date", "amount"]), cash_daily=pd.Series(0.0, index=prices.index))

        sleeve = self._Constant()
        books = sleeve_books([sleeve], context)
        config = PortfolioConfig(sleeve_target_vol=0.10, cost_bps=5.0)
        scaled = vol_target_sleeves(books, returns, config, [sleeve])

        turnover = backtest.run(scaled["constant"], returns, 0.0).turnover
        # A constant position rescaled monthly trades about twelve times a year,
        # not 250. Anything above ~40 means the scalar is moving daily.
        assert float(turnover.mean() * 252) < 40.0

    def test_equalises_sleeve_volatility(self, synthetic):
        prices, returns = synthetic
        context = SleeveContext(prices=prices, dividends=pd.DataFrame(
            columns=["ticker", "date", "amount"]), cash_daily=pd.Series(0.0, index=prices.index))
        sleeve = self._Constant()
        config = PortfolioConfig(sleeve_target_vol=0.10, cost_bps=5.0)
        scaled = vol_target_sleeves(sleeve_books([sleeve], context), returns, config, [sleeve])

        realised = backtest.run(scaled["constant"], returns, 0.0).returns
        annualised = float(realised.loc[realised.ne(0).idxmax():].std() * np.sqrt(252))
        assert 0.06 < annualised < 0.16, f"expected near 10%, got {annualised:.2%}"

    def test_disabled_returns_the_books_unchanged(self, synthetic):
        prices, returns = synthetic
        context = SleeveContext(prices=prices, dividends=pd.DataFrame(
            columns=["ticker", "date", "amount"]), cash_daily=pd.Series(0.0, index=prices.index))
        sleeve = self._Constant()
        books = sleeve_books([sleeve], context)
        config = PortfolioConfig(sleeve_target_vol=None)
        assert vol_target_sleeves(books, returns, config, [sleeve]) is books


class TestExcessReturnsAndCosts:
    def test_excess_returns_subtract_the_bill(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.0]}, index=dates)
        cash = pd.Series(0.0002, index=dates)
        excess = data.excess_returns(returns, cash)
        assert excess["A"].tolist() == pytest.approx([0.0098, 0.0198, -0.0002])

    def test_costs_widen_with_volatility(self):
        dates = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(31)
        quiet = rng.normal(0, 0.004, 200)
        wild = rng.normal(0, 0.030, 200)
        returns = pd.Series(np.concatenate([quiet, wild]), index=dates)

        schedule = backtest.volatility_scaled_costs(5.0, returns)
        assert schedule.iloc[350] > schedule.iloc[150] * 1.5
        assert schedule.max() <= 5.0 * 4.0 + 1e-9, "cap must bind"

    def test_a_series_cost_is_charged_per_day(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"X": [0.0] * 4}, index=dates)
        weights = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0]}, index=dates)
        schedule = pd.Series([10.0, 10.0, 100.0, 10.0], index=dates)
        result = backtest.run(weights, returns, schedule)
        # One unit of turnover on day 1, paid at day 2's rate of 100bps.
        assert result.returns.iloc[2] == pytest.approx(-100.0 / 10_000.0)


class TestDeflatedSharpe:
    def test_more_trials_raise_the_bar(self):
        rng = np.random.default_rng(41)
        returns = pd.Series(rng.normal(0.0004, 0.008, 2500))
        trials = list(rng.normal(0.0, 0.3, 25))

        few = deflated.deflated_sharpe_ratio(returns, 3, trials)
        many = deflated.deflated_sharpe_ratio(returns, 50, trials)
        assert many["expected_max_sharpe_under_null"] > few["expected_max_sharpe_under_null"]
        assert many["deflated_sharpe"] < few["deflated_sharpe"]

    def test_a_pure_noise_strategy_does_not_survive(self):
        rng = np.random.default_rng(42)
        noise = pd.Series(rng.normal(0.0, 0.01, 2500))
        result = deflated.deflated_sharpe_ratio(noise, 25, list(rng.normal(0, 0.3, 25)))
        assert result["deflated_sharpe"] < 0.95
        assert "does not survive" in result["verdict"]

    def test_haircut_is_harsher_under_bonferroni_than_bhy(self):
        strict = deflated.haircut_sharpe(0.8, 19.0, 25, method="bonferroni")
        lenient = deflated.haircut_sharpe(0.8, 19.0, 25, method="bhy")
        assert strict["haircut_sharpe"] <= lenient["haircut_sharpe"]
        assert 0.0 <= lenient["haircut_sharpe"] <= 0.8

    def test_haircut_reports_full_loss_when_nothing_survives(self):
        result = deflated.haircut_sharpe(0.05, 2.0, 100, method="bonferroni")
        assert result["haircut_sharpe"] == 0.0
        assert result["haircut_pct"] == 100.0
        assert result["significant_at_5pct"] is False


class TestSharpeTiltLookback:
    def test_uses_only_the_configured_window(self):
        """Six months of Sharpe has a standard error of ~1.41 Sharpe units. The
        tilt now looks back three years."""
        assert TrailingSharpeTilt().lookback_days == 756
        assert TrailingSharpeTilt().min_observations == 504

        dates = pd.bdate_range("2015-01-01", periods=2000)
        rng = np.random.default_rng(51)
        window = pd.DataFrame({
            "a": rng.normal(0.0015, 0.008, 2000),
            "b": rng.normal(0.0000, 0.008, 2000),
        }, index=dates)
        # Recent six months reversed: a short lookback would flip the ranking.
        window.iloc[-126:, 0] = rng.normal(-0.002, 0.008, 126)
        window.iloc[-126:, 1] = rng.normal(0.004, 0.008, 126)

        recent = window.tail(126)
        recent_sharpe = (recent.mean() * 252) / (recent.std(ddof=1) * np.sqrt(252))
        assert recent_sharpe["b"] > recent_sharpe["a"], "precondition: six months favours b"

        weights = TrailingSharpeTilt().allocate(window)
        assert weights[0] > weights[1], "three years should outvote six months"


class TestEdgeGate:
    """The audit's top recommendation, made testable.

    Three of five sleeves lost money out of sample and no allocation scheme
    rescued the book. The gate decides *whether* a sleeve is funded; the inner
    allocator still decides how much.
    """

    @staticmethod
    def _window(seed=61, n=1200):
        dates = pd.bdate_range("2016-01-01", periods=n)
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "good": rng.normal(0.0006, 0.007, n),
            "flat": rng.normal(0.0000, 0.007, n),
            "bad": rng.normal(-0.0006, 0.007, n),
        }, index=dates)

    def test_defunds_a_persistently_losing_sleeve_entirely(self):
        from src.allocators import EdgeGated, EqualWeight

        weights = EdgeGated(EqualWeight()).allocate(self._window())
        assert weights[2] == pytest.approx(0.0), "a losing sleeve gets nothing, not a floor"
        assert weights[0] > 0.0

    def test_is_a_binary_gate_not_a_proportional_tilt(self):
        """The distinction from TrailingSharpeTilt, which chases: crossing the
        bar must not change how much a sleeve gets."""
        from src.allocators import EdgeGated, EqualWeight

        window = self._window()
        weights = EdgeGated(EqualWeight()).allocate(window)
        funded = weights[weights > 0]
        assert len(funded) >= 1
        assert funded.max() == pytest.approx(funded.min()), \
            "inner allocator is 1/N, so survivors must be equally funded"

    def test_delegates_sizing_to_the_inner_allocator(self):
        from src.allocators import EdgeGated, InverseVolatility

        dates = pd.bdate_range("2016-01-01", periods=1200)
        rng = np.random.default_rng(62)
        window = pd.DataFrame({
            "calm": rng.normal(0.0006, 0.004, 1200),
            "wild": rng.normal(0.0006, 0.020, 1200),
            "bad": rng.normal(-0.0008, 0.007, 1200),
        }, index=dates)

        weights = EdgeGated(InverseVolatility()).allocate(window)
        assert weights[2] == pytest.approx(0.0), "gate removes the loser"
        assert weights[0] > weights[1], "inner allocator still risk-balances the rest"

    def test_a_higher_bar_funds_fewer_sleeves(self):
        """Only meaningful while at least one sleeve still clears the bar. Above
        that the allocator hits its fallback and funds everything again, which
        is a documented limitation rather than the gate loosening -- so the bar
        here sits between two known t-statistics."""
        from src.allocators import EdgeGated, EqualWeight

        n = 1200
        dates = pd.bdate_range("2016-01-01", periods=n)
        rng = np.random.default_rng(65)
        # Standard error of the mean is 0.007/sqrt(1200) = 2.0e-4, so these
        # drifts put the three sleeves near t = +4, +1 and -4.
        window = pd.DataFrame({
            "strong": rng.normal(8.1e-4, 0.007, n),
            "weak": rng.normal(2.0e-4, 0.007, n),
            "bad": rng.normal(-8.1e-4, 0.007, n),
        }, index=dates)

        lenient = EdgeGated(EqualWeight(), min_t_stat=0.0).allocate(window)
        strict = EdgeGated(EqualWeight(), min_t_stat=2.0).allocate(window)

        assert int((lenient > 1e-12).sum()) == 2, "bar at 0 admits strong and weak"
        assert int((strict > 1e-12).sum()) == 1, "bar at 2 admits only strong"
        assert strict[0] == pytest.approx(1.0)

    def test_falls_back_rather_than_returning_nothing(self):
        """Its contract is weights summing to one, so it cannot hold cash. When
        every sleeve fails the bar it says so by returning 1/N and leaves the
        de-risking to the volatility target."""
        from src.allocators import EdgeGated, EqualWeight

        dates = pd.bdate_range("2016-01-01", periods=1200)
        rng = np.random.default_rng(63)
        all_bad = pd.DataFrame({
            "a": rng.normal(-0.0008, 0.007, 1200),
            "b": rng.normal(-0.0008, 0.007, 1200),
        }, index=dates)
        weights = EdgeGated(EqualWeight()).allocate(all_bad)
        assert weights.sum() == pytest.approx(1.0)
        assert weights == pytest.approx(np.full(2, 0.5))

    def test_uses_a_newey_west_statistic(self):
        """A monthly-rebalanced sleeve holds the same position for twenty days,
        so a plain t-statistic overstates the evidence and the gate would admit
        sleeves it should reject."""
        from src.allocators import EdgeGated

        rng = np.random.default_rng(64)
        overlapping = pd.Series(rng.normal(0, 1, 3000)).rolling(20).mean().dropna() + 0.04
        arr = overlapping.to_numpy(dtype=float)
        naive = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))
        assert abs(EdgeGated._newey_west_t(arr)) < abs(naive)

    def test_a_minimum_larger_than_the_lookback_is_refused(self):
        """The bug this guard exists for: an allocator whose minimum exceeds the
        window it is handed falls back on every date and never runs, producing
        output identical to the fallback with no error."""
        from src.allocators import EdgeGated, EqualWeight

        gate = EdgeGated(EqualWeight(), min_observations=756)
        with pytest.raises(ValueError, match="never run"):
            gate.check_window(504)
        gate.check_window(756)     # exactly enough is fine

    def test_default_gate_fits_the_default_lookback(self):
        from src.allocators import EdgeGated
        from src.portfolio import PortfolioConfig

        EdgeGated().check_window(PortfolioConfig().lookback_days)

    def test_every_default_allocator_fits_the_default_lookback(self):
        from src.allocators import default_allocators
        from src.portfolio import PortfolioConfig

        lookback = PortfolioConfig().lookback_days
        for allocator in default_allocators():
            allocator.check_window(lookback)


class TestRollYield:
    """Real commodity carry, from a front-month and a laddered fund.

    The audit recorded this as impossible on free data -- "DBC's roll is already
    inside its price series and cannot be separated without futures-curve data".
    True of DBC alone; false of a pair on the same underlying.
    """

    @staticmethod
    def _context(front_drift, ladder_drift, n=800):
        dates = pd.bdate_range("2015-01-01", periods=n)
        prices = pd.DataFrame({"DBC": 100.0, "SPY": 100.0}, index=dates)
        auxiliary = pd.DataFrame({
            "USO": 100.0 * np.exp(np.arange(n) * front_drift),
            "USL": 100.0 * np.exp(np.arange(n) * ladder_drift),
        }, index=dates)
        return SleeveContext(prices=prices,
                             dividends=pd.DataFrame(columns=["ticker", "date", "amount"]),
                             cash_daily=pd.Series(0.0, index=dates),
                             auxiliary=auxiliary)

    def test_contango_reads_negative(self):
        """Front-month bleeding against the ladder is contango: negative carry."""
        from src.sleeves.base import roll_yield

        context = self._context(front_drift=-0.0004, ladder_drift=0.0)
        value = roll_yield(context, "USO", "USL").dropna()
        assert value.mean() < 0
        # -0.0004/day compounded over 252 days is about -10% a year.
        assert value.iloc[-1] == pytest.approx(-0.10, abs=0.03)

    def test_backwardation_reads_positive(self):
        from src.sleeves.base import roll_yield

        context = self._context(front_drift=0.0004, ladder_drift=0.0)
        assert roll_yield(context, "USO", "USL").dropna().mean() > 0

    def test_identical_curves_read_zero(self):
        from src.sleeves.base import roll_yield

        context = self._context(front_drift=0.0003, ladder_drift=0.0003)
        assert roll_yield(context, "USO", "USL").dropna().abs().max() < 1e-9

    def test_missing_auxiliary_degrades_quietly(self):
        from src.sleeves.base import roll_yield

        context = self._context(0.0, 0.0)
        bare = SleeveContext(prices=context.prices, dividends=context.dividends,
                             cash_daily=context.cash_daily)
        assert roll_yield(bare, "USO", "USL").empty
        assert roll_yield(context, "NOPE", "USL").empty

    def test_uses_only_trailing_data(self):
        from src.sleeves.base import roll_yield

        context = self._context(-0.0004, 0.0)
        original = roll_yield(context, "USO", "USL")

        corrupted_aux = context.auxiliary.copy()
        cutoff = corrupted_aux.index[500]
        corrupted_aux.loc[corrupted_aux.index > cutoff] *= 3.0
        corrupted = roll_yield(
            SleeveContext(prices=context.prices, dividends=context.dividends,
                          cash_daily=context.cash_daily, auxiliary=corrupted_aux),
            "USO", "USL")

        before = original.index <= cutoff
        pd.testing.assert_series_equal(original[before], corrupted[before])


class TestCarryUsesTheRightMeasurePerAsset:
    def test_physically_backed_bullion_gets_no_reading(self):
        """Ranking gold bottom because it pays no coupon is a statement about
        the data field, not about carry."""
        from src.universe import CARRY_SOURCE

        assert CARRY_SOURCE["GLD"] == "none"
        assert CARRY_SOURCE["SLV"] == "none"
        assert CARRY_SOURCE["DBC"] == "roll"


class TestReversalIsCutFromTheDefaultBook:
    def test_default_book_excludes_it_but_it_remains_importable(self):
        """Best-of-twelve repair variants scored +0.153 on the selection half and
        -0.132 on the confirmation half. A sleeve whose best variant loses money
        out of sample does not belong in a book."""
        from src.sleeves.strategies import all_sleeves, default_sleeves

        default_names = [s.name for s in default_sleeves()]
        every_name = [s.name for s in all_sleeves()]

        assert "reversal" not in default_names
        assert "reversal" in every_name, "the measurement must stay reproducible"
        assert len(default_names) == 4


class TestExtendedUniverseCandidates:
    """The literature-sourced candidates and the 35-year universe."""

    @staticmethod
    def _context(n=1600, seed=71):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2010-01-01", periods=n)
        assets = ["LOWBETA", "HIGHBETA", "MID", "CASHLIKE"]
        common = rng.normal(0.0002, 0.008, n)
        betas = {"LOWBETA": 0.3, "HIGHBETA": 1.8, "MID": 1.0, "CASHLIKE": 0.05}
        steps = np.column_stack([
            betas[a] * common + rng.normal(0.0, 0.003, n) for a in assets
        ])
        prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)),
                              index=dates, columns=assets)
        return SleeveContext(
            prices=prices,
            dividends=pd.DataFrame(columns=["ticker", "date", "amount"]),
            cash_daily=pd.Series(0.0, index=dates),
            asset_class={a: "Equity" for a in assets},
        )

    def test_multi_horizon_trend_trades_MORE_not_less(self):
        """The expected benefit was lower turnover. The opposite is true, and
        the test records the real behaviour rather than the hoped-for one.

        Averaging three sign() signals gives a four-state signal instead of a
        two-state one, so it moves whenever ANY horizon flips and each flip
        retrades a third of the book. The literature's turnover benefit needs a
        continuous signal, which this is not.
        """
        from src.sleeves.strategies import MultiHorizonTrend, TrendFollowing

        context = self._context()

        def turnover(book):
            live = book[book.abs().sum(axis=1) > 0]
            return float(live.diff().abs().sum(axis=1).mean())

        single = turnover(TrendFollowing().weights(context))
        blended = turnover(MultiHorizonTrend().weights(context))
        assert blended > single, "sign-averaging adds intermediate states, so it trades more"

    def test_multi_horizon_book_changes_on_more_days(self):
        """The mechanism behind the extra turnover, asserted where it is visible.

        Not via distinct weight magnitudes -- inverse-volatility scaling and
        gross normalisation give both books hundreds of those regardless. What
        differs is how OFTEN the book moves: the blend moves whenever any one of
        its three horizons flips.
        """
        from src.sleeves.strategies import MultiHorizonTrend, TrendFollowing

        context = self._context()

        def change_days(book):
            live = book[book.abs().sum(axis=1) > 0]
            return int((live.diff().abs().sum(axis=1) > 1e-9).sum())

        assert change_days(MultiHorizonTrend().weights(context)) >= \
            change_days(TrendFollowing().weights(context))

    def test_bab_is_beta_neutral_by_construction(self):
        """Skipping the beta-matching produces a permanently net-short-beta
        book, which is a different and worse strategy."""
        from src.sleeves.strategies import BettingAgainstBeta

        context = self._context()
        book = BettingAgainstBeta().weights(context)
        live = book[book.abs().sum(axis=1) > 0]
        assert len(live) > 100

        returns = context.prices.pct_change()
        market = returns.mean(axis=1)
        # Shift the FULL book, not the filtered subset: `live` has a
        # non-contiguous index, so shift(1) on it moves by row position across
        # gaps and silently misaligns the position with the return it earned.
        strategy = (book.shift(1) * returns).sum(axis=1)
        both = pd.concat([strategy, market], axis=1).dropna()
        both = both[both.iloc[:, 0] != 0.0]
        beta = float(np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / np.var(both.iloc[:, 1]))
        assert abs(beta) < 0.35, f"BAB should be near beta-neutral, got {beta:.2f}"

    def test_bab_is_long_the_low_beta_asset(self):
        from src.sleeves.strategies import BettingAgainstBeta

        book = BettingAgainstBeta().weights(self._context())
        live = book[book.abs().sum(axis=1) > 0]
        assert live["LOWBETA"].mean() > 0
        assert live["HIGHBETA"].mean() < 0

    def test_turn_of_month_is_in_the_market_about_a_fifth_of_the_time(self):
        from src.sleeves.strategies import TurnOfMonth

        book = TurnOfMonth().weights(self._context())
        after_warmup = book.iloc[TurnOfMonth().warmup_days:]
        share = float((after_warmup.abs().sum(axis=1) > 0).mean())
        # 4 days of a ~21-day month.
        assert 0.12 < share < 0.30, f"in the market {share:.1%} of days"

    def test_turn_of_month_holds_the_month_boundary(self):
        from src.sleeves.strategies import TurnOfMonth

        book = TurnOfMonth().weights(self._context())
        held = book.abs().sum(axis=1) > 0
        month_end = held.index.to_series().groupby(
            held.index.to_period("M")).transform("max") == held.index
        # Every month-end day after the warmup must be held.
        tail = held.iloc[TurnOfMonth().warmup_days:]
        assert tail[month_end.iloc[TurnOfMonth().warmup_days:]].all()

    def test_extended_universe_covers_the_pristine_period(self):
        from src import extended

        assert extended.EXTENDED_START < extended.ETF_ERA_START
        assert "VTRIX" in extended.STALE_NAV, "stale-NAV funds must be flagged"
        assert len(extended.EXTENDED_UNIVERSE) >= 10
        classes = set(extended.asset_class_map().values())
        assert {"Equity", "Rates", "Credit", "RealAsset"} <= classes


class TestBreadthDoesNotAutomaticallyHelp:
    """More markets is not more diversification, and the code should say so.

    Adding 28 markets to the trend book took effective independent bets from
    1.35 down to 1.06 and the Sharpe from 0.614 to 0.580, because 18 of the
    additions were US equity sectors that all trend together.
    """

    def test_wide_universe_costs_no_history(self):
        from src import extended

        wide = extended.wide_universe()
        assert len(wide) > len(extended.EXTENDED_UNIVERSE)
        # The core funds must all still be there.
        assert set(extended.EXTENDED_UNIVERSE) <= set(wide)

    def test_liquidated_fund_is_excluded(self):
        """VCVSX was wound up in 2021. Including it truncates the whole panel."""
        from src import extended

        assert "VCVSX" not in extended.wide_universe()

    def test_effective_bets_falls_when_correlated_markets_are_added(self):
        """The mechanism, on synthetic data: bolting a block of near-identical
        markets onto a diversified set lowers the effective bet count even
        though the market count rises."""
        from src.risk import effective_bets, ledoit_wolf_covariance

        rng = np.random.default_rng(81)
        n = 1500
        diversified = rng.normal(0, 0.01, size=(n, 4))
        sector_driver = rng.normal(0, 0.01, size=(n, 1))
        sectors = sector_driver + rng.normal(0, 0.002, size=(n, 12))

        narrow = pd.DataFrame(diversified)
        wide = pd.DataFrame(np.hstack([diversified, sectors]))

        def bets(frame):
            covariance, _ = ledoit_wolf_covariance(frame)
            equal = np.full(frame.shape[1], 1.0 / frame.shape[1])
            return effective_bets(equal, covariance)

        assert wide.shape[1] > narrow.shape[1], "precondition: more markets"
        assert bets(wide) < bets(narrow), "but fewer independent bets"
