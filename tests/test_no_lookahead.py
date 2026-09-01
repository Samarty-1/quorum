"""The tests that decide whether any other number in this repo means anything.

A backtest that leaks future information does not fail, crash, or look wrong.
It looks *excellent*. So the leak has to be tested for directly, and the way to
do that is to change the future and assert the past did not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest
from src.allocators import default_allocators
from src.portfolio import PortfolioConfig, run_portfolio, sleeve_books, sleeve_return_streams
from src.sleeves.base import SleeveContext
from src.sleeves.strategies import all_sleeves, default_sleeves


@pytest.fixture(scope="module")
def synthetic():
    """A deterministic panel long enough to clear every sleeve's warmup."""
    rng = np.random.default_rng(12345)
    dates = pd.bdate_range("2005-01-03", periods=2600)
    assets = [f"A{i}" for i in range(8)]
    steps = rng.normal(0.0003, 0.011, size=(len(dates), len(assets)))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=assets)

    dividend_rows = []
    for asset in assets[:5]:
        for date in dates[::63]:
            dividend_rows.append({"ticker": asset, "date": date, "amount": 0.25})
    dividends = pd.DataFrame(dividend_rows, columns=["ticker", "date", "amount"])
    cash = pd.Series(0.02 / 252.0, index=dates, name="daily_cash")
    return SleeveContext(prices=prices, dividends=dividends, cash_daily=cash)


def _corrupt_future(context: SleeveContext, cutoff: pd.Timestamp) -> SleeveContext:
    """Same history, a completely different future.

    The corruption has to be ASSET-SPECIFIC, and getting this wrong is easy: an
    earlier version scaled every future price by the same factor, which leaves
    every cross-sectional rank untouched. The dollar-neutral sleeves then
    produced byte-identical weights after the cutoff and the test passed while
    testing nothing. So the future is replaced by a fresh random walk with a
    different per-asset drift, which moves levels and orderings together.
    """
    prices = context.prices.copy()
    future = prices.index > cutoff
    n_future = int(future.sum())

    rng = np.random.default_rng(999)
    drifts = np.linspace(-0.0015, 0.0015, prices.shape[1])
    steps = rng.normal(0.0, 0.02, size=(n_future, prices.shape[1])) + drifts
    anchor = prices.loc[~future].iloc[-1].to_numpy()
    prices.loc[future] = anchor * np.exp(np.cumsum(steps, axis=0))

    dividends = context.dividends.copy()
    later = pd.to_datetime(dividends["date"]) > cutoff
    # Per-ticker multipliers, so the carry sleeve's ranking really moves.
    multiplier = {t: 1.0 + 3.0 * i for i, t in enumerate(sorted(dividends["ticker"].unique()))}
    dividends.loc[later, "amount"] = [
        amount * multiplier[ticker]
        for amount, ticker in zip(dividends.loc[later, "amount"], dividends.loc[later, "ticker"])
    ]
    return SleeveContext(prices=prices, dividends=dividends, cash_daily=context.cash_daily)


class TestSleevesDoNotSeeTheFuture:
    @pytest.mark.parametrize("sleeve", all_sleeves(), ids=lambda s: s.name)
    def test_weights_before_the_cutoff_are_unchanged(self, sleeve, synthetic):
        cutoff = synthetic.prices.index[1800]

        original = sleeve.weights(synthetic)
        corrupted = sleeve.weights(_corrupt_future(synthetic, cutoff))

        before = original.index <= cutoff
        pd.testing.assert_frame_equal(original[before], corrupted[before])

        # And the corruption must actually have done something after it, or the
        # test would pass on a sleeve that ignores prices entirely.
        after = original.index > cutoff
        assert not original[after].equals(corrupted[after]), \
            "future corruption changed nothing at all -- the test is not testing"


class TestSleeveInvariants:
    @pytest.mark.parametrize("sleeve", all_sleeves(), ids=lambda s: s.name)
    def test_unit_gross_exposure_when_live(self, sleeve, synthetic):
        weights = sleeve.weights(synthetic)
        gross = weights.abs().sum(axis=1)
        live = gross > 0
        assert live.any(), "sleeve never took a position"
        assert np.allclose(gross[live], 1.0), "sleeve is not unit-gross"

    @pytest.mark.parametrize("sleeve", all_sleeves(), ids=lambda s: s.name)
    def test_warmup_is_flat(self, sleeve, synthetic):
        weights = sleeve.weights(synthetic)
        if sleeve.warmup_days > 0:
            assert (weights.iloc[: sleeve.warmup_days].abs().sum(axis=1) == 0).all()

    @pytest.mark.parametrize("name", ["xs_momentum", "reversal", "carry", "value"])
    def test_cross_sectional_sleeves_are_dollar_neutral(self, name, synthetic):
        """These express relative views and must not smuggle in a market bet."""
        sleeve = next(s for s in all_sleeves() if s.name == name)
        weights = sleeve.weights(synthetic)
        live = weights.abs().sum(axis=1) > 0
        assert np.allclose(weights[live].sum(axis=1), 0.0, atol=1e-9)

    def test_trend_is_allowed_to_be_directional(self, synthetic):
        """The one sleeve that may be net long or short -- that is its job."""
        sleeve = next(s for s in all_sleeves() if s.name == "trend")
        weights = sleeve.weights(synthetic)
        live = weights.abs().sum(axis=1) > 0
        assert np.abs(weights[live].sum(axis=1)).max() > 0.01

    @pytest.mark.parametrize("sleeve", all_sleeves(), ids=lambda s: s.name)
    def test_rebalances_no_more_often_than_declared(self, sleeve, synthetic):
        """A sleeve declaring monthly must not actually trade every day."""
        weights = sleeve.weights(synthetic)
        live = weights[weights.abs().sum(axis=1) > 0]
        changed = (live.diff().abs().sum(axis=1) > 1e-12).sum()
        expected = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}[sleeve.rebalance]
        years = len(live) / 252.0
        # Generous ceiling: the point is to catch a sleeve trading daily when it
        # claims to trade monthly, not to police the calendar exactly.
        assert changed <= expected * years * 1.5 + 5


class TestPortfolioDoesNotSeeTheFuture:
    def test_allocation_before_the_cutoff_is_unchanged(self, synthetic):
        cutoff = synthetic.prices.index[1800]
        asset_returns = synthetic.prices.pct_change().iloc[1:]
        config = PortfolioConfig(lookback_days=252, cost_bps=5.0, target_vol=0.08)

        sleeves = all_sleeves()
        books = sleeve_books(sleeves, synthetic)
        sleeve_returns = sleeve_return_streams(books, asset_returns, 5.0)
        original = run_portfolio(books, sleeve_returns, asset_returns,
                                 default_allocators()[1], config)

        corrupted_context = _corrupt_future(synthetic, cutoff)
        corrupted_returns = corrupted_context.prices.pct_change().iloc[1:]
        corrupted_books = sleeve_books(sleeves, corrupted_context)
        corrupted_streams = sleeve_return_streams(corrupted_books, corrupted_returns, 5.0)
        corrupted = run_portfolio(corrupted_books, corrupted_streams, corrupted_returns,
                                  default_allocators()[1], config)

        before = original.sleeve_weights.index <= cutoff
        pd.testing.assert_frame_equal(
            original.sleeve_weights[before], corrupted.sleeve_weights[before]
        )

    def test_returns_before_the_cutoff_are_unchanged(self, synthetic):
        """Not just the weights -- the realised P&L too, which also depends on
        the volatility scalar and would catch an unlagged risk overlay."""
        cutoff = synthetic.prices.index[1800]
        asset_returns = synthetic.prices.pct_change().iloc[1:]
        config = PortfolioConfig(lookback_days=252, cost_bps=5.0, target_vol=0.08)

        sleeves = all_sleeves()
        books = sleeve_books(sleeves, synthetic)
        streams = sleeve_return_streams(books, asset_returns, 5.0)
        original = run_portfolio(books, streams, asset_returns, default_allocators()[0], config)

        corrupted_context = _corrupt_future(synthetic, cutoff)
        corrupted_returns = corrupted_context.prices.pct_change().iloc[1:]
        corrupted_books = sleeve_books(sleeves, corrupted_context)
        corrupted_streams = sleeve_return_streams(corrupted_books, corrupted_returns, 5.0)
        corrupted = run_portfolio(corrupted_books, corrupted_streams, corrupted_returns,
                                  default_allocators()[0], config)

        before = original.result.returns.index <= cutoff
        pd.testing.assert_series_equal(
            original.result.returns[before], corrupted.result.returns[before]
        )
