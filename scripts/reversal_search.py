"""Can the reversal sleeve be repaired, or should it be cut?

A search, run with the discipline a search needs. Twelve variants are scored on
the SELECTION half only; the single best is then scored once on the confirmation
half. If it does not survive, the sleeve is cut from the default book rather than
kept because a variant of it looked good on the half that chose it.

The count matters and is reported: twelve variants is twelve more trials on top
of the study's existing thirty, and the deflation in run_study.py has to know.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import backtest, data
from src.sleeves.base import Sleeve, SleeveContext, rank_scores
from src.universe import UNIVERSE

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


class ReversalVariant(Sleeve):
    """Short-term reversal with the two knobs worth testing.

    `top_fraction` concentrates the book into the extremes: the audit found the
    signal's information is in the tails, and ranking the whole cross-section
    spends turnover on names the signal has no view on.
    """

    name = "reversal"
    rebalance = "weekly"
    warmup_days = 60

    def __init__(self, lookback_days: int = 5, top_fraction: float | None = None,
                 band: float = 0.0):
        self.lookback_days = lookback_days
        self.top_fraction = top_fraction
        self.no_trade_band = band

    def raw_weights(self, context: SleeveContext) -> pd.DataFrame:
        prices = context.prices
        scores = rank_scores(-(prices / prices.shift(self.lookback_days) - 1.0))
        if self.top_fraction is None:
            return scores
        # Keep only the extremes; everything between them goes flat.
        lower = scores.quantile(self.top_fraction, axis=1)
        upper = scores.quantile(1.0 - self.top_fraction, axis=1)
        keep = scores.le(lower, axis=0) | scores.ge(upper, axis=0)
        return scores.where(keep, 0.0)


def main() -> None:
    prices, dividends, cash, auxiliary = data.load(allow_fetch=False)
    prices = data.common_sample(prices)
    cash_daily = data.daily_cash_rate(cash, prices.index)
    asset_returns = data.excess_returns(data.daily_returns(prices), cash_daily)
    context = SleeveContext(prices=prices, dividends=dividends, cash_daily=cash_daily,
                            asset_class={t: m[1] for t, m in UNIVERSE.items()},
                            auxiliary=auxiliary)

    index = asset_returns.index
    cut = index[int(len(index) * 0.5)]
    selection = index[index <= cut]
    confirmation = index[index > cut]

    market = asset_returns.mean(axis=1)
    costs = backtest.volatility_scaled_costs(5.0, market)

    print(f"selection    {selection[0].date()} .. {selection[-1].date()}")
    print(f"confirmation {confirmation[0].date()} .. {confirmation[-1].date()}")
    print("\nSELECTION HALF ONLY -- the confirmation half is not looked at until the end\n")

    rows = []
    for lookback in (3, 5, 10):
        for fraction in (None, 0.33):
            for band in (0.0, 1.0):
                sleeve = ReversalVariant(lookback, fraction, band)
                book = sleeve.weights(context)
                net = backtest.run(book, asset_returns, costs)
                gross = backtest.run(book, asset_returns, 0.0)
                window = net.returns.reindex(selection).dropna()
                rows.append({
                    "lookback": lookback,
                    "concentration": "full" if fraction is None else "tercile",
                    "band": band,
                    "sel_gross": round(backtest.sharpe_of(
                        gross.returns.reindex(selection).dropna()), 3),
                    "sel_net": round(backtest.sharpe_of(window), 3),
                    "sel_t": round(backtest.newey_west_sharpe_tstat(window), 2),
                    "turnover": round(float(net.turnover.reindex(selection).mean() * 252), 1),
                })

    table = pd.DataFrame(rows).sort_values("sel_net", ascending=False)
    print(table.to_string(index=False))

    best = table.iloc[0]
    print(f"\nBest on the selection half: lookback={best['lookback']}, "
          f"{best['concentration']}, band={best['band']} -> net {best['sel_net']}")
    print(f"Variants searched: {len(table)}")

    sleeve = ReversalVariant(int(best["lookback"]),
                             None if best["concentration"] == "full" else 0.33,
                             float(best["band"]))
    net = backtest.run(sleeve.weights(context), asset_returns, costs)
    held_back = net.returns.reindex(confirmation).dropna()

    print("\nCONFIRMATION HALF -- scored once")
    print(f"  net Sharpe {backtest.sharpe_of(held_back):+.3f}   "
          f"t {backtest.newey_west_sharpe_tstat(held_back):+.2f}   "
          f"turnover {float(net.turnover.reindex(confirmation).mean() * 252):.1f}/yr")

    survived = backtest.sharpe_of(held_back) > 0
    print(f"\n  VERDICT: {'keep the repaired sleeve' if survived else 'CUT -- the best variant did not survive'}")


if __name__ == "__main__":
    main()
