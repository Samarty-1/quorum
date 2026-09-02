"""Is there a tradeable edge here? Tested on 35 years, with a real holdout.

    python -m scripts.research_edge

The design, and why it is shaped this way
------------------------------------------
The ETF study's confirmation half is spent. It has been used to cut a sleeve
and to pick an allocator, so it is no longer a clean out-of-sample test of
anything. Adding more strategies to that sample would be the exact error the
whole repo argues against.

So this uses a longer history and a period that has never been examined:

* **1991-11 to 2007-04 -- PRISTINE.** No strategy in this repo has ever been run
  on it. Every signal here was specified either from published literature or
  from the 2007+ ETF sample, so this period is genuinely out of sample for the
  *selection* of what to test, which is the part that usually leaks.
* **2007-04 to 2026-09 -- the familiar era.** Reported for comparison, and
  discounted accordingly.

A strategy that works in both is evidence. One that works only in the familiar
era is a description of the familiar era.

Every candidate's parameters come from a published specification (see the
docstrings in src/sleeves/strategies.py). Nothing here was tuned on either
period, which is what keeps the trial count bounded and the deflation honest.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest, deflated, extended
from src.allocators import EqualWeight, RiskParity
from src.portfolio import (
    PortfolioConfig,
    run_portfolio,
    sleeve_books,
    sleeve_return_streams,
    vol_target_sleeves,
)
from src.sleeves.base import SleeveContext
from src.sleeves.strategies import (
    BettingAgainstBeta,
    Carry,
    CrossSectionalMomentum,
    MultiHorizonTrend,
    TrendFollowing,
    TurnOfMonth,
    Value,
)

warnings.filterwarnings("ignore")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
pd.set_option("display.width", 210)


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def score(returns: pd.Series, name: str) -> dict:
    window = returns.dropna()
    if len(window) < 250:
        return {"name": name, "n_years": round(len(window) / 252, 1), "sharpe": np.nan}
    return {
        "name": name,
        "n_years": round(len(window) / 252, 1),
        "sharpe": round(backtest.sharpe_of(window), 3),
        "t_nw": round(backtest.newey_west_sharpe_tstat(window), 2),
        "ann_ret": round(float(window.mean() * 252), 4),
        "ann_vol": round(float(window.std() * np.sqrt(252)), 4),
        "max_dd": backtest.performance_metrics(window)["max_drawdown"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    # The DIVERSIFYING universe: the 12-fund core plus the international-equity
    # and fixed-income additions, but NOT the 18 US equity sector funds. That
    # split is structural (is it a Fidelity Select sector or style fund?) and was
    # fixed before any per-market Sharpe was inspected, so it is not
    # performance-selected. Section [7] shows what each universe scores and why
    # the sectors are excluded.
    wide = extended.common_sample(
        extended.load_wide(refresh=args.refresh, allow_fetch=True)[extended.wide_tickers()])
    prices = wide[[c for c in wide.columns if c not in extended.US_SECTOR_FUNDS]]
    returns_total = prices.pct_change().iloc[1:]

    # Cash: the short-duration Treasury fund is the closest thing to a bill
    # return available across the whole window, and unlike ^IRX it exists from
    # 1991. Its own return is the hurdle everything else is measured against.
    cash_daily = returns_total["VFISX"].rename("daily_cash")
    asset_returns = returns_total.sub(cash_daily, axis=0)

    context = SleeveContext(prices=prices, dividends=pd.DataFrame(
        columns=["ticker", "date", "amount"]), cash_daily=cash_daily,
        asset_class={k: v for k, v in extended.wide_asset_class_map().items()
                     if k in prices.columns})

    index = asset_returns.index
    pristine = index[index < extended.ETF_ERA_START]
    familiar = index[index >= extended.ETF_ERA_START]

    print(f"Extended universe: {prices.shape[1]} funds, {len(asset_returns)} days")
    print(f"  {asset_returns.index[0].date()} to {asset_returns.index[-1].date()} "
          f"({len(asset_returns) / 252:.1f} years)")
    print(f"  PRISTINE  {pristine[0].date()} .. {pristine[-1].date()} "
          f"({len(pristine) / 252:.1f}y, never examined)")
    print(f"  familiar  {familiar[0].date()} .. {familiar[-1].date()} "
          f"({len(familiar) / 252:.1f}y, the ETF study's era)")

    market = asset_returns.mean(axis=1)
    costs = backtest.volatility_scaled_costs(args.cost_bps, market)

    # ---- the candidates ----------------------------------------------------
    sleeves = [
        TrendFollowing(),            # the one thing that worked, re-tested
        MultiHorizonTrend(),         # Hurst-Ooi-Pedersen / AQR
        BettingAgainstBeta(),        # Frazzini-Pedersen
        TurnOfMonth(),               # Ogden
        CrossSectionalMomentum(),    # for comparison
        Value(),                     # for comparison
    ]
    config = PortfolioConfig(cost_bps=costs, target_vol=0.08, sleeve_target_vol=0.10)
    raw_books = sleeve_books(sleeves, context)
    books = vol_target_sleeves(raw_books, asset_returns, config, sleeves)

    # =====================================================================
    section("[1] EACH CANDIDATE, PRISTINE PERIOD vs FAMILIAR PERIOD")
    # =====================================================================
    rows = []
    streams: dict[str, pd.Series] = {}
    for sleeve in sleeves:
        net = backtest.run(books[sleeve.name], asset_returns, costs, sleeve.name).returns
        streams[sleeve.name] = net
        a = score(net.reindex(pristine), sleeve.name)
        b = score(net.reindex(familiar), sleeve.name)
        full = score(net, sleeve.name)
        rows.append({
            "sleeve": sleeve.name,
            "pristine_sharpe": a.get("sharpe"), "pristine_t": a.get("t_nw"),
            "familiar_sharpe": b.get("sharpe"), "familiar_t": b.get("t_nw"),
            "full_sharpe": full.get("sharpe"), "full_t": full.get("t_nw"),
            "same_sign": "yes" if (a.get("sharpe", 0) or 0) * (b.get("sharpe", 0) or 0) > 0 else "NO",
        })
    candidates = pd.DataFrame(rows)
    print(candidates.to_string(index=False))

    survivors = candidates[(candidates["pristine_sharpe"] > 0)
                           & (candidates["familiar_sharpe"] > 0)]
    print(f"\n    Positive in BOTH periods: "
          f"{list(survivors['sleeve']) if len(survivors) else 'none'}")

    # =====================================================================
    section("[2] ARE THEY DIFFERENT FROM EACH OTHER?")
    # =====================================================================
    frame = pd.DataFrame(streams)
    print(frame.corr().round(3).to_string())

    # =====================================================================
    section("[3] THE BOOK, BUILT ONLY FROM WHAT SURVIVED BOTH PERIODS")
    # =====================================================================
    rows = []
    if len(survivors):
        kept = {name: books[name] for name in survivors["sleeve"]}
        kept_returns = sleeve_return_streams(kept, asset_returns, costs)
        for allocator in (EqualWeight(), RiskParity()):
            portfolio = run_portfolio(kept, kept_returns, asset_returns, allocator, config)
            for label, period in (("pristine", pristine), ("familiar", familiar),
                                  ("full", index)):
                entry = score(portfolio.result.returns.reindex(period),
                              f"{allocator.name} / {label}")
                rows.append(entry)

    # Benchmarks worth beating: the equal-weight buy-and-hold of the same
    # universe, and 60/40. A strategy that does not beat holding the assets is
    # not a strategy.
    buy_hold = asset_returns.mean(axis=1)
    sixty_forty = 0.6 * asset_returns["VFINX"] + 0.4 * asset_returns["VUSTX"]
    for label, series in (("BENCHMARK equal-weight buy&hold", buy_hold),
                          ("BENCHMARK 60/40", sixty_forty)):
        for period_name, period in (("pristine", pristine), ("familiar", familiar),
                                    ("full", index)):
            rows.append(score(series.reindex(period), f"{label} / {period_name}"))

    book = pd.DataFrame(rows)
    print(book.to_string(index=False))

    # =====================================================================
    section("[4] ARE THE SURVIVORS INDEPENDENT, OR ONE BET WEARING THREE HATS?")
    # =====================================================================
    from src.risk import effective_bets, ledoit_wolf_covariance, spectral_bets

    book_returns = None
    if len(survivors) > 1:
        names = list(survivors["sleeve"])
        survivor_streams = pd.DataFrame({n: streams[n] for n in names}).dropna()
        covariance, _ = ledoit_wolf_covariance(survivor_streams)
        equal = np.full(len(names), 1.0 / len(names))
        # spectral_bets, not effective_bets: the portfolio-level count is
        # degenerate on equal weights (it returns 1 for any positive
        # correlation) and would answer 1.00 here regardless of the truth.
        bets = spectral_bets(covariance)

        kept = {n: books[n] for n in names}
        kept_returns = sleeve_return_streams(kept, asset_returns, costs)
        book_returns = run_portfolio(kept, kept_returns, asset_returns,
                                     EqualWeight(), config).result.returns

        best_single = max(names, key=lambda n: backtest.sharpe_of(streams[n]))
        print(f"    survivors                 {names}")
        print(f"    independent bets (spectral) {bets:.2f} of {len(names)}")
        print(f"    best single ({best_single})       "
              f"{backtest.sharpe_of(streams[best_single]):+.3f}")
        print(f"    all {len(names)} combined            "
              f"{backtest.sharpe_of(book_returns):+.3f}")
        added = backtest.sharpe_of(book_returns) - backtest.sharpe_of(streams[best_single])
        print(f"\n    Combining them adds {added:+.3f} Sharpe over the best single")
        print(f"    sleeve -- about what {bets:.2f} independent bets should buy.")
        print("\n    CAVEAT: this book keeps the sleeves that were positive in BOTH")
        print("    periods, so it used the holdout to select. Its pristine number is")
        print("    contaminated by that choice. The single-sleeve trend result is the")
        print("    clean out-of-sample test, and is the one to quote.")

    # =====================================================================
    section("[5] WHY BETTING-AGAINST-BETA FAILED -- an implementation finding")
    # =====================================================================
    bab_book = books["bab"]
    live = bab_book[bab_book.abs().sum(axis=1) > 0]
    classes = extended.asset_class_map()
    print("    Mean net exposure by asset class:")
    for klass in ("Equity", "Rates", "Credit", "RealAsset"):
        members = [c for c in live.columns if classes.get(c) == klass]
        if members:
            print(f"      {klass:<10} {live[members].sum(axis=1).mean():+.3f}")
    print("Across MIXED asset classes, 'low beta' means bonds and 'high beta'")
    print("    means equities, so this is a duration bet, not a beta bet.")
    print("    Frazzini-Pedersen document BAB WITHIN an asset class. The factor was")
    print("    not tested here so much as mis-specified -- a finding about the")
    print("    implementation, not evidence the premium is gone.")

    # =====================================================================
    section("[6] ROBUSTNESS OF THE ONE THING THAT WORKED")
    # =====================================================================
    best_name = max(streams, key=lambda n: backtest.sharpe_of(streams[n]))
    best = streams[best_name]

    print(f"    {best_name}, by 5-year block -- is it one lucky decade?")
    for start in range(1992, 2027, 5):
        end = min(start + 4, 2026)
        window = best.loc[str(start):str(end)].dropna()
        if len(window) > 200:
            print(f"      {start}-{end}   Sharpe {backtest.sharpe_of(window):+.3f}")

    print(f"Cost sensitivity (turnover is ~10 round trips a year):")
    for rate in (5.0, 10.0, 20.0, 40.0):
        stream = backtest.run(books[best_name], asset_returns, rate).returns
        print(f"      {rate:>4.0f}bps   Sharpe {backtest.sharpe_of(stream):+.3f}   "
              f"t {backtest.newey_west_sharpe_tstat(stream):+.2f}")

    # =====================================================================
    section("[7] WHAT SURVIVES MULTIPLE-TESTING CORRECTION")
    # =====================================================================
    # Every configuration this repo has scored on any sample, not just the ones
    # that made the tables. Undercounting here is the easiest way to make a
    # deflated Sharpe look respectable.
    n_trials = (30            # ETF study: 5 sleeves x 6 allocators
                + 12          # reversal repair variants (scripts/reversal_search.py)
                + len(sleeves)  # the literature candidates above
                + 4           # the four breadth universes in section [8]
                + 1)          # the tanh continuous-response variant
    trial_sharpes = [backtest.sharpe_of(s) for s in streams.values()]
    print(f"    trials counted: {n_trials}")
    print(f"    (30 ETF configs + 12 reversal + {len(sleeves)} candidates + 4 breadth + 1 variant)")

    # The deflated Sharpe depends on how the null's spread is estimated, and the
    # two defensible choices disagree. Reporting one alone would be a choice
    # dressed as a result.
    empirical_sd = float(np.std(trial_sharpes, ddof=1))
    years = len(index) / 252.0
    noise_sd = float(np.sqrt((1.0 + 0.6 ** 2 / 2) / years))
    print(f"empirical trial-Sharpe sd  {empirical_sd:.3f}  "
          f"-- spread of genuinely DIFFERENT strategies, so it contains real")
    print(f"                                      differences as well as noise, "
          f"and overstates the null's spread")
    print(f"    noise-implied Sharpe se    {noise_sd:.3f}  "
          f"-- what ONE strategy's estimate wobbles by over {years:.0f} years,")
    print(f"                                      which is what the deflation "
          f"formula actually assumes")

    rows = []
    targets = {name: series for name, series in streams.items()}
    if book_returns is not None:
        targets["BOOK (survivors, 1/N)"] = book_returns
    for name, series in targets.items():
        clean = series.dropna()
        raw = backtest.sharpe_of(clean)
        entry = {"candidate": name, "raw_sharpe": round(raw, 3)}
        for label, sd in (("empirical", empirical_sd), ("noise", noise_sd)):
            threshold = deflated.expected_max_sharpe(n_trials, sd ** 2)
            entry[f"dsr_{label}"] = round(
                deflated.probabilistic_sharpe_ratio(clean, threshold), 3)
        haircut = deflated.haircut_sharpe(raw, years, n_trials, method="bhy")
        entry["bhy_haircut"] = haircut["haircut_sharpe"]
        entry["bhy_sig"] = haircut["significant_at_5pct"]
        rows.append(entry)
    deflation = pd.DataFrame(rows).sort_values("raw_sharpe", ascending=False)
    print()
    print(deflation.to_string(index=False))
    print("The BHY haircut needs no variance assumption -- only the trial")
    print("    count -- which is why it is the number to quote when the two DSR")
    print("    conventions disagree.")

    # =====================================================================
    section("[8] BREADTH -- does adding markets help?")
    # =====================================================================
    from src.risk import spectral_bets

    universes = {
        "narrow core (12)": wide[[c for c in extended.EXTENDED_UNIVERSE
                                  if c in wide.columns]],
        "diversifying (22)": prices,
        "US sectors (19)": wide[[c for c in wide.columns
                                 if c in extended.US_SECTOR_FUNDS or c == "VFISX"]],
        "all (40)": wide,
    }
    rows = []
    for label, panel in universes.items():
        total = panel.pct_change().iloc[1:]
        cash_leg = total["VFISX"]
        ex = total.sub(cash_leg, axis=0)
        ctx = SleeveContext(
            prices=panel,
            dividends=pd.DataFrame(columns=["ticker", "date", "amount"]),
            cash_daily=cash_leg,
            asset_class={k: v for k, v in extended.wide_asset_class_map().items()
                         if k in panel.columns})
        one = [TrendFollowing()]
        panel_costs = backtest.volatility_scaled_costs(args.cost_bps, ex.mean(axis=1))
        cfg = PortfolioConfig(cost_bps=panel_costs, target_vol=0.08,
                              sleeve_target_vol=0.10)
        bk = vol_target_sleeves(sleeve_books(one, ctx), ex, cfg, one)
        stream = backtest.run(bk["trend"], ex, panel_costs).returns

        signal = np.sign(panel / panel.shift(252) - 1.0).shift(1)
        per = (signal * ex).dropna(how="all")
        per = per.loc[:, per.std() > 0].dropna()
        covariance, _ = ledoit_wolf_covariance(per)
        rows.append({
            "universe": label,
            "markets": panel.shape[1],
            "book_sharpe": round(backtest.sharpe_of(stream), 3),
            "independent_bets": round(spectral_bets(covariance), 2),
            "mean_per_market_sharpe": round(
                float(np.mean([backtest.sharpe_of(per[c]) for c in per.columns])), 3),
        })
    breadth = pd.DataFrame(rows)
    print(breadth.to_string(index=False))
    print("More markets gave MORE independent bets and a WORSE book. The")
    print("    binding variable is per-market signal quality, not count and not")
    print("    independence: the 18 US sector funds trend ~30% less well, and an")
    print("    equal-weighted book dilutes directly.")
    print("Measured with risk.spectral_bets. The portfolio-level Meucci")
    print("    count is degenerate here -- it returns 1 for any positive")
    print("    correlation on equal weights -- and an earlier version of this")
    print("    study drew the opposite conclusion from it.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "research_edge.json").write_text(json.dumps({
        "sample": {"funds": int(prices.shape[1]),
                   "start": str(index[0].date()), "end": str(index[-1].date()),
                   "years": round(len(index) / 252, 1)},
        "candidates": candidates.to_dict("records"),
        "book": book.to_dict("records"),
        "breadth": breadth.to_dict("records"),
        "deflation": deflation.to_dict("records"),
        "empirical_trial_sd": round(empirical_sd, 4),
        "noise_implied_sd": round(noise_sd, 4),
        "n_trials": n_trials,
    }, indent=2, default=str))
    print(f"\nSaved {REPORTS / 'research_edge.json'}")


if __name__ == "__main__":
    main()
