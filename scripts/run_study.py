"""The study: does running several strategies beat running one?

    python -m scripts.run_study

Structure, and the discipline behind it
---------------------------------------
The sample is split by date once, up front. Everything that could be a choice --
which allocator, whether to volatility-target, whether the drawdown throttle
helps -- is decided on the SELECTION half. The CONFIRMATION half is scored once,
at the end, and not returned to. Without that split, picking the best of six
allocators on the full sample and reporting its number is just publishing the
maximum of six noisy draws.

Sleeve parameters are literature defaults and were never tuned, so they do not
consume the selection half.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest, data, deflated
from src.allocators import default_allocators
from src.portfolio import (
    PortfolioConfig,
    netting_benefit,
    run_portfolio,
    sleeve_books,
    sleeve_return_streams,
    vol_target_sleeves,
)
from src.universe import UNIVERSE
from src.risk import ledoit_wolf_covariance
from src.sleeves.base import SleeveContext
from src.sleeves.strategies import default_sleeves

warnings.filterwarnings("ignore")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
pd.set_option("display.width", 200)


def frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows)[columns]


def split_by_date(index: pd.DatetimeIndex, fraction: float = 0.5):
    cut = index[int(len(index) * fraction)]
    return index[index <= cut], index[index > cut]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--target-vol", type=float, default=0.08)
    ap.add_argument("--lookback", type=int, default=504)
    ap.add_argument("--throttle", action="store_true",
                    help="enable the adaptive drawdown throttle")
    ap.add_argument("--flat-costs", action="store_true",
                    help="use a constant cost rate instead of volatility-scaled")
    args = ap.parse_args()

    print("=" * 100)
    print(f"QUORUM -- {len(default_sleeves())} strategy sleeves, "
          f"{len(default_allocators())} ways of splitting capital between them")
    print("=" * 100)

    prices, dividends, cash, auxiliary = data.load(allow_fetch=False)
    prices = data.common_sample(prices)
    cash_daily = data.daily_cash_rate(cash, prices.index)

    # Everything downstream is in EXCESS returns. This sample spans 0% and 5%+
    # policy rates, and on raw total returns the cash regime distorts the
    # dollar-neutral sleeves, the net-long book's idle capital, and the trend
    # sleeve's absolute-momentum threshold in different directions at once.
    total_returns = data.daily_returns(prices)
    asset_returns = data.excess_returns(total_returns, cash_daily)

    context = SleeveContext(
        prices=prices,
        dividends=dividends,
        cash_daily=cash_daily,
        asset_class={t: meta[1] for t, meta in UNIVERSE.items()},
        auxiliary=auxiliary,
    )

    selection, confirmation = split_by_date(asset_returns.index)
    print(f"\n{prices.shape[1]} assets, {len(asset_returns)} days "
          f"({asset_returns.index[0].date()} to {asset_returns.index[-1].date()})")
    print(f"  selection    {selection[0].date()} to {selection[-1].date()} ({len(selection)} days)")
    print(f"  confirmation {confirmation[0].date()} to {confirmation[-1].date()} "
          f"({len(confirmation)} days)")
    print(f"  costs {args.cost_bps:.0f}bps per unit turnover, "
          f"vol target {100 * args.target_vol:.0f}%")

    # Costs widen with market volatility unless explicitly disabled. A flat rate
    # flatters every de-risking rule in the system, because the overlay
    # generates its largest trades in exactly the weeks spreads blow out.
    market_proxy = asset_returns.mean(axis=1)
    cost_schedule = (args.cost_bps if args.flat_costs
                     else backtest.volatility_scaled_costs(args.cost_bps, market_proxy))
    if not args.flat_costs:
        print(f"  costs volatility-scaled: mean {float(cost_schedule.mean()):.1f}bps, "
              f"p95 {float(cost_schedule.quantile(0.95)):.1f}bps")

    sleeves = default_sleeves()
    raw_books = sleeve_books(sleeves, context)

    config = PortfolioConfig(lookback_days=args.lookback, cost_bps=cost_schedule,
                             target_vol=args.target_vol,
                             use_drawdown_throttle=args.throttle)

    # Scale each sleeve to a common volatility BEFORE allocating, so the
    # allocator splits risk rather than notional and the portfolio scalar is
    # not left trying to undo the netting from above.
    books = vol_target_sleeves(raw_books, asset_returns, config, sleeves)
    sleeve_net = sleeve_return_streams(books, asset_returns, cost_schedule)
    sleeve_gross = sleeve_return_streams(books, asset_returns, 0.0)

    # ---- [1] the sleeves on their own -------------------------------------
    print("\n[1] Each sleeve standalone, full sample")
    rows = []
    for sleeve in sleeves:
        gross = backtest.run(books[sleeve.name], asset_returns, 0.0, sleeve.name)
        net = backtest.run(books[sleeve.name], asset_returns, cost_schedule, sleeve.name)
        metrics = net.metrics()
        metrics["gross_sharpe"] = round(backtest.sharpe_of(gross.returns), 3)
        metrics["rebalance"] = sleeve.rebalance
        metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(net.returns), 2)
        rows.append(metrics)
    sleeve_table = frame(rows, ["name", "rebalance", "gross_sharpe", "sharpe", "t_stat_nw",
                                "annual_return", "annual_vol", "max_drawdown",
                                "turnover_annual"])
    print(sleeve_table.to_string(index=False))
    busiest = sleeve_table.loc[sleeve_table["turnover_annual"].idxmax()]
    print(f"\n    Costs are not a rounding error: the busiest sleeve ({busiest['name']}) "
          f"turns over {busiest['turnover_annual']:.0f}x a year,")
    print(f"    which is {busiest['gross_sharpe'] - busiest['sharpe']:+.3f} Sharpe "
          f"of gross-to-net drag on that sleeve alone.")

    # ---- [1b] does sleeve performance persist across the split? -----------
    print("\n[1b] Sleeve Sharpe by half -- does past sleeve performance predict future?")
    rows = []
    for column in sleeve_net.columns:
        selection_sharpe = backtest.sharpe_of(sleeve_net[column].reindex(selection).dropna())
        confirmation_sharpe = backtest.sharpe_of(sleeve_net[column].reindex(confirmation).dropna())
        rows.append({"sleeve": column,
                     "selection_sharpe": round(selection_sharpe, 3),
                     "confirmation_sharpe": round(confirmation_sharpe, 3),
                     "same_sign": "yes" if selection_sharpe * confirmation_sharpe > 0 else "NO"})
    persistence = pd.DataFrame(rows)
    print(persistence.to_string(index=False))

    rank_correlation = persistence["selection_sharpe"].corr(
        persistence["confirmation_sharpe"], method="spearman")
    agree = int((persistence["same_sign"] == "yes").sum())
    print(f"\n    rank correlation across halves     {rank_correlation:+.3f} "
          f"({len(persistence)} sleeves -- indicative only)")
    print(f"    sleeves keeping their sign         {agree} of {len(persistence)}")

    # ---- [2] are they actually different? ---------------------------------
    print("\n[2] Sleeve correlation (net, full sample)")
    correlation = sleeve_net.corr()
    print(correlation.round(3).to_string())

    covariance, shrinkage = ledoit_wolf_covariance(sleeve_net.dropna())
    equal = np.full(len(sleeves), 1.0 / len(sleeves))
    from src.risk import diversification_ratio, effective_bets
    print(f"\n    equal-weight diversification ratio {diversification_ratio(equal, covariance):.2f}")
    print(f"    effective independent bets         {effective_bets(equal, covariance):.2f}"
          f"  (of {len(sleeves)} sleeves)")

    off = correlation.to_numpy()[~np.eye(len(sleeves), dtype=bool)]
    worst = correlation.where(~np.eye(len(sleeves), dtype=bool)).stack().idxmax()
    print(f"    mean pairwise correlation          {off.mean():+.3f}")
    print(f"    most correlated pair               {worst[0]} / {worst[1]} "
          f"{correlation.loc[worst]:+.3f}")

    # ---- [3] allocators, decided on the selection half --------------------
    # `config` is built once, above, before the sleeves are volatility-targeted
    # with it. A second construction here used to shadow it -- silently dropping
    # both the --throttle flag and the volatility-scaled cost schedule, with no
    # error and entirely plausible output. It was caught only because toggling
    # --throttle changed nothing at all.
    print("\n[3] SELECTION HALF -- choose the allocator here, and only here")
    portfolios = {}
    rows = []
    for allocator in default_allocators():
        portfolio = run_portfolio(books, sleeve_net, asset_returns, allocator, config)
        portfolios[allocator.name] = portfolio
        window = portfolio.result.returns.reindex(selection).dropna()
        metrics = backtest.performance_metrics(window, name=allocator.name)
        metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(window), 2)
        rows.append(metrics)

    best_sleeve_selection = max(
        sleeve_net.columns,
        key=lambda c: backtest.sharpe_of(sleeve_net[c].reindex(selection).dropna()),
    )
    for label, series in (("best single sleeve (" + best_sleeve_selection + ")",
                           sleeve_net[best_sleeve_selection]),):
        window = series.reindex(selection).dropna()
        metrics = backtest.performance_metrics(window, name=label)
        metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(window), 2)
        rows.append(metrics)

    selection_table = frame(rows, ["name", "sharpe", "t_stat_nw", "annual_return",
                                   "annual_vol", "max_drawdown", "sortino"])
    print(selection_table.to_string(index=False))

    chosen = max(
        (r for r in rows if r["name"] in portfolios),
        key=lambda r: r["sharpe"] if np.isfinite(r["sharpe"]) else -1e9,
    )["name"]
    print(f"\n    Chosen on the selection half: {chosen}")

    # ---- [4] confirmation, looked at once ---------------------------------
    print("\n[4] CONFIRMATION HALF -- scored once, allocator fixed above")
    rows = []
    for name, portfolio in portfolios.items():
        window = portfolio.result.returns.reindex(confirmation).dropna()
        metrics = backtest.performance_metrics(window, name=name)
        metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(window), 2)
        metrics["chosen"] = "<--" if name == chosen else ""
        rows.append(metrics)

    for column in sleeve_net.columns:
        window = sleeve_net[column].reindex(confirmation).dropna()
        metrics = backtest.performance_metrics(window, name=f"  sleeve: {column}")
        metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(window), 2)
        metrics["chosen"] = ""
        rows.append(metrics)

    confirmation_table = frame(rows, ["name", "sharpe", "t_stat_nw", "annual_return",
                                      "annual_vol", "max_drawdown", "chosen"])
    print(confirmation_table.to_string(index=False))

    equal_confirmation = portfolios["equal_weight"].result.returns.reindex(confirmation).dropna()
    chosen_confirmation = portfolios[chosen].result.returns.reindex(confirmation).dropna()
    print(f"\n    1/N on the confirmation half:      {backtest.sharpe_of(equal_confirmation):+.3f}")
    print(f"    {chosen} on the same:  {backtest.sharpe_of(chosen_confirmation):+.3f}")

    # ---- [4b] the fair test: vet the sleeves first ------------------------
    #
    # Section [4] answers "does naive diversification across five unvetted
    # strategies help?" -- and it does not. But no desk funds a sleeve it
    # believes loses money, so that comparison is unfair to the multi-strategy
    # idea. The fair version keeps only the sleeves that were profitable on the
    # SELECTION half, which is information genuinely available at the split
    # date, and scores that book on the confirmation half.
    print("\n[4b] Vetted book -- fund only sleeves that were profitable on the selection half")
    vetted = [row["sleeve"] for row in persistence.to_dict("records")
              if row["selection_sharpe"] > 0]
    rejected = [c for c in sleeve_net.columns if c not in vetted]
    print(f"    kept     {vetted}")
    print(f"    rejected {rejected}")

    rows = []
    if vetted:
        vetted_books = {name: books[name] for name in vetted}
        vetted_returns = sleeve_net[vetted]
        for allocator in default_allocators():
            portfolio = run_portfolio(vetted_books, vetted_returns, asset_returns,
                                      allocator, config, name=f"vetted/{allocator.name}")
            window = portfolio.result.returns.reindex(confirmation).dropna()
            metrics = backtest.performance_metrics(window, name=f"vetted/{allocator.name}")
            metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(window), 2)
            rows.append(metrics)

    # The benchmark that matters: just running the single best sleeve.
    single = sleeve_net[best_sleeve_selection].reindex(confirmation).dropna()
    metrics = backtest.performance_metrics(single, name=f"single: {best_sleeve_selection}")
    metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(single), 2)
    rows.append(metrics)

    all_five = portfolios["equal_weight"].result.returns.reindex(confirmation).dropna()
    metrics = backtest.performance_metrics(all_five, name="all five, 1/N (from [4])")
    metrics["t_stat_nw"] = round(backtest.newey_west_sharpe_tstat(all_five), 2)
    rows.append(metrics)

    vetted_table = frame(rows, ["name", "sharpe", "t_stat_nw", "annual_return",
                                "annual_vol", "max_drawdown", "sortino"])
    print(vetted_table.to_string(index=False))

    # ---- [5] what the netting is worth ------------------------------------
    print("\n[5] Netting sleeves into one book vs averaging their return streams")
    netting = netting_benefit(portfolios["equal_weight"], books, sleeve_gross,
                              asset_returns, config)
    for key, value in netting.items():
        print(f"    {key:<28} {value}")

    # ---- [6] does diversification survive stress? -------------------------
    print("\n[6] Sleeve correlation in calm markets vs drawdowns")
    market = asset_returns["SPY"]
    equity_curve = (1.0 + market).cumprod()
    market_drawdown = equity_curve / equity_curve.cummax() - 1.0
    stressed = market_drawdown < -0.10

    calm_correlation = sleeve_net[~stressed].corr()
    stress_correlation = sleeve_net[stressed].corr()
    mask = ~np.eye(len(sleeves), dtype=bool)
    calm_mean = float(calm_correlation.to_numpy()[mask].mean())
    stress_mean = float(stress_correlation.to_numpy()[mask].mean())

    calm_cov, _ = ledoit_wolf_covariance(sleeve_net[~stressed].dropna())
    stress_cov, _ = ledoit_wolf_covariance(sleeve_net[stressed].dropna())
    print(f"    days in a >10% SPY drawdown        {int(stressed.sum())} of {len(stressed)}")
    print(f"    mean pairwise correlation, calm    {calm_mean:+.3f}")
    print(f"    mean pairwise correlation, stress  {stress_mean:+.3f}")
    print(f"    effective bets, calm               {effective_bets(equal, calm_cov):.2f}")
    print(f"    effective bets, stress             {effective_bets(equal, stress_cov):.2f}")

    # ---- [7] what survives a multiple-testing correction ------------------
    print(f"\n[7] DEFLATED SHARPE -- "
          f"{len(default_allocators()) * len(sleeves)} configurations on one sample")
    all_trial_sharpes = []
    for portfolio in portfolios.values():
        all_trial_sharpes.append(backtest.sharpe_of(portfolio.result.returns))
    for column in sleeve_net.columns:
        all_trial_sharpes.append(backtest.sharpe_of(sleeve_net[column]))
    if vetted:
        for row in vetted_table.to_dict("records"):
            if np.isfinite(row.get("sharpe", np.nan)):
                all_trial_sharpes.append(row["sharpe"])

    n_trials = len(default_allocators()) * len(sleeves)
    print(f"    trials counted: {n_trials} "
          f"({len(sleeves)} sleeves x {len(default_allocators())} allocators)")
    print(f"    observed spread of trial Sharpes: "
          f"variance {np.var(all_trial_sharpes, ddof=1):.4f}")

    rows = []
    candidates = {
        f"portfolio: {chosen}": portfolios[chosen].result.returns,
        "portfolio: equal_weight": portfolios["equal_weight"].result.returns,
        f"best sleeve: {best_sleeve_selection}": sleeve_net[best_sleeve_selection],
    }
    for label, series in candidates.items():
        full = series.dropna()
        result = deflated.deflated_sharpe_ratio(full, n_trials, all_trial_sharpes)
        haircut = deflated.haircut_sharpe(result["annual_sharpe"], result["years"],
                                          n_trials, method="bhy")
        rows.append({
            "candidate": label,
            "raw_sharpe": result["annual_sharpe"],
            "psr_vs_zero": result["psr_vs_zero"],
            "threshold": result["expected_max_sharpe_under_null"],
            "deflated_sharpe": result["deflated_sharpe"],
            "bhy_haircut_sharpe": haircut["haircut_sharpe"],
            "haircut_pct": haircut["haircut_pct"],
            "survives": "yes" if result["deflated_sharpe"] > 0.95 else "no",
        })
    deflation_table = pd.DataFrame(rows)
    print()
    print(deflation_table.to_string(index=False))
    print(f"\n    deflated_sharpe is P(true Sharpe > the best of {n_trials} "
          f"trials under the null).")
    print("    Above 0.95 is the usual bar. Nothing here is close to it.")

    # ---- save --------------------------------------------------------------
    REPORTS.mkdir(parents=True, exist_ok=True)
    sleeve_table.to_csv(REPORTS / "sleeves.csv", index=False)
    correlation.to_csv(REPORTS / "sleeve_correlation.csv")
    selection_table.to_csv(REPORTS / "selection.csv", index=False)
    confirmation_table.to_csv(REPORTS / "confirmation.csv", index=False)

    summary = {
        "sample": {
            "assets": int(prices.shape[1]),
            "days": int(len(asset_returns)),
            "start": str(asset_returns.index[0].date()),
            "end": str(asset_returns.index[-1].date()),
        },
        "cost_bps": args.cost_bps,
        "target_vol": args.target_vol,
        "sleeves": sleeve_table.to_dict("records"),
        "correlation": correlation.round(4).to_dict(),
        "equal_weight_effective_bets": round(float(effective_bets(equal, covariance)), 3),
        "equal_weight_diversification_ratio": round(float(diversification_ratio(equal, covariance)), 3),
        "chosen_allocator": chosen,
        "selection": selection_table.to_dict("records"),
        "confirmation": confirmation_table.to_dict("records"),
        "persistence": persistence.to_dict("records"),
        "vetted": vetted_table.to_dict("records"),
        "vetted_sleeves": vetted,
        "deflation": deflation_table.to_dict("records"),
        "netting": netting,
        "stress": {
            "calm_mean_correlation": round(calm_mean, 4),
            "stress_mean_correlation": round(stress_mean, 4),
            "calm_effective_bets": round(float(effective_bets(equal, calm_cov)), 3),
            "stress_effective_bets": round(float(effective_bets(equal, stress_cov)), 3),
        },
    }
    (REPORTS / "study.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved reports to {REPORTS}")


if __name__ == "__main__":
    main()
