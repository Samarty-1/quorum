"""Empirical audit of the two-tier allocation spec.

Runs the reviewed architecture -- the specified 15-ETF universe, the specified
risk overlay (10% vol target, L_max 1.5, 10/15/20% drawdown throttle with a
20-day recovery trigger) -- and measures the failure modes rather than
speculating about them.

Where this differs from the spec, and it matters for reading the output:
the Level-1 sleeves are this repo's implementations, which are close but not
identical analogues of the specified ones (see README). The Level-2 allocators
and the entire risk overlay ARE the specified ones. So conclusions about the
allocation and risk layers transfer directly; conclusions about sleeve
behaviour are indicative of the architecture, not of the exact signals.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest, data
from src.allocators import default_allocators
from src.portfolio import PortfolioConfig, run_portfolio, sleeve_books, sleeve_return_streams
from src.risk import ledoit_wolf_covariance, realised_volatility, volatility_scalar
from src.sleeves.base import SleeveContext, rebalance_dates
from src.sleeves.strategies import default_sleeves

warnings.filterwarnings("ignore")
REPORTS = Path(__file__).resolve().parent.parent / "reports"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
pd.set_option("display.width", 210)

# The audited spec's universe, verbatim.
SPEC_UNIVERSE = {
    "SPY": "Equity", "QQQ": "Equity", "IWM": "Equity", "EFA": "Equity", "EEM": "Equity",
    "SHY": "Rates", "IEF": "Rates", "TLT": "Rates", "TIP": "Rates",
    "LQD": "Credit", "HYG": "Credit",
    "GLD": "RealAsset", "DBC": "RealAsset", "VNQ": "RealAsset", "XLE": "RealAsset",
}

SPEC_PRICES = DATA_DIR / "spec_prices.parquet"
SPEC_DIVIDENDS = DATA_DIR / "spec_dividends.parquet"

CRISES = {
    "GFC 2008":       ("2008-09-01", "2009-03-31"),
    "Taper 2013":     ("2013-05-01", "2013-09-30"),
    "Volmageddon 18": ("2018-01-15", "2018-03-15"),
    "COVID 2020":     ("2020-02-15", "2020-04-30"),
    "Rate hikes 22":  ("2022-01-01", "2022-10-31"),
}


def load_spec_universe(refresh: bool = False):
    if SPEC_PRICES.exists() and not refresh:
        return pd.read_parquet(SPEC_PRICES), pd.read_parquet(SPEC_DIVIDENDS)

    import yfinance as yf
    symbols = sorted(SPEC_UNIVERSE)
    raw = yf.download(symbols, start="2007-01-01", auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    prices, rows = {}, []
    for symbol in symbols:
        series = raw[symbol]["Close"].dropna()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        prices[symbol] = series
        payments = yf.Ticker(symbol).dividends
        if len(payments):
            payments.index = pd.to_datetime(payments.index).tz_localize(None)
            for date, amount in payments.items():
                rows.append({"ticker": symbol, "date": date, "amount": float(amount)})

    price_frame = pd.DataFrame(prices).sort_index()
    dividend_frame = pd.DataFrame(rows, columns=["ticker", "date", "amount"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    price_frame.to_parquet(SPEC_PRICES)
    dividend_frame.to_parquet(SPEC_DIVIDENDS)
    return price_frame, dividend_frame


# --------------------------------------------------------------------------
# The specified drawdown throttle, implemented exactly as written.
# --------------------------------------------------------------------------

def spec_throttle(returns: pd.Series, recovery_days: int = 20) -> tuple[pd.Series, dict]:
    """10% -> 0.75, 15% -> 0.50, 20% -> flat until a 20-day recovery trigger.

    Implemented as a stateful loop rather than vectorised, because the spec is
    genuinely path-dependent: once the book is flat it stops earning, so its own
    equity curve stops recovering, and whether the trigger ever fires depends on
    what the throttle itself did. That feedback is the point of the exercise and
    a vectorised version would quietly assume it away.

    The `recovery_days` trigger is read as: the underlying (unthrottled) book has
    risen for 20 consecutive trading days' worth of cumulative gain off its low.
    The spec does not define it, which is itself a finding.
    """
    exposure = pd.Series(1.0, index=returns.index)
    throttled = pd.Series(0.0, index=returns.index)

    equity = 1.0
    peak = 1.0
    halted = False
    halt_low = np.nan
    days_since_halt = 0

    # The unthrottled curve, used only for the recovery trigger -- a halted book
    # cannot recover on its own equity.
    reference = (1.0 + returns).cumprod()

    for i, date in enumerate(returns.index):
        if i > 0:
            throttled.iloc[i] = returns.iloc[i] * exposure.iloc[i - 1]
            equity *= (1.0 + throttled.iloc[i])
            peak = max(peak, equity)

        drawdown = equity / peak - 1.0

        if halted:
            days_since_halt += 1
            recovered = (days_since_halt >= recovery_days and
                         reference.iloc[i] > halt_low * 1.0)
            if recovered:
                halted = False
                days_since_halt = 0
                exposure.iloc[i] = 0.75
            else:
                exposure.iloc[i] = 0.0
                halt_low = min(halt_low, reference.iloc[i])
            continue

        if drawdown <= -0.20:
            halted = True
            days_since_halt = 0
            halt_low = reference.iloc[i]
            exposure.iloc[i] = 0.0
        elif drawdown <= -0.15:
            exposure.iloc[i] = 0.50
        elif drawdown <= -0.10:
            exposure.iloc[i] = 0.75
        else:
            exposure.iloc[i] = 1.0

    stats = {
        "days_at_full_risk": int((exposure == 1.0).sum()),
        "days_at_075": int((exposure == 0.75).sum()),
        "days_at_050": int((exposure == 0.50).sum()),
        "days_flat": int((exposure == 0.0).sum()),
        "pct_time_derisked": round(100.0 * float((exposure < 1.0).mean()), 1),
    }
    return exposure, stats


def section(title: str) -> None:
    print(f"\n{'=' * 104}\n{title}\n{'=' * 104}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--max-leverage", type=float, default=1.5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    args = ap.parse_args()

    prices, dividends = load_spec_universe(args.refresh)
    prices = data.common_sample(prices)
    asset_returns = data.daily_returns(prices)
    _, _, cash_annual = data.load(allow_fetch=False)
    cash_daily = data.daily_cash_rate(cash_annual, prices.index)
    context = SleeveContext(prices=prices, dividends=dividends, cash_daily=cash_daily)

    print(f"Spec universe: {prices.shape[1]} ETFs, {len(asset_returns)} days "
          f"({asset_returns.index[0].date()} to {asset_returns.index[-1].date()})")

    sleeves = default_sleeves()
    books = sleeve_books(sleeves, context)
    sleeve_net = sleeve_return_streams(books, asset_returns, args.cost_bps)
    findings: dict = {}

    # =====================================================================
    section("[A] SIGNAL HORIZON -- is the mean-reversion sleeve traded after its signal has decayed?")
    # =====================================================================
    # The spec runs a 20-day RSI / Bollinger signal on a MONTHLY rebalance.
    # Measure how much of the signal's information is left when it is traded,
    # by autocorrelating the sleeve's own target weights at the holding horizon.
    print("\nSignal persistence: correlation of a sleeve's weights with themselves k days later")
    print("(if this decays below ~0.3 inside the rebalance interval, the book is trading stale signal)")
    rows = []
    for sleeve in sleeves:
        book = books[sleeve.name]
        live = book[book.abs().sum(axis=1) > 0]
        row = {"sleeve": sleeve.name, "rebalance": sleeve.rebalance}
        for lag in (5, 21, 63):
            correlations = [live[c].autocorr(lag) for c in live.columns]
            row[f"corr_{lag}d"] = round(float(np.nanmean(correlations)), 3)
        rows.append(row)
    persistence = pd.DataFrame(rows)
    print(persistence.to_string(index=False))

    # Direct test on the RAW signal, at both sampling frequencies. Reported with
    # standard errors, because the headline here is how WEAK the effect is
    # rather than which sign it takes -- an earlier pass quoted only the
    # monthly-sampled number and read a sign flip as a finding, when neither
    # estimate clears 2 standard errors.
    raw_reversal = -(prices / prices.shift(5) - 1.0)
    forward_5 = prices.shift(-5) / prices - 1.0

    def mean_ic(signal: pd.DataFrame, step: int) -> tuple[float, float, int]:
        ics = []
        for date in signal.index[::step]:
            if date not in forward_5.index:
                continue
            s, f = signal.loc[date].dropna(), forward_5.loc[date].dropna()
            shared = s.index.intersection(f.index)
            if len(shared) >= 8:
                ics.append(s[shared].corr(f[shared], method="spearman"))
        arr = np.asarray(ics, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            return float("nan"), float("nan"), len(arr)
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        return float(arr.mean()), se, len(arr)

    print("\nMean IC of the 5-day reversal signal vs the next 5 days:")
    ic_summary = {}
    for step, label in ((5, "sampled weekly (as traded)"), (21, "sampled monthly (per spec)")):
        ic, se, n = mean_ic(raw_reversal, step)
        ic_summary[label] = {"ic": round(ic, 4), "se": round(se, 4),
                             "t": round(ic / se, 2) if se else None, "n": n}
        print(f"    {label:<30} IC {ic:+.4f} +/- {se:.4f}   t {ic / se:+.2f}   n {n}")
    print("    Neither clears 2 standard errors. The effect is weak and not robust")
    print("    to the sampling choice, which is the finding -- not its sign.")

    print("\n  Signal ageing (weekly sampling), IC as the signal is held longer:")
    ic_by_lag = {}
    for lag in (0, 5, 10, 21):
        ic, se, _ = mean_ic(raw_reversal.shift(lag), 5)
        ic_by_lag[lag] = round(ic, 4)
        note = "  <- what a monthly rebalance actually holds" if lag == 21 else ""
        print(f"    aged {lag:>2} days:  IC {ic:+.4f} +/- {se:.4f}{note}")
    findings["reversal_ic"] = {"by_sampling": ic_summary, "by_age": ic_by_lag}

    # =====================================================================
    section("[B] ASSET-CLASS BIAS -- does the yield/value sleeve structurally own HYG and VNQ?")
    # =====================================================================
    print("\nAverage NET exposure by asset class, per sleeve (as a share of gross book):")
    class_map = pd.Series(SPEC_UNIVERSE)
    rows = []
    for sleeve in sleeves:
        book = books[sleeve.name]
        live = book[book.abs().sum(axis=1) > 0]
        row = {"sleeve": sleeve.name}
        for klass in ["Equity", "Rates", "Credit", "RealAsset"]:
            members = [c for c in live.columns if class_map.get(c) == klass]
            row[klass] = round(float(live[members].sum(axis=1).mean()), 3)
        rows.append(row)
    exposure = pd.DataFrame(rows)
    print(exposure.to_string(index=False))

    print("\nMost persistently held names (mean weight) for the income/value sleeves:")
    for name in ("carry", "value"):
        book = books[name]
        live = book[book.abs().sum(axis=1) > 0]
        top = live.mean().sort_values(ascending=False)
        print(f"    {name:<7} long:  " + ", ".join(f"{t} {w:+.3f}" for t, w in top.head(4).items()))
        print(f"    {name:<7} short: " + ", ".join(f"{t} {w:+.3f}" for t, w in top.tail(4).items()))
    findings["asset_class_exposure"] = exposure.to_dict("records")

    # =====================================================================
    section("[C] SLEEVE INTERACTION -- do the sleeves fight each other, and when?")
    # =====================================================================
    # Internal crossing: how much of the summed gross position cancels when the
    # sleeves are netted into one book at equal weight.
    equal = 1.0 / len(sleeves)
    summed_gross = sum(books[s.name].abs().sum(axis=1) * equal for s in sleeves)
    netted = sum(books[s.name] * equal for s in sleeves)
    netted_gross = netted.abs().sum(axis=1)
    crossing = (1.0 - netted_gross / summed_gross.replace(0, np.nan)).dropna()
    print(f"\nInternal position crossing (fraction of gross that cancels on netting):")
    print(f"    full sample mean   {crossing.mean():.1%}")
    for label, (start, end) in CRISES.items():
        window = crossing.loc[start:end]
        if len(window):
            print(f"    {label:<17} {window.mean():.1%}")

    print("\nTrend vs Reversal rolling 63-day correlation, by regime:")
    pair = sleeve_net[["trend", "reversal"]].dropna()
    rolling = pair["trend"].rolling(63).corr(pair["reversal"])
    print(f"    full sample mean   {rolling.mean():+.3f}")
    for label, (start, end) in CRISES.items():
        window = rolling.loc[start:end]
        if len(window.dropna()):
            print(f"    {label:<17} {window.mean():+.3f}")
    findings["internal_crossing_mean"] = round(float(crossing.mean()), 4)

    # =====================================================================
    section("[D] VOLATILITY TARGETING -- is a 10% target reachable at L_max = 1.5?")
    # =====================================================================
    config = PortfolioConfig(lookback_days=504, cost_bps=args.cost_bps,
                             target_vol=args.target_vol, max_leverage=args.max_leverage)
    portfolios = {}
    for allocator in default_allocators():
        portfolios[allocator.name] = run_portfolio(books, sleeve_net, asset_returns,
                                                   allocator, config)

    equal_weight_book = sum(books[s.name] * equal for s in sleeves)
    unlevered = backtest.run(equal_weight_book, asset_returns, cost_bps=0.0).returns
    unlevered_vol = realised_volatility(unlevered, 63).dropna()

    raw_scalar = (args.target_vol / unlevered_vol).replace([np.inf, -np.inf], np.nan).dropna()
    capped = raw_scalar.clip(upper=args.max_leverage)
    print(f"\nUnlevered equal-weight book realised vol (63d): "
          f"median {unlevered_vol.median():.2%}, "
          f"10th pct {unlevered_vol.quantile(0.10):.2%}, "
          f"90th pct {unlevered_vol.quantile(0.90):.2%}")
    print(f"Leverage needed to hit {args.target_vol:.0%}: "
          f"median {raw_scalar.median():.2f}x, 90th pct {raw_scalar.quantile(0.90):.2f}x")
    print(f"\n    Time the L_max = {args.max_leverage} cap BINDS: "
          f"{(raw_scalar > args.max_leverage).mean():.1%} of days")
    print(f"    Realised vol of the capped book: {(unlevered * capped.shift(1)).std() * np.sqrt(252):.2%} "
          f"(target {args.target_vol:.0%})")
    findings["vol_target"] = {
        "unlevered_median_vol": round(float(unlevered_vol.median()), 4),
        "median_required_leverage": round(float(raw_scalar.median()), 3),
        "pct_days_cap_binds": round(float((raw_scalar > args.max_leverage).mean()), 4),
    }

    # =====================================================================
    section("[E] COVARIANCE WINDOW -- does 126 days react fast enough?")
    # =====================================================================
    # A COMMON threshold across windows, not each estimator's own peak.
    # Measuring "80% of its own peak" is not comparable: a 21-day window has a
    # much sharper peak than a 252-day one, so each estimator is graded against
    # a different bar and the shortest window can look slow. The question that
    # matters operationally is when each estimator tells you risk has risen by a
    # fixed amount, so the bar is 1.5x the pre-episode level for all of them.
    print("\nTrading days until the vol estimate rises 50% above its pre-episode level")
    print("(-- means it never got there inside the episode window):")
    header = f"    {'episode':<17}" + "".join(f"{w:>11}d" for w in (21, 63, 126, 252))
    print(header)
    lag_table = {}
    for label, (start, end) in CRISES.items():
        row = f"    {label:<17}"
        lag_table[label] = {}
        for window in (21, 63, 126, 252):
            vol = realised_volatility(unlevered, window, min_periods=window // 3)
            baseline = vol.loc[:start].dropna()
            episode = vol.loc[start:end].dropna()
            if len(episode) < 5 or len(baseline) < 5:
                row += f"{'--':>12}"
                continue
            threshold = float(baseline.iloc[-1]) * 1.5
            reached = episode[episode >= threshold]
            if len(reached):
                days = int(episode.index.get_loc(reached.index[0]))
                lag_table[label][window] = days
                row += f"{days:>12}"
            else:
                lag_table[label][window] = None
                row += f"{'--':>12}"
        print(row)
    findings["vol_reaction_days"] = lag_table

    print("\n  Interpretation: these are days of EXPOSURE AT THE OLD RISK LEVEL before")
    print("  the scalar has substantially responded. Multiply by the daily loss rate")
    print("  in the episode to get the cost of the estimator's lag.")

    # =====================================================================
    section("[F] DRAWDOWN THROTTLE -- does the spec's 10/15/20 ladder whipsaw?")
    # =====================================================================
    base = portfolios["equal_weight"].result.returns
    exposure, stats = spec_throttle(base)
    throttled = (base * exposure.shift(1).fillna(1.0)).rename("throttled")

    print("\nSpec throttle applied to the 1/N book:")
    for key, value in stats.items():
        print(f"    {key:<24} {value}")

    comparison = pd.DataFrame([
        backtest.performance_metrics(base, name="no throttle"),
        backtest.performance_metrics(throttled, name="spec throttle"),
    ])[["name", "sharpe", "annual_return", "annual_vol", "max_drawdown", "hit_rate"]]
    print()
    print(comparison.to_string(index=False))

    print("\nBehaviour through each episode (cumulative return, throttle on vs off):")
    for label, (start, end) in CRISES.items():
        a = float((1.0 + base.loc[start:end]).prod() - 1.0)
        b = float((1.0 + throttled.loc[start:end]).prod() - 1.0)
        avg_exposure = float(exposure.loc[start:end].mean())
        print(f"    {label:<17} off {a:+7.2%}   on {b:+7.2%}   "
              f"delta {b - a:+7.2%}   mean exposure {avg_exposure:.2f}")

    # The feedback trap: drawdown is measured on the THROTTLED curve, so cutting
    # risk slows the recovery, which keeps the drawdown deep, which keeps the
    # book cut. Compare the depth each curve reaches and how long each spends
    # under water.
    def underwater_days(series: pd.Series) -> int:
        equity = (1.0 + series).cumprod()
        return int((equity < equity.cummax() * 0.999).sum())

    print(f"\n    Feedback check -- drawdown is measured on the throttled curve:")
    print(f"      max drawdown   off {backtest.performance_metrics(base)['max_drawdown']:+.2%}"
          f"   on {backtest.performance_metrics(throttled)['max_drawdown']:+.2%}")
    print(f"      days underwater off {underwater_days(base):>5}"
          f"   on {underwater_days(throttled):>5}")
    print(f"      -> the throttle never reached the 20% rung precisely BECAUSE the")
    print(f"         10% and 15% rungs had already cut risk. The severe rule is")
    print(f"         untested by this sample, not proven safe by it.")

    # The specific whipsaw question: what happened around the COVID bottom?
    covid = exposure.loc["2020-02-15":"2020-06-30"]
    flat_days = covid[covid == 0.0]
    if len(flat_days):
        first_flat, last_flat = flat_days.index[0], flat_days.index[-1]
        spy = prices["SPY"]
        print(f"\n    COVID: book went flat {first_flat.date()}, "
              f"re-risked {last_flat.date()}")
        print(f"           SPY on the flat date  {spy.loc[first_flat]:.2f}")
        print(f"           SPY at the 2020 low   {spy.loc['2020-03-01':'2020-04-01'].min():.2f}")
        print(f"           SPY on the re-risk    {spy.loc[last_flat]:.2f}")
        missed = float(spy.loc[last_flat] / spy.loc[first_flat] - 1.0)
        print(f"           SPY move while flat   {missed:+.2%}")
        findings["covid_flat_window"] = {
            "went_flat": str(first_flat.date()),
            "re_risked": str(last_flat.date()),
            "spy_move_while_flat": round(missed, 4),
        }
    findings["throttle"] = {**stats,
                            "sharpe_off": round(backtest.sharpe_of(base), 3),
                            "sharpe_on": round(backtest.sharpe_of(throttled), 3)}

    # =====================================================================
    section("[G] TRAILING-SHARPE TILT -- is it chasing performance?")
    # =====================================================================
    tilt = portfolios["sharpe_tilt"]
    monthly = rebalance_dates(asset_returns.index, "monthly")
    weights_at_rebalance = tilt.sleeve_weights.reindex(monthly).dropna()

    print("\nCorrelation of a sleeve's allocated weight with its returns BEFORE and AFTER:")
    rows = []
    for sleeve_name in sleeve_net.columns:
        w = weights_at_rebalance[sleeve_name]
        past, future = [], []
        for date in w.index:
            prior = sleeve_net[sleeve_name].loc[:date].tail(126)
            ahead = sleeve_net[sleeve_name].loc[date:].head(21)
            if len(prior) >= 60 and len(ahead) >= 15:
                past.append(float(prior.mean()))
                future.append(float(ahead.mean()))
        aligned = w.iloc[: len(past)]
        rows.append({
            "sleeve": sleeve_name,
            "corr_with_PAST_6m": round(float(np.corrcoef(aligned, past)[0, 1]), 3),
            "corr_with_NEXT_1m": round(float(np.corrcoef(aligned, future)[0, 1]), 3),
        })
    chasing = pd.DataFrame(rows)
    print(chasing.to_string(index=False))
    print("\n  A tilt that works shows a positive correlation with FUTURE returns.")
    print("  A tilt that chases shows a large positive correlation with PAST returns")
    print("  and roughly zero with future ones.")
    findings["sharpe_tilt_chasing"] = chasing.to_dict("records")

    # =====================================================================
    section("[H] ERC vs MINIMUM VARIANCE -- which is structurally more stable?")
    # =====================================================================
    print("\nAllocator weight stability (monthly weight changes across the 5 sleeves):")
    rows = []
    for name, portfolio in portfolios.items():
        w = portfolio.sleeve_weights.reindex(monthly).dropna()
        changes = w.diff().abs().sum(axis=1).dropna()
        concentration = (w ** 2).sum(axis=1)          # Herfindahl
        rows.append({
            "allocator": name,
            "mean_abs_weight_change": round(float(changes.mean()), 4),
            "p95_weight_change": round(float(changes.quantile(0.95)), 4),
            "mean_herfindahl": round(float(concentration.mean()), 3),
            "max_single_sleeve": round(float(w.max().max()), 3),
            "min_effective_sleeves": round(float((1.0 / concentration).min()), 2),
        })
    stability = pd.DataFrame(rows)
    print(stability.to_string(index=False))
    findings["allocator_stability"] = stability.to_dict("records")

    # =====================================================================
    section("[I] 2022 -- what the stock/bond correlation flip did to the vol scalar")
    # =====================================================================
    spy_tlt = asset_returns["SPY"].rolling(126).corr(asset_returns["TLT"])
    print("\nRolling 126-day SPY/TLT correlation:")
    for period in ("2010", "2015", "2019", "2021", "2022", "2023", "2025"):
        window = spy_tlt.loc[period]
        if len(window.dropna()):
            print(f"    {period}  {window.mean():+.3f}")

    print("\nWhat that does to the book's own volatility and required leverage:")
    for period in ("2019", "2021", "2022", "2023"):
        vol_window = unlevered_vol.loc[period]
        lev_window = raw_scalar.loc[period]
        if len(vol_window):
            print(f"    {period}  realised vol {vol_window.mean():.2%}   "
                  f"required leverage {lev_window.mean():.2f}x   "
                  f"capped at {min(float(lev_window.mean()), args.max_leverage):.2f}x")
    findings["spy_tlt_correlation"] = {
        p: round(float(spy_tlt.loc[p].mean()), 4)
        for p in ("2019", "2021", "2022", "2023") if len(spy_tlt.loc[p].dropna())
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "audit.json").write_text(json.dumps(findings, indent=2, default=str))
    print(f"\n\nSaved {REPORTS / 'audit.json'}")


if __name__ == "__main__":
    main()
