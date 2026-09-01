"""A 35-year multi-asset universe, and why the extra history is the whole point.

The binding constraint on everything in this repo has been statistical power,
not ideas. Nineteen years of fifteen ETFs contains two genuine crises, and the
best result anywhere in the study reached t = 2.13 before multiple-testing
correction and t < 1 after it. No amount of additional strategy testing fixes
that; more strategies on the same sample makes it strictly worse, because every
new test raises the bar the winner has to clear.

More data does fix it. Long-lived mutual funds track the same asset classes as
the ETF universe and start ten to twenty-five years earlier, which buys:

* **1991-11 to 2026-09, roughly 35 years** against the ETF sample's 19.
* **Six major regimes instead of two** -- the 1994 bond rout, 1998 LTCM/Asia,
  2000-02 dotcom unwind, 2008, 2020 and 2022 -- rather than 2008 and 2020.
* Most importantly, **1991-2007 has never been looked at**. The ETF study's
  confirmation half has been used several times now (to cut a sleeve, to pick an
  allocator) and is no longer a clean holdout. This period is.

What the substitution costs, stated plainly
-------------------------------------------
**Mutual fund NAVs are not exchange prices.** They are struck once daily at 4pm
ET, and for a fund holding international securities the underlying markets
closed hours earlier. That stale pricing creates genuine, mechanical
autocorrelation in the NAV series: today's US move predicts tomorrow's
international NAV. It is a real effect that real investors were prohibited from
harvesting (it is what triggered the 2003 market-timing scandals and the
fair-value-pricing rules that followed).

It would inflate any SHORT-horizon signal on the international fund. Every
signal tested here is 21 days or longer, where the effect is negligible, and
short-term reversal has already been cut from the book. `VTRIX` carries the
issue; it is flagged rather than quietly used.

**Funds have expense ratios and cash drag**, so their returns understate the
index by 20-80bps a year. That biases *against* finding an edge, not toward it,
which is the direction an honest study prefers to err in.

**Survivorship**: these are funds that still exist in 2026. Selecting on
survival is real, though far weaker for large index-tracking funds from major
families than for, say, a universe of active equity funds.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTENDED_FILE = DATA_DIR / "extended_prices.parquet"

# ticker -> (name, asset_class, note)
EXTENDED_UNIVERSE: dict[str, tuple[str, str, str]] = {
    # --- equity ---
    "VFINX": ("Vanguard 500 Index", "Equity", "US large cap"),
    "NAESX": ("Vanguard Small-Cap Index", "Equity", "US small cap"),
    "VTRIX": ("Vanguard International Value", "Equity",
              "developed ex-US; STALE NAV -- see module docstring"),
    # --- equity sectors, for beta dispersion ---
    "FKUTX": ("Franklin Utilities", "Equity", "low-beta defensive sector"),
    "VGENX": ("Vanguard Energy", "Equity", "high-beta cyclical sector"),
    # --- rates ---
    "VUSTX": ("Vanguard Long-Term Treasury", "Rates", "long duration"),
    "VFITX": ("Vanguard Intermediate Treasury", "Rates", "intermediate duration"),
    "VFISX": ("Vanguard Short-Term Treasury", "Rates", "short duration"),
    # --- credit ---
    "VWESX": ("Vanguard Long-Term Investment Grade", "Credit", "IG spread + duration"),
    "VWEHX": ("Vanguard High-Yield Corporate", "Credit", "HY spread; equity-like in stress"),
    # --- real assets ---
    "OPGSX": ("Invesco Gold & Special Minerals", "RealAsset", "gold complex"),
    "PRNEX": ("T. Rowe Price New Era", "RealAsset", "natural resources"),
}

#: The short-duration Treasury fund is the last to start, and it sets the sample.
EXTENDED_START = "1991-11-01"

#: Where the ETF study's sample begins. Everything before this date has never
#: been examined by any strategy in this repo, which is what makes it a real
#: holdout rather than another pass over familiar data.
ETF_ERA_START = "2007-04-12"

#: Flagged for stale-NAV autocorrelation; excluded from short-horizon tests.
STALE_NAV = {"VTRIX"}


def tickers() -> list[str]:
    return sorted(EXTENDED_UNIVERSE)


def asset_class_map() -> dict[str, str]:
    return {t: meta[1] for t, meta in EXTENDED_UNIVERSE.items()}


def universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"ticker": t, "name": name, "asset_class": klass, "note": note}
         for t, (name, klass, note) in sorted(EXTENDED_UNIVERSE.items())]
    )


def load(refresh: bool = False, allow_fetch: bool = True) -> pd.DataFrame:
    """Daily total-return NAV series for the extended universe."""
    if EXTENDED_FILE.exists() and not refresh:
        return pd.read_parquet(EXTENDED_FILE)

    if not allow_fetch:
        raise FileNotFoundError(
            f"no cached data at {EXTENDED_FILE}; run scripts/fetch_extended.py"
        )

    import yfinance as yf

    series: dict[str, pd.Series] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers():
            data = yf.download(ticker, start="1980-01-01", auto_adjust=True,
                               progress=False)["Close"]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            data.index = pd.to_datetime(data.index).tz_localize(None)
            series[ticker] = data.dropna()

    frame = pd.DataFrame(series).sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(EXTENDED_FILE)
    return frame


def common_sample(prices: pd.DataFrame, start: str = EXTENDED_START) -> pd.DataFrame:
    """Trim to the window where every fund exists.

    A backtest that silently starts with six assets and grows to twelve measures
    the funds' inception dates as much as the strategies.
    """
    trimmed = prices.loc[start:].dropna(axis=0, how="any")
    if trimmed.empty:
        raise ValueError(f"no complete rows from {start}")
    return trimmed
