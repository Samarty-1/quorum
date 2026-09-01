"""Price and distribution data, cached on disk.

The cache is not a convenience. Every number in this repo has to be
reproducible, and yfinance returns slightly different history from one call to
the next -- adjustments get restated, late ticks arrive, the odd day appears or
vanishes. A backtest re-run against silently different data is a backtest whose
results cannot be compared to yesterday's, which makes every "improvement"
unfalsifiable. So the fetch happens once, lands in a parquet file, and every
later run reads that file unless explicitly refreshed.

Total returns, not price returns
--------------------------------
`auto_adjust=True` folds dividends into the price series. This matters more
than usual here: the universe holds bond and REIT ETFs yielding several percent
a year, and a price-only series would show them drifting down for reasons that
have nothing to do with the signals being tested. It would make the carry sleeve
look like it was shorting free money.

Distributions are fetched separately anyway, because the carry sleeve needs the
yield itself as a signal, not just its contribution to return.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.universe import (
    CASH_YIELD,
    DEFAULT_START,
    ROLL_PROXY_FRONT,
    ROLL_PROXY_LADDER,
    tickers,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_FILE = DATA_DIR / "prices.parquet"
DIVIDENDS_FILE = DATA_DIR / "dividends.parquet"
CASH_FILE = DATA_DIR / "cash.parquet"
AUXILIARY_FILE = DATA_DIR / "auxiliary.parquet"


class DataUnavailable(RuntimeError):
    """Raised when a fetch is needed but cannot be performed."""


def _fetch_from_yahoo(start: str):
    import yfinance as yf

    symbols = tickers()
    raw = yf.download(symbols, start=start, auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)

    prices: dict[str, pd.Series] = {}
    dividends: dict[str, pd.Series] = {}
    for symbol in symbols:
        if symbol not in raw.columns.get_level_values(0):
            continue
        series = raw[symbol]["Close"].dropna()
        if len(series) < 250:
            continue
        series.index = pd.to_datetime(series.index).tz_localize(None)
        prices[symbol] = series

        payments = yf.Ticker(symbol).dividends
        if len(payments):
            payments.index = pd.to_datetime(payments.index).tz_localize(None)
        dividends[symbol] = payments

    price_frame = pd.DataFrame(prices).sort_index()

    # Distributions are irregular events, so they are stored long rather than
    # forced onto the trading calendar. Reindexing them here would either
    # forward-fill a payment into days it did not happen on, or drop payments
    # whose ex-date fell on a day some other ticker did not trade.
    rows = []
    for symbol, payments in dividends.items():
        for date, amount in payments.items():
            rows.append({"ticker": symbol, "date": date, "amount": float(amount)})
    dividend_frame = pd.DataFrame(rows, columns=["ticker", "date", "amount"])

    cash = yf.download(CASH_YIELD, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(cash, pd.DataFrame):
        cash = cash.iloc[:, 0]
    cash.index = pd.to_datetime(cash.index).tz_localize(None)
    # ^IRX quotes an annualised percentage; the backtest wants a daily rate.
    cash = (cash.dropna() / 100.0).rename("annual_yield")

    # Auxiliary series: not tradeable universe members, only signal inputs.
    # The front-month and laddered crude funds differ by exactly the roll, so
    # their relative drift measures it (see universe.ROLL_PROXY_FRONT).
    auxiliary: dict[str, pd.Series] = {}
    for symbol in (ROLL_PROXY_FRONT, ROLL_PROXY_LADDER):
        series = yf.download(symbol, start=start, progress=False,
                             auto_adjust=True)["Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series.index = pd.to_datetime(series.index).tz_localize(None)
        auxiliary[symbol] = series.dropna()
    auxiliary_frame = pd.DataFrame(auxiliary).sort_index()

    return price_frame, dividend_frame, cash, auxiliary_frame


def load(start: str = DEFAULT_START, refresh: bool = False, allow_fetch: bool = True):
    """Prices (wide), dividends (long), the cash yield, and auxiliary series.

    Reads the on-disk cache unless `refresh`. Set `allow_fetch=False` in tests
    and CI so a missing cache fails loudly rather than quietly reaching for the
    network and producing a different sample.

    The auxiliary frame holds signal inputs that are NOT tradeable universe
    members -- currently the front-month and laddered crude funds the commodity
    carry signal is built from. Keeping them out of `prices` matters: anything
    in `prices` is something a sleeve may take a position in, and these are
    measurement instruments, not holdings.
    """
    have_cache = (PRICES_FILE.exists() and DIVIDENDS_FILE.exists()
                  and CASH_FILE.exists() and AUXILIARY_FILE.exists())

    if have_cache and not refresh:
        prices = pd.read_parquet(PRICES_FILE)
        dividends = pd.read_parquet(DIVIDENDS_FILE)
        cash = pd.read_parquet(CASH_FILE)["annual_yield"]
        auxiliary = pd.read_parquet(AUXILIARY_FILE)
        return prices, dividends, cash, auxiliary

    if not allow_fetch:
        raise DataUnavailable(
            f"cache incomplete under {DATA_DIR} and allow_fetch=False; "
            "run `python -m scripts.fetch_data` first"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prices, dividends, cash, auxiliary = _fetch_from_yahoo(start)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICES_FILE)
    dividends.to_parquet(DIVIDENDS_FILE)
    cash.to_frame().to_parquet(CASH_FILE)
    auxiliary.to_parquet(AUXILIARY_FILE)
    return prices, dividends, cash, auxiliary


def common_sample(prices: pd.DataFrame, min_assets: int | None = None) -> pd.DataFrame:
    """Trim to the window where the universe is actually complete.

    A backtest that silently starts with four assets and grows to fifteen is
    measuring the universe's inception dates as much as the strategies: early
    "diversification" would be an artifact of having fewer things to be
    correlated with. Better to lose a year of history than to explain that away
    later.
    """
    required = min_assets if min_assets is not None else prices.shape[1]
    complete = prices.notna().sum(axis=1) >= required
    if not complete.any():
        raise DataUnavailable(f"no date has {required} of {prices.shape[1]} assets")
    return prices.loc[complete.idxmax():].dropna(axis=0, how="any")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().iloc[1:]


def daily_cash_rate(cash_annual: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Annualised cash yield converted to a daily rate on the trading calendar.

    Forward-filled, never back-filled: on a day the bill market did not print,
    the last observed yield is what an investor knew.
    """
    aligned = cash_annual.reindex(index.union(cash_annual.index)).ffill().reindex(index)
    return (aligned.fillna(0.0) / 252.0).rename("daily_cash")


def excess_returns(returns: pd.DataFrame, cash_daily: pd.Series) -> pd.DataFrame:
    """Asset returns in excess of the daily bill rate.

    Everything downstream works in excess terms, and the reason is that this
    sample spans 0% (2009-2015, 2020-2021) and 5%+ (2023-2026). Three things
    go wrong on raw total returns:

    * **A dollar-neutral book has no cash drag in excess terms**, which is
      correct -- the shorts fund the longs. On raw returns you have to bolt on
      a short-rebate assumption, and whatever you choose is wrong in one of the
      two rate regimes.
    * **A net-long book earns the bill on its uninvested capital.** In excess
      terms that is automatically zero; on raw returns it has to be added, and
      forgetting to is a real understatement in 2023-2026.
    * **An absolute-momentum filter must compare against cash, not zero.** In
      2007 a +4% twelve-month return was *underperforming* T-bills. A trend
      filter thresholded at zero goes long assets losing to cash, and the sign
      flips in exactly the high-rate periods at both ends of this sample.

    Total return is recovered as excess + cash, so nothing is lost -- but the
    strategy arithmetic happens where the rate regime cannot distort it.
    """
    aligned = cash_daily.reindex(returns.index).fillna(0.0)
    return returns.sub(aligned, axis=0)
