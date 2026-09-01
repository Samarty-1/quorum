"""Fetch and cache the universe. Run once; every other script reads the cache."""
from __future__ import annotations

import argparse

from src import data
from src.universe import DEFAULT_START


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--refresh", action="store_true", help="re-download over the cache")
    args = ap.parse_args()

    prices, dividends, cash, auxiliary = data.load(args.start, refresh=args.refresh)
    common = data.common_sample(prices)

    print(f"prices     {prices.shape[0]:>6} rows x {prices.shape[1]} assets "
          f"({prices.index.min().date()} to {prices.index.max().date()})")
    print(f"complete   {common.shape[0]:>6} rows x {common.shape[1]} assets "
          f"({common.index.min().date()} to {common.index.max().date()})")
    print(f"dividends  {len(dividends):>6} payments across {dividends['ticker'].nunique()} tickers")
    print(f"cash       {len(cash):>6} rows, latest {100 * cash.iloc[-1]:.2f}%")
    print(f"auxiliary  {auxiliary.shape[0]:>6} rows x {auxiliary.shape[1]} "
          f"roll-proxy series ({', '.join(auxiliary.columns)})")
    print()
    print("first trade date per asset:")
    for ticker in prices.columns:
        print(f"  {ticker:<5} {prices[ticker].first_valid_index().date()}")


if __name__ == "__main__":
    main()
