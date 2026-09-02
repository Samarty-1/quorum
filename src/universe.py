"""The traded universe: liquid ETFs spanning four asset classes.

Why ETFs and not single stocks
------------------------------
A multi-strategy book needs sleeves that behave differently from each other,
and the cleanest source of that difference is asset class, not stock selection.
Trend following earns its living in bonds and commodities as much as equities;
a single-stock universe would leave most of the sleeves trading the same
underlying risk and would make the diversification question uninteresting.

It also sidesteps two data problems this repo cannot honestly solve on free
data. A single-stock universe assembled today is survivorship-biased -- the
names that delisted are exactly the ones a value or reversal sleeve would have
been long. And the fundamentals a real value sleeve needs are only available
restated, which inflates any backtest that uses them (measured at 0.17 -> 0.27
Sharpe in point-in-time-warehouse). Broad ETFs have neither problem: none of
these delisted, and every signal here is built from price and distributions.

The cost of the choice, stated plainly: these are fifteen instruments, so
cross-sectional sleeves rank a small cross-section and their signals are
noisier than the same idea run over five hundred names.

On the start date
-----------------
Every ticker below trades from 2007-01 or earlier, which is deliberate: the
sample has to contain 2008. A multi-strategy book that has never been through a
correlation shock has not been tested, it has been described.
"""

from __future__ import annotations

import pandas as pd

# ticker -> (name, asset_class, note)
UNIVERSE: dict[str, tuple[str, str, str]] = {
    # --- developed equity ---
    "SPY": ("S&P 500", "Equity", "US large cap"),
    "QQQ": ("Nasdaq 100", "Equity", "US growth tilt"),
    "IWM": ("Russell 2000", "Equity", "US small cap"),
    "EFA": ("MSCI EAFE", "Equity", "developed ex-US"),
    "EWJ": ("MSCI Japan", "Equity", "kept separate from EFA: different rate regime"),
    # --- emerging equity ---
    "EEM": ("MSCI Emerging Markets", "Equity", "EM"),
    # --- rates and credit ---
    "TLT": ("20+ Year Treasury", "Rates", "long duration"),
    "IEF": ("7-10 Year Treasury", "Rates", "intermediate duration"),
    "TIP": ("TIPS", "Rates", "real rates"),
    "LQD": ("Investment Grade Credit", "Credit", "IG spread"),
    "HYG": ("High Yield Credit", "Credit", "HY spread; equity-like in stress"),
    # --- real assets ---
    "GLD": ("Gold", "Commodity", ""),
    "SLV": ("Silver", "Commodity", "higher beta than gold"),
    "DBC": ("Commodity Index", "Commodity", "broad, energy-heavy"),
    "VNQ": ("US REITs", "Real Estate", "rate-sensitive equity"),
}

# The sample must contain 2008. Every ticker above trades by this date.
DEFAULT_START = "2007-01-01"

# --- carry, per asset class -------------------------------------------------
#
# Carry is not one measurement. A dividend yield is the right reading for an
# equity or a REIT, meaningless for gold, and actively misleading for a
# futures-based commodity fund -- ranking DBC on its (zero) distribution yield
# is a statement about the data field, not about carry.
#
# The right reading for a futures-based fund is ROLL YIELD, and the audit
# recorded it as not implementable on free data because DBC's roll is already
# inside its price series. That was wrong: a front-month fund and a
# laddered-maturity fund on the same underlying differ by exactly the roll, so
# their relative drift measures it. USO is front-month WTI, USL a 12-month
# ladder, and both are free and daily from 2007.
#
# DBC is roughly half energy, so crude's term structure is a defensible proxy
# for the broad index's roll. It is a proxy and is labelled as one.
ROLL_PROXY_FRONT = "USO"      # front-month WTI
ROLL_PROXY_LADDER = "USL"     # 12-month ladder, same underlying

# How each universe member earns (or pays) carry.
#   "distribution" -- the trailing dividend/coupon yield is the carry
#   "roll"         -- futures-based; carry is the roll yield
#   "none"         -- physically backed; no coupon and no roll to harvest
CARRY_SOURCE: dict[str, str] = {
    "GLD": "none", "SLV": "none",   # physically backed bullion
    "DBC": "roll",                  # futures-based commodity index
}

# Risk-free proxy for excess returns. BIL only starts 2007-05, so the short end
# comes from the 13-week bill yield instead, which has full history.
CASH_YIELD = "^IRX"


def tickers() -> list[str]:
    return sorted(UNIVERSE)


def asset_class_of(ticker: str) -> str | None:
    entry = UNIVERSE.get(ticker)
    return entry[1] if entry else None


def universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": t, "name": name, "asset_class": klass, "note": note}
            for t, (name, klass, note) in sorted(UNIVERSE.items())
        ]
    )


def asset_classes() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ticker, (_, klass, _) in sorted(UNIVERSE.items()):
        out.setdefault(klass, []).append(ticker)
    return out
