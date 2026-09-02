"""Four charts that carry the findings faster than the tables do.

    python -m scripts.make_charts

Deliberately four, not a gallery. Each one answers a question the README poses
and would otherwise need a paragraph of numbers to settle:

1. Does the edge exist, and does it beat holding the assets?
2. Is it one lucky decade, and is it decaying?
3. Did the multi-strategy premise work?
4. Why did adding markets not help?

Plain matplotlib, no seaborn, no styling beyond what makes the lines readable
in both a light and dark viewer.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import backtest, extended
from src.portfolio import PortfolioConfig, sleeve_books, vol_target_sleeves
from src.risk import ledoit_wolf_covariance, spectral_bets
from src.sleeves.base import SleeveContext
from src.sleeves.strategies import TrendFollowing

warnings.filterwarnings("ignore")
CHARTS = Path(__file__).resolve().parent.parent / "charts"

SECTORS = {"FSPHX", "FSPTX", "FSELX", "FIDSX", "FSRBX", "FSPCX", "FSLBX", "FSCHX",
           "FSDPX", "FSRPX", "FSAVX", "FSHOX", "FSTCX", "FBIOX", "FSENX", "FSUTX",
           "VWNDX", "PRFDX"}

INK = "#1f2a37"
ACCENT = "#0b6bcb"
MUTED = "#8a94a6"
WARN = "#c2410c"


def style(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def build(prices: pd.DataFrame, cost: float = 5.0):
    total = prices.pct_change().iloc[1:]
    cash = total["VFISX"]
    excess = total.sub(cash, axis=0)
    context = SleeveContext(
        prices=prices,
        dividends=pd.DataFrame(columns=["ticker", "date", "amount"]),
        cash_daily=cash,
        asset_class={k: v for k, v in extended.wide_asset_class_map().items()
                     if k in prices.columns},
    )
    sleeve = TrendFollowing()
    config = PortfolioConfig(cost_bps=cost, target_vol=0.08, sleeve_target_vol=0.10)
    books = vol_target_sleeves(sleeve_books([sleeve], context), excess, config, [sleeve])
    return backtest.run(books["trend"], excess, cost).returns, excess


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)

    wide = extended.common_sample(extended.load_wide(allow_fetch=False)[extended.wide_tickers()])
    diversifying = wide[[c for c in wide.columns if c not in SECTORS]]
    trend, excess = build(diversifying)
    split = pd.Timestamp(extended.ETF_ERA_START)

    # ---- 1. the edge, against what you would otherwise hold -----------------
    buy_hold = excess.mean(axis=1)
    sixty_forty = 0.6 * excess["VFINX"] + 0.4 * excess["VUSTX"]

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=140)
    for series, label, colour, width in (
        (trend, f"Trend  (Sharpe {backtest.sharpe_of(trend):.2f})", ACCENT, 1.8),
        (sixty_forty, f"60/40  ({backtest.sharpe_of(sixty_forty):.2f})", MUTED, 1.1),
        (buy_hold, f"Equal-weight buy & hold  ({backtest.sharpe_of(buy_hold):.2f})", "#b0b7c3", 1.1),
    ):
        ax.plot(series.index, (1 + series).cumprod(), label=label, color=colour, lw=width)
    ax.axvline(split, color=WARN, ls="--", lw=1.0, alpha=0.7)
    ax.text(split, ax.get_ylim()[1] * 0.97, "  holdout ends", color=WARN, fontsize=8,
            va="top")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax, "Trend-following vs the alternatives, excess of cash, 1991-2026",
          "growth of 1 (log)")
    ax.text(0.005, -0.16, "Everything left of the dashed line is the holdout: "
            "no strategy in this repo had seen it.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(CHARTS / "01_edge.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 2. is it one lucky decade, and is it decaying? ---------------------
    blocks, labels = [], []
    for start in range(1992, 2027, 5):
        end = min(start + 4, 2026)
        window = trend.loc[str(start):str(end)]
        if len(window) > 200:
            blocks.append(backtest.sharpe_of(window))
            labels.append(f"{start}–{str(end)[2:]}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=140,
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    colours = [ACCENT if b > 0 else WARN for b in blocks]
    ax1.bar(labels, blocks, color=colours, width=0.62)
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.tick_params(axis="x", rotation=45)
    style(ax1, "Sharpe by 5-year block", "Sharpe")

    rolling = trend.rolling(756).mean() / trend.rolling(756).std() * np.sqrt(252)
    ax2.plot(rolling.index, rolling, color=ACCENT, lw=1.4)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.axvline(split, color=WARN, ls="--", lw=1.0, alpha=0.7)
    ax2.fill_between(rolling.index, 0, rolling, where=rolling.notna(),
                     color=ACCENT, alpha=0.12)
    style(ax2, "Rolling 3-year Sharpe — positive throughout, but decaying", "Sharpe")
    fig.tight_layout()
    fig.savefig(CHARTS / "02_decay.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 3. did the multi-strategy premise work? ---------------------------
    import json
    study = json.loads((Path(__file__).resolve().parent.parent
                        / "reports" / "study.json").read_text())
    rows = {r["name"]: r["sharpe"] for r in study["confirmation"]
            if not str(r["name"]).startswith("  sleeve")}
    sleeve_rows = {str(r["name"]).replace("  sleeve: ", ""): r["sharpe"]
                   for r in study["confirmation"] if str(r["name"]).startswith("  sleeve")}

    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=140)
    names = list(sleeve_rows) + ["", "1/N book"]
    values = list(sleeve_rows.values()) + [np.nan, rows.get("equal_weight", np.nan)]
    colours = [ACCENT if (isinstance(v, float) and v > 0) else WARN for v in values]
    ax.bar(range(len(names)), values, color=colours, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.axhline(0, color=INK, lw=0.8)
    style(ax, "The multi-strategy book on its held-back half: "
              "combining sleeves did not rescue them", "Sharpe")
    fig.tight_layout()
    fig.savefig(CHARTS / "03_multistrategy.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 4. why more markets did not help ----------------------------------
    universes = {
        "narrow core\n(12)": wide[[c for c in extended.EXTENDED_UNIVERSE if c in wide.columns]],
        "diversifying\n(22)": diversifying,
        "US sectors\n(19)": wide[[c for c in wide.columns if c in SECTORS or c == "VFISX"]],
        "all\n(40)": wide,
    }
    sharpes, bets, quality = [], [], []
    for prices in universes.values():
        stream, ex = build(prices)
        sharpes.append(backtest.sharpe_of(stream))
        signal = np.sign(prices / prices.shift(252) - 1.0).shift(1)
        per = (signal * ex).dropna(how="all")
        per = per.loc[:, per.std() > 0].dropna()
        covariance, _ = ledoit_wolf_covariance(per)
        bets.append(spectral_bets(covariance))
        quality.append(np.mean([backtest.sharpe_of(per[c]) for c in per.columns]))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=140)
    for ax, values, title, ylabel in (
        (axes[0], sharpes, "Book Sharpe", "Sharpe"),
        (axes[1], bets, "Independent bets available", "spectral N_eff"),
        (axes[2], quality, "Mean PER-MARKET trend Sharpe", "Sharpe"),
    ):
        ax.bar(range(len(universes)), values, color=ACCENT, width=0.6)
        ax.set_xticks(range(len(universes)))
        ax.set_xticklabels(list(universes), fontsize=7.5)
        style(ax, title, ylabel)
    fig.suptitle("More markets gave MORE independent bets and a WORSE book — "
                 "signal quality is the binding variable",
                 fontsize=10.5, color=INK, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(CHARTS / "04_breadth.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote 4 charts to {CHARTS}")
    for path in sorted(CHARTS.glob("*.png")):
        print(f"  {path.name}  {path.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
