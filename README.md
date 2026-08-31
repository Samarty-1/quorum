# Quorum

A multi-strategy systematic book: five independent strategy sleeves, five ways
of splitting capital between them, and an honest account of what running more
than one strategy actually bought.

**The short answer: less than advertised, and on this sample, nothing.** Naive
diversification across five standard strategies produced a **−0.26** Sharpe out
of sample while simply running the single best sleeve produced **+0.42**.

## The question

"Professional traders use more than one strategy" is true, and it is usually
where the thinking stops. This repo takes the claim apart into three separate
questions that get answered separately, because they have different answers:

1. Are the strategies actually different from each other?
2. Does *how* you split capital between them matter?
3. Does running several beat running the best one?

## The book

Five sleeves, each a published idea at its literature default — none of the
parameters was tuned on this sample, so the sleeves do not consume the
selection half.

| Sleeve | Idea | Rebalance |
|---|---|---|
| `trend` | Time-series momentum, sign of 12m excess return, inverse-vol sized | monthly |
| `xs_momentum` | Cross-sectional 12-1 month momentum, dollar-neutral | monthly |
| `reversal` | 5-day short-term reversal, dollar-neutral | weekly |
| `carry` | Trailing dividend yield rank, dollar-neutral | monthly |
| `value` | Long-horizon reversal (5y→1y), the cross-asset value proxy | quarterly |

Universe is 15 liquid ETFs across equity, rates, credit and real assets,
2007-04 to 2026-08 — 4,877 days, deliberately including 2008. Costs are 5bps
per unit of turnover and the vol target is 8%.

Allocators: equal weight (1/N), inverse volatility, risk parity, minimum
variance, and a trailing-Sharpe tilt. All estimated walk-forward on a trailing
two-year window; none ever sees the returns it is about to earn.

## What the sleeves do on their own

| Sleeve | Gross Sharpe | Net Sharpe | t (NW) | Turnover/yr |
|---|---|---|---|---|
| trend | 0.367 | **0.343** | 1.67 | 3.4 |
| xs_momentum | 0.231 | **0.209** | 1.10 | 4.4 |
| reversal | 0.296 | **−0.028** | −0.15 | **67.5** |
| carry | −0.253 | −0.260 | −1.28 | 0.9 |
| value | −0.354 | −0.362 | −1.79 | 1.0 |

**Costs are not a rounding error.** Short-term reversal earns a 0.30 gross
Sharpe and hands every basis point of it back in trading — 67 round trips a
year at 5bps. It is the only sleeve whose sign is decided by the cost model, and
a study that defaulted to zero costs would have published it as the second-best
idea in the book.

Note also what the t-statistics say: nothing here is significant. The best
sleeve over 19 years reaches t = 1.67. That is the honest baseline against which
the rest of this should be read.

## 1. Are the strategies actually different?

Not as different as five names suggests.

```
             trend  xs_momentum  reversal  carry  value
trend        1.000        0.740    -0.049 -0.082  0.023
xs_momentum  0.740        1.000    -0.066 -0.095 -0.053
reversal    -0.049       -0.066     1.000 -0.221 -0.148
carry       -0.082       -0.095    -0.221  1.000  0.341
value        0.023       -0.053    -0.148  0.341  1.000
```

**Trend following and momentum correlate at 0.74.** They appear as separate
entries on every list of trading strategies, and they are built on the same
12-month signal — one takes its sign, the other takes its cross-sectional rank.
Owning both is close to owning one twice.

Measured properly, the five sleeves are **3.10 independent bets**, not five.

That number is the Meucci effective-number-of-bets: diagonalise the covariance,
express the book as exposures to the resulting uncorrelated factors, and take
the entropy of their variance contributions. The obvious alternative — entropy
of each sleeve's marginal risk contribution — is what this repo computed first,
and it reported **4.29**. It was wrong, and wrong in the flattering direction:
five *perfectly correlated* sleeves at equal weight each contribute a fifth of
the risk, so that version scores 5 for a book holding one bet. There is a test
asserting it now returns 1.

## 2. Does the allocator matter?

Barely, and less than any other decision in the study.

**Selection half** (2007-04 to 2016-12), where the allocator was chosen:

| Allocator | Sharpe | t (NW) |
|---|---|---|
| equal_weight | 0.355 | 1.18 |
| **inverse_vol** | **0.562** | 1.88 |
| risk_parity | 0.529 | 1.77 |
| min_variance | 0.035 | 0.10 |
| sharpe_tilt | 0.484 | 1.53 |

**Confirmation half** (2016-12 to 2026-08), scored once:

| Allocator | Sharpe | t (NW) |
|---|---|---|
| equal_weight | −0.262 | −0.84 |
| inverse_vol *(chosen)* | −0.215 | −0.67 |
| risk_parity | −0.218 | −0.68 |
| min_variance | −0.104 | −0.32 |
| sharpe_tilt | −0.023 | −0.07 |

Every allocator is negative. The one picked on the selection half was not the
best on the confirmation half, and the whole spread between the best and worst
allocator is **0.24 Sharpe** — smaller than the effect of a single decision
about *which sleeves to fund at all* (below).

This reproduces DeMiguel, Garlappi and Uppal (2009) on a fresh dataset: the
estimation error in an optimiser's inputs costs about as much as the
optimisation gains. Minimum variance, which uses the most estimated quantities,
is the worst on the selection half by a distance.

## 3. Does running five beat running one?

No — and this is the finding.

| Book | Confirmation Sharpe |
|---|---|
| All five sleeves, 1/N | **−0.262** |
| Vetted four (drop `value`), 1/N | +0.091 |
| Vetted four, inverse vol | +0.126 |
| **Single best sleeve (`trend`)** | **+0.415** |

Three of the five sleeves lost money out of sample. Diversification reduces
variance; it does not manufacture expected return. Adding a −0.58 Sharpe sleeve
to a +0.42 Sharpe sleeve produces something worse than the good one, no matter
how elegantly the capital is split, and every allocator here was reduced to
choosing among mostly-bad options.

Ranking the three decisions by how much Sharpe they moved on the confirmation
half:

| Decision | Effect |
|---|---|
| Which sleeves to fund (5 vs vetted 4) | **0.35** |
| Concentrating in the best sleeve | **0.32** |
| Which allocator (best vs worst) | 0.24 |

**Sleeve selection dominates capital allocation.** The sophisticated part of a
multi-strategy system is the least important part of it.

### The honest caveat on that

"Just run trend" is what this sample says, and the sample is one draw. `trend`
was best on both halves, but with five sleeves that happens 20% of the time by
chance alone, and its t-statistic is 1.33 on the confirmation half — not
significant. Sleeve performance was only *partly* persistent: three of five
sleeves kept their sign across the split, and both `reversal` and `carry` went
from mildly positive to clearly negative.

So the defensible conclusion is not "concentrate". It is: **diversification is
not edge, and a book of five strategies where three have no edge is worse than
the one that does.**

## 4. Two things that did work

**Netting the sleeves into one book** rather than averaging their return
streams. When trend is long SPY and reversal is short it, the netted book trades
neither — the crossing is free. Worth **7.7% of turnover** here, and it is a
genuine operational edge with nothing to do with signal quality. Modest because
these sleeves mostly trade different things; it would be much larger on a book
of same-asset strategies.

**Correlations rose in stress**, exactly as the folklore says, and the effect is
small enough to be worth quantifying rather than repeating:

| | Calm | SPY drawdown >10% |
|---|---|---|
| Mean pairwise correlation | −0.014 | +0.079 |
| Effective bets | 2.66 | 2.72 |

Correlations do rise. But on this book the diversification loss is marginal —
the sleeves that hurt were the ones with no edge, not the ones that stopped
diversifying.

## Method

- **Selection/confirmation split by date, decided once.** The allocator is
  chosen on the first half; the second half is scored once and not returned to.
  Picking the best of five allocators on the full sample and reporting its
  number is publishing the maximum of five noisy draws.
- **Walk-forward allocation.** Each allocator sees a trailing two-year window
  ending on the decision date. Nothing after it can reach back.
- **Costs charged on turnover**, on the day the book changes, including the
  initial trade from flat.
- **Newey-West t-statistics.** A monthly-rebalanced book holds the same position
  for twenty days, so daily returns are autocorrelated and the plain
  t-statistic overstates the evidence.
- **Total returns**, so the bond and REIT sleeves are not penalised for paying
  their yield out.

## Tests

52 tests. The important ones corrupt the future and assert the past does not
move:

```
tests/test_no_lookahead.py   sleeves and the walk-forward allocator, under
                             asset-specific future corruption
tests/test_mechanics.py      the shift, cost timing, netting, allocator
                             properties, shrinkage, effective bets
```

Two test bugs worth naming, both of which made a test pass while testing
nothing:

- **The future corruption was uniform.** Multiplying every future price by 3
  leaves every cross-sectional *rank* untouched, so the four dollar-neutral
  sleeves produced byte-identical weights and the leak test was vacuous for
  them. The corruption is now asset-specific.
- **The shrinkage test used independent columns.** With genuinely independent
  data the constant-correlation target is already correct, so Ledoit-Wolf
  shrinkage saturates at 1.0 for every sample size and "shrinkage rises when
  data is scarce" cannot be observed. It needs heterogeneous correlation, and
  now uses two blocks.

## Layout

```
src/universe.py          15 ETFs across four asset classes, and why ETFs
src/data.py              cached fetch; the cache is for reproducibility
src/sleeves/base.py      the three rules every sleeve obeys, enforced centrally
src/sleeves/strategies.py the five sleeves
src/allocators.py        1/N, inverse vol, risk parity, min variance, Sharpe tilt
src/risk.py              Ledoit-Wolf shrinkage, vol targeting, drawdown throttle,
                         diversification ratio, effective bets
src/portfolio.py         walk-forward combination, netting, the risk overlay
src/backtest.py          the shift, the costs, the metrics
scripts/run_study.py     the whole study
```

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

python -m scripts.fetch_data      # once; caches to data/
python -m scripts.run_study
pytest -q                         # 52 tests, no network
```

Full output is in `reports/study_output.txt`; the numbers above come from
`reports/study.json`.

## Limits, stated plainly

- **One sample, one universe.** 19 years of 15 ETFs is a single draw from a
  regime that was dominated by US equities. `value` and `carry` were
  structurally short US tech for the whole period, which is most of why they
  lost. A different sample could reverse the ranking of every sleeve.
- **Nothing here is statistically significant.** The best result in the study is
  t = 1.67. The comparisons are informative about *relative* magnitudes — which
  decision matters more — not about whether any sleeve has an edge.
- **15 assets is a thin cross-section.** The four dollar-neutral sleeves rank
  fifteen things; the same ideas over 500 names would be less noisy and might
  well work better.
- **One cost assumption.** 5bps per unit turnover is reasonable for liquid ETFs
  and is applied uniformly; a real book faces spreads that widen exactly when
  the sleeves want to trade.
- **The value sleeve is a proxy.** Long-horizon reversal is what cross-asset
  factors use when a book value does not exist for gold, but it is not
  fundamental value, and it should not be read as a test of value investing.
- **No shorting frictions.** Borrow costs and short availability are ignored,
  which flatters the four dollar-neutral sleeves.
