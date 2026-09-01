# Quorum

A multi-strategy systematic book: five independent strategy sleeves, five ways
of splitting capital between them, and an honest account of what running more
than one strategy actually bought.

**The short answer: less than advertised.** Diversification across four
standard strategies produced a **+0.11** Sharpe out of sample while simply
running the single best sleeve produced **+0.49**. Vetting once with hindsight
gets the book to **+0.29**; vetting *walk-forward* — the only version available
to someone actually running it — still trails a single sleeve.

A fifth sleeve, short-term reversal, was **cut** after twelve repair variants
were searched on the selection half and the best scored −0.13 on the held-back
half. Removing it took the book from +0.01 to +0.11 and cut turnover from 28 to
6.2 round trips a year — the single largest improvement in the study.

> **Revised after a structural audit** (`scripts/audit.py`, output in
> `reports/audit_output.txt`). The first version ran unit-*gross* sleeves under
> a portfolio-level volatility target, a raw-percentage drawdown ladder, and
> total returns. Repairing the risk layer moved the 1/N book from **−0.26 to
> −0.01**, the vetted book from **+0.09 to +0.13**, and took the trend sleeve to
> **t = 2.13**. The conclusions did not change — the numbers had been depressed
> by defects, not by the idea. See [What the audit changed](#what-the-audit-changed).

## The edge, and its ceiling

The ETF study below asks whether multi-strategy diversification pays. It
mostly does not. A second study (`scripts/research_edge.py`) then asks the
harder question — *is there a tradeable edge here at all* — on **34.8 years**
of 40 markets built from long-lived mutual funds, with **1991–2007 as a
holdout no strategy in this repo had ever seen**.

One thing survives: **trend-following.**

| | Pristine (15.4y) | Familiar (19.4y) | Full 34.8y | t |
|---|---|---|---|---|
| **trend** | **0.960** | 0.387 | **0.638** | **3.33** |
| cross-sectional momentum | 0.496 | 0.264 | 0.368 | 2.17 |
| turn-of-month (Ogden) | 0.435 | −0.149 | 0.113 | 0.60 |
| value | −0.277 | −0.226 | −0.244 | −1.48 |
| betting-against-beta | −0.680 | 0.009 | −0.295 | −1.83 |

Positive in **all seven** 5-year blocks. Survives to 20bps costs. Beats
equal-weight buy-and-hold (0.526) and 60/40 (0.568) at roughly half the
drawdown. Against 52 trials the BHY haircut leaves **0.571, significant at
5%** — though the deflated Sharpe (0.892) still does not clear 0.95.

### This is the ceiling on free data, and the reason is measurable

Two improvements were specified from theory *before* testing, and both failed:

| Universe | Markets | Sharpe | **Effective bets** |
|---|---|---|---|
| narrow core | 12 | 0.614 | 1.35 |
| wide, all | 40 | 0.580 | **1.06** |
| diversifying only | 22 | **0.638** | **1.39** |
| US sectors only | 19 | 0.459 | 1.04 |

**Adding 28 markets made it worse.** Correlation of *trend returns* rose from
+0.204 to +0.253 and effective bets fell, because 18 of the additions were US
equity sectors that trend on one cycle. Sharpe tracks effective bets, not
market count.

The tanh continuous-response signal — the construction the literature actually
uses — delivered its predicted turnover reduction (9.7 vs 11.7 round trips) and
a *lower* Sharpe (0.563 vs 0.638).

**So the binding constraint is data, not code.** Effective bets sit at ~1.4
where a real managed-futures programme runs 8–15, and that gap is the whole
distance between 0.6 and the ~1.0 those programmes report. Closing it needs
currencies, international rates and physical commodity futures — genuinely
independent drivers, none available free with 1991 history.

It is also decaying, consistent with the [~50% post-publication decay the
crowding literature documents](https://arxiv.org/pdf/2512.11913): by 5-year
block 0.540, 0.903, 1.444, 0.397, 0.715, **0.225, 0.269**.

Worth stating plainly: 0.638 on a 35-year, six-regime sample with an untouched
15-year holdout is a better number than a 1.5 from a search nobody can bound.
It is small because it is real.

## The question

"Professional traders use more than one strategy" is true, and it is usually
where the thinking stops. This repo takes the claim apart into three questions
that get answered separately, because they have different answers:

1. Are the strategies actually different from each other?
2. Does *how* you split capital between them matter?
3. Does running several beat running the best one?

## The book

Five sleeves, each a published idea at its literature default — no parameter was
tuned on this sample, so the sleeves do not consume the selection half.

| Sleeve | Idea | Rebalance |
|---|---|---|
| `trend` | Time-series momentum, sign of 12m excess return, inverse-vol sized | monthly |
| `xs_momentum` | Cross-sectional 12−1 month momentum, dollar-neutral | monthly |
| `carry` | Yield for payers, **roll yield** for futures commodities, class-neutralised | monthly |
| `value` | Long-horizon reversal (5y→1y), class-neutralised | quarterly |
| ~~`reversal`~~ | *Cut — see [Why reversal was cut](#why-reversal-was-cut)* | — |

15 liquid ETFs across equity, rates, credit and real assets, 2007-04 to 2026-08
— 4,877 days, deliberately including 2008. Everything runs in **excess returns**
over the daily bill rate. Costs are 5bps per unit turnover, **scaled by trailing
volatility** (mean 5.8bps, p95 11.4bps). Sleeves are volatility-targeted to 10%
individually; the portfolio targets 8%.

Allocators: equal weight (1/N), inverse volatility, risk parity, minimum
variance, a trailing-Sharpe tilt, and an **edge gate** that funds only sleeves
clearing a t-statistic bar on trailing data. All estimated walk-forward on a
trailing two-year window; none ever sees the returns it is about to earn.

## What the sleeves do on their own

| Sleeve | Gross Sharpe | Net Sharpe | t (NW) | Turnover/yr |
|---|---|---|---|---|
| trend | 0.484 | **0.450** | **2.13** | 6.4 |
| xs_momentum | 0.411 | **0.379** | 1.78 | 5.9 |
| carry | −0.391 | −0.425 | −1.99 | 5.9 |
| value | −0.165 | −0.181 | −0.78 | 2.8 |

**Costs are not a rounding error.** The sleeve that used to sit here —
short-term reversal — earned a 0.285 gross Sharpe and handed all of it back
across ~70 round trips a year. Its sign was decided entirely by the cost model,
and a study defaulting to zero costs would have published it as the third-best
idea in the book. It has since been cut outright.

Only trend clears t = 2, and it does not survive a multiple-testing correction
(section 5). That is the honest baseline against which the rest should be read.

## 1. Are the strategies actually different?

Not as different as five names suggests.

```
             trend  xs_momentum  carry  value
trend        1.000        0.658 -0.097  0.091
xs_momentum  0.658        1.000 -0.207  0.079
carry       -0.097       -0.207  1.000  0.376
value        0.091        0.079  0.376  1.000
```

**Trend following and momentum correlate at 0.66.** They appear as separate
entries on every list of trading strategies, and they are built on the same
12-month signal — one takes its sign, the other its cross-sectional rank. Owning
both is close to owning one twice.

Measured properly, the four sleeves are **1.55 independent bets**, not four.

That number is the Meucci effective-number-of-bets: diagonalise the covariance,
express the book as exposures to the resulting uncorrelated factors, take the
entropy of their variance contributions. The obvious alternative — entropy of
each sleeve's marginal risk contribution — is what this repo computed first, and
it reported 4.29 on the original book. It was wrong in the flattering direction:
five *perfectly correlated* sleeves at equal weight each contribute a fifth of
the risk, so that version scores 5 for a book holding one bet. A test asserts it
now returns 1.

## 2. Does the allocator matter?

Barely, and less than any other decision in the study.

**Selection half** (2007-04 to 2016-12), where the allocator was chosen:

| Allocator | Sharpe | t (NW) |
|---|---|---|
| equal_weight | 0.229 | 0.73 |
| inverse_vol | 0.219 | 0.71 |
| risk_parity | 0.129 | 0.40 |
| min_variance | −0.170 | −0.52 |
| **sharpe_tilt** | **0.411** | 1.34 |
| gated/risk_parity | 0.308 | 1.01 |

**Confirmation half** (2016-12 to 2026-08), scored once:

| Allocator | Sharpe |
|---|---|
| equal_weight | +0.109 |
| **sharpe_tilt** *(chosen)* | **+0.150** |
| gated/risk_parity | +0.111 |

The allocator picked on the selection half was not the best on the confirmation
half, and the entire spread from best to worst is **0.10 Sharpe** — far smaller
than the effect of deciding *which sleeves to fund at all*.

This reproduces DeMiguel, Garlappi and Uppal (2009) on a fresh dataset: the
estimation error in an optimiser's inputs costs about as much as the
optimisation gains. Minimum variance, which uses the most estimated quantities,
is worst in both halves — and structurally the least stable, with a p95 monthly
weight change of 0.173 against risk parity's 0.091 and a mean Herfindahl of
0.455 against 0.251. **If you are going to optimise at all, use ERC.**

## 3. Does running five beat running one?

No — and this is the finding.

| Book | Confirmation Sharpe |
|---|---|
| All four sleeves, 1/N | **+0.109** |
| All four, sharpe_tilt | +0.150 |
| Vetted three (drop `carry`), sharpe_tilt | +0.293 |
| **Single best sleeve (`trend`)** | **+0.485** |

Three of the five sleeves lost money out of sample. Diversification reduces
variance; it does not manufacture expected return. Adding a −0.39 Sharpe sleeve
to a +0.48 Sharpe sleeve produces something worse than the good one however
elegantly the capital is split, and every allocator here was reduced to choosing
among mostly-bad options.

Ranking the decisions by how much Sharpe each moved on the confirmation half:

| Decision | Effect |
|---|---|
| Concentrating in the best sleeve | **0.49** |
| Which sleeves to fund (5 vs vetted 4) | **0.13** |
| Which allocator (best vs worst) | 0.10 |
| Walk-forward sleeve gating | **−0.10** |

**Sleeve selection dominates capital allocation.** The sophisticated part of a
multi-strategy system is the least important part of it.

### The honest caveat

"Just run trend" is what this sample says, and the sample is one draw. `trend`
was best on both halves, but with five sleeves that happens 20% of the time by
chance, and its confirmation-half t-statistic is 1.52 — not significant. Sleeve
performance was only partly persistent: three of five kept their sign across the
split, and both `reversal` and `value` flipped from mildly positive to clearly
negative.

The defensible conclusion is not "concentrate". It is: **diversification is not
edge, and a book of five strategies where three have no edge is worse than the
one that does.**

## 3b. Does vetting sleeves work walk-forward? No.

Section 3 vets sleeves *once*, using the whole 9.5-year selection half, and the
book improves to +0.13. That is the version of the argument that flatters it.

`EdgeGated` does the same thing properly: every month, fund only the sleeves
whose trailing two-year Newey-West t-statistic is positive, then allocate over
the survivors with risk parity. Nothing about it peeks.

| Book | Selection | Confirmation |
|---|---|---|
| equal_weight | 0.455 | +0.010 |
| **gated/risk_parity** | **0.217** | **−0.090** |
| Vetted once on the full selection half | — | +0.129 |

**Walk-forward vetting destroys value.** It halves the Sharpe on the selection
half and turns the confirmation half negative, while the identical idea applied
as a single retrospective decision looks like it adds 0.12.

The difference is entirely the decision frequency and the window. Deciding once
with ten years of evidence is a different problem from deciding monthly with
two, and only the second is available to someone actually running the book. A
two-year Sharpe has a standard error near 0.7 — the gate is mostly reacting to
noise, and it is the trailing-Sharpe tilt's failure again in binary form.

This is the audit's own top recommendation — *establish that a sleeve has edge
before funding it* — failing its own test. The recommendation is not wrong, but
it needs evidence the walk-forward window cannot supply. Keeping the allocator
in the default set as a measured negative result is the point.

## Why reversal was cut

The audit said "move it to weekly or cut it". It was already weekly, and it
still lost money, so the question needed answering properly rather than
deferring.

`scripts/reversal_search.py` searches twelve variants — lookbacks of 3, 5 and 10
days, full cross-section against terciles, with and without a no-trade band — on
the **selection half only**, then scores the single best once on the held-back
half.

| | Selection | Confirmation |
|---|---|---|
| Best of 12 variants (5d, full, band 1.0) | +0.153 | **−0.132** |

The search did not even find a configuration better than the one already in use,
and the best variant lost money out of sample. Cutting it was worth more than
any other single change in the study:

| | 5 sleeves | 4 sleeves |
|---|---|---|
| 1/N, confirmation half | +0.010 | **+0.109** |
| Vetted book | +0.129 | **+0.293** |
| Netting benefit | 14.4% | **33.8%** |
| Book turnover | 28.0/yr | **6.2/yr** |

Short-horizon reversal is a single-name microstructure effect — liquidity
provision, bid-ask bounce — and it does not transfer to broad ETFs. Its IC
decays from +0.026 fresh to +0.001 after five days, and neither the weekly
(t = 1.76) nor the monthly-sampled (t = −1.79) estimate clears two standard
errors.

The class stays importable and `all_sleeves()` still returns it, so every
measurement above is reproducible. Keeping it *in the book* because a variant
looked good on the half that chose it is the exact error this repo exists to
avoid.

## Real commodity carry

The audit recorded this as impossible on free data: *"DBC's roll is already
inside its price series and cannot be separated without futures-curve data."*
That was true of DBC alone and false of a **pair**.

USO is a front-month WTI fund; USL is a 12-month ladder on the same underlying.
Two funds differing only in maturity differ by exactly the roll, so their
relative drift measures it — and both are free and daily from 2007.

```
roll yield, annualised:  mean −3.7%   sd 19.0%   min −117%   max +106%
days in contango (negative carry): 62.8%
```

That matches crude's actual history, and it means the carry sleeve now applies
the right measurement to each asset:

| Asset class | Carry measure |
|---|---|
| Equities, credit, rates, REITs | trailing distribution yield |
| Futures-based commodities (DBC) | **roll yield** |
| Physically backed bullion (GLD, SLV) | none — no coupon, no roll |

Previously DBC was excluded entirely (a zero dividend yield is a missing
reading, not zero carry) and before that it was ranked permanently bottom, which
was a statement about the data field rather than about carry. It now carries a
real view.

Its Sharpe did not improve. That is the honest outcome: the sleeve measures what
it claims to measure, and what it measures does not pay on this sample.

## 4. Three things that did work

**Netting sleeves into one book** rather than averaging return streams. When
trend is long SPY and reversal is short it, the netted book trades neither — the
crossing is free. Worth **14.4% of turnover**, a genuine operational edge with
nothing to do with signal quality.

**Volatility-targeting each sleeve before allocating.** This is what makes the
allocator compare like with like: unit *gross* is not unit *risk*, so an
equal-weight allocation over unequal-volatility sleeves was already an implicit
risk bet.

**Class-neutralising the carry and value sleeves.** Before it, carry's mean net
exposure was Equity **−0.279**, Credit **+0.223**, Real assets **+0.116**, with
HYG, VNQ and LQD as its largest persistent longs — a permanent long-credit,
short-US-equity macro position wearing a yield label. After neutralisation those
collapse to **−0.037 / −0.004 / +0.009**. Its Sharpe did not improve; its
*interpretability* did. The sleeve's failure is now a statement about carry
rather than about a hidden equity short.

**Correlations in stress**, for completeness — the folklore is directionally
right and quantitatively small:

| | Calm | SPY drawdown >10% |
|---|---|---|
| Mean pairwise correlation | +0.113 | +0.075 |
| Effective bets | 2.72 | 2.01 |

Effective bets fall by a quarter under stress. The damage to this book came from
sleeves with no edge, not from correlations breaking.

## 5. What survives a multiple-testing correction

Nothing.

24 configurations (4 sleeves × 6 allocators) were tested on one 19-year sample,
plus 12 reversal repair variants — 36 in all.
Reporting the best cell's Sharpe is reporting the maximum of 25 correlated draws.

| Candidate | Raw Sharpe | PSR vs 0 | Threshold | Deflated | BHY haircut | Survives |
|---|---|---|---|---|---|---|
| portfolio: sharpe_tilt | 0.279 | 0.890 | 0.436 | 0.245 | 0.048 | no |
| portfolio: equal_weight | 0.168 | 0.770 | 0.436 | 0.120 | 0.000 | no |
| best sleeve: trend | 0.450 | 0.976 | 0.436 | 0.524 | 0.304 | **no** |

`trend` is the only candidate whose probabilistic Sharpe clears 0.95 against
zero — but the bar is not zero, it is the **0.436** expected maximum of 24
zero-skill trials. Against that bar its deflated Sharpe is 0.524, and a
Benjamini-Yekutieli haircut takes 0.450 down to **0.304**, a 32% cut.

The correct reading: this study establishes *relative* magnitudes — which
decision matters more than which — and does not establish that any sleeve here
has edge.

## What the audit changed

`scripts/audit.py` ran the architecture against its own specification and
measured the failure modes. Seven fixes, in the priority order the audit set.

### P0 — the volatility target was inoperative

The netted five-sleeve book ran at **2.40% median volatility** and needed
**4.15×** leverage to reach a 10% target. With `L_max = 1.5` the cap bound on
**95.3% of days**: the "dynamic leverage scalar" was a constant.

Cause: with unit-gross dollar-neutral sleeves averaged at 1/N, ~51% of gross
cancels on netting, so the book holds ~0.5 gross exposure. A low-volatility book
is the arithmetic consequence of the netting.

Fix: volatility-target each sleeve *before* allocating, plus a realistic `L_max`
and a separate gross cap. Measured:

| Configuration | Realised vol | Cap binds |
|---|---|---|
| 10% target, L_max 1.5, no per-sleeve | 5.30% | **92.0%** |
| 10% target, L_max 1.5, per-sleeve 10% | 7.33% | 85.5% |
| **8% target, L_max 3.0, per-sleeve 10%** | **8.16%** | **2.9%** |

Both changes are needed — per-sleeve targeting alone cannot escape a 1.5 cap.

### P0 — the drawdown ladder's severe rung was untested

The specified 10%/15%/20% ladder **never reached its 20% rung in 19 years**,
because the shallower rungs had already cut risk. The rule most likely to cause
a catastrophic mistake — go to cash, re-enter on an undefined "20-day recovery
trigger" — was untested rather than proven safe. Meanwhile the ladder spent
**39.6% of days de-risked** and cost **3.9pp through 2022**, a year the book
finished up 13%.

Fix (`risk.adaptive_throttle`), four changes each targeting a measured failure:

1. **Volatility-adjusted depth.** 15% off the high is 0.75σ at 20% vol and 1.9σ
   at 8%; only the second is information.
2. **Measured on the shadow (unthrottled) book.** Reading its own throttled
   curve is a ratchet — cutting risk slows recovery, which keeps the drawdown
   deep, which keeps the book cut. Re-entry then needs no timer.
3. **Continuous, not stepped.** Rungs guarantee a book oscillating around a
   threshold trades on every crossing.
4. **A floor, never zero.** A book at zero exposure cannot earn its way back.

| Through | No throttle | Spec ladder | Adaptive |
|---|---|---|---|
| COVID 2020 | −7.26% | −0.22% | −2.21% |
| **Rate hikes 2022** | **+10.56%** | **−2.77%** | **+3.11%** |
| Full-sample Sharpe | 0.239 | 0.246 | **0.256** |
| Max drawdown | −38.0% | −25.2% | −25.6% |

The 2022 row is the whipsaw fix: a 5.9pp swing from recognising that the
drawdown was unremarkable relative to the volatility of the moment.

**It is still off by default.** Over the confirmation half the throttle takes
1/N from −0.005 to −0.158: it is insurance that pays in a genuine crisis and
costs money the rest of the time, and the confirmation half's only crisis was a
V-shaped recovery — the worst case for any de-risking rule.

### P1 — the 126-day covariance window was too slow

Trading days to register a 50% rise in volatility:

| Episode | 21d | 63d | **126d** | 252d |
|---|---|---|---|---|
| GFC 2008 | 11 | 14 | **22** | 32 |
| COVID 2020 | 14 | 21 | **40** | never |
| Volmageddon 2018 | 14 | 15 | **never** | never |
| Rate hikes 2022 | never | never | never | never |

Fix: `risk.asymmetric_volatility` — the max of a 20-day and a 60-day EWMA, so
the book **de-levers fast and re-levers slow**. Volatility clusters, so a quiet
week after a crisis is more likely to be followed by another violent one than
the fast estimate alone implies; re-levering on it is what turns one drawdown
into two.

2022 registers on *no* window, which is the point rather than a failure:
volatility targeting defends against spikes, not slow grinds.

### P1 — the yield and value sleeves were the same macro bet

Covered in section 4. Class-neutralisation collapsed carry's equity exposure
from −0.279 to −0.037.

### P1 — reversal's no-trade band

The signal's IC decays from **+0.026 fresh to +0.001 after five days**, and the
sleeve's own weight autocorrelation is −0.018 at 5 days: the position wanted next
week is uncorrelated with the one held. The band is calibrated from the L1
distance between consecutive target books (median 1.32, p10 0.75) — a band below
~0.75 never binds, which is why an initial 0.60 changed nothing. At 1.0 turnover
falls 67→58 and net Sharpe goes −0.028→+0.085 in isolation.

### P2 — Sharpe tilt lengthened, excess returns, scaled costs, deflation

The 6-month tilt was measurably chasing: allocated weight correlated **+0.13**
with a sleeve's trailing returns and **−0.10** with its next month, with four of
five sleeves negative against the future. A 126-observation Sharpe has a standard
error of ~1.41 Sharpe units. Lengthened to three years; `RiskParity` is now the
documented default.

Everything runs in excess returns (the sample spans 0% and 5%+ policy rates),
costs scale with trailing volatility, and `src/deflated.py` implements the
Bailey–López de Prado deflated Sharpe and Harvey–Liu haircuts.

## Bugs found while fixing

Three, all of the "produces plausible numbers and no error" kind:

- **A second `PortfolioConfig` shadowed the first** in the study script,
  silently dropping both the `--throttle` flag and the volatility-scaled cost
  schedule. Caught only because toggling `--throttle` changed nothing at all.
- **The edge gate never ran.** Its `min_observations` of 756 exceeded the
  engine's 504-day lookback, so `run_portfolio` hit the fallback on every single
  rebalance date and the allocator was byte-identical to equal weight. There is
  now an `Allocator.check_window` that refuses that configuration outright,
  because the failure mode is invisible in the output.
- **`neutralise_within_class` was fed a NaN-filled frame**, so a class holding
  both distributing and non-distributing assets had its mean dragged toward zero
  by the non-payers — a lone commodity ETF that paid a coupon would have been
  measured against a mean a third of its own yield and ranked wildly cheap. It
  does not bite on this universe, where the commodity class pays nothing at all,
  which is exactly why it needed a test rather than an inspection.
- **The per-sleeve volatility target was itself partly inoperative.** At a 4x
  cap the quietest sleeve (carry, 3.08% raw volatility) was pinned below its 10%
  target on 36.9% of days — the portfolio-level defect reproduced one level
  down. Raised to 6x, where it binds on 2.0%.
- **The per-sleeve volatility scalar was applied daily**, re-trading the whole
  book every day even when no signal moved — trend went from 3.4 to 8.3 round
  trips a year, carry from 0.9 to 5.9, pure cost. The scalar now steps on the
  sleeve's own rebalance schedule.
- **The no-trade band could never open the book.** A unit-gross target sits
  exactly 1.0 from flat, so any band above 1.0 held the sleeve at zero forever —
  with a turnover of exactly nothing to give it away.

Earlier passes also produced two measurement errors worth recording: the
covariance-window comparison graded each estimator against *its own* peak (not
comparable across window lengths), and the reversal IC was first quoted only on
monthly sampling, where it is negative, when the better-powered weekly estimate
is positive and neither clears two standard errors.

## Tests

79 tests, no network. The important ones corrupt the future and assert the past
does not move:

```
tests/test_no_lookahead.py   sleeves and the walk-forward allocator, under
                             asset-specific future corruption
tests/test_mechanics.py      the shift, cost timing, netting, allocator
                             properties, shrinkage, effective bets
tests/test_fixes.py          every audit fix, plus the three bugs above
```

Two test bugs worth naming, both of which made a test pass while testing
nothing:

- **The future corruption was uniform.** Multiplying every future price by 3
  leaves cross-sectional *ranks* untouched, so the four dollar-neutral sleeves
  produced byte-identical weights and the leak test was vacuous for them.
- **The shrinkage test used independent columns.** With independent data the
  constant-correlation target is already correct, so Ledoit-Wolf shrinkage
  saturates at 1.0 for every sample size and the property cannot be observed.

## Layout

```
src/universe.py          15 ETFs across four asset classes, and why ETFs
src/data.py              cached fetch, excess returns
src/sleeves/base.py      the rules every sleeve obeys, enforced centrally;
                         class neutralisation, no-trade band
src/sleeves/strategies.py the five sleeves
src/allocators.py        1/N, inverse vol, ERC, min variance, Sharpe tilt
src/risk.py              Ledoit-Wolf shrinkage, EWMA and asymmetric volatility,
                         adaptive throttle, diversification ratio, effective bets
src/portfolio.py         per-sleeve vol targeting, walk-forward combination,
                         netting, risk overlay, gross cap
src/backtest.py          the shift, volatility-scaled costs, metrics
src/deflated.py          deflated Sharpe and multiple-testing haircuts
scripts/run_study.py     the multi-strategy study (15 ETFs, 2007-2026)
scripts/audit.py         the structural audit that drove the risk-layer fixes
scripts/research_edge.py the 35-year edge study, with the 1991-2007 holdout
scripts/reversal_search.py  the search that cut the reversal sleeve
src/extended.py          the 40-market, 34.8-year universe
```

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

python -m scripts.fetch_data          # once; caches to data/
python -m scripts.run_study           # add --throttle, --flat-costs to compare
python -m scripts.audit               # the failure-mode measurements
pytest -q                             # 79 tests
```

Output in `reports/study_output.txt` and `reports/audit_output.txt`; machine-readable
numbers in `reports/study.json` and `reports/audit.json`.

## Limits, stated plainly

- **One sample, one universe.** 19 years of 15 ETFs is a single draw from a
  US-equity-dominated regime. `value` and `carry` were structurally short US
  equity for the whole period, which is most of why they lost.
- **Nothing is statistically significant** after correcting for 25 trials. The
  comparisons are informative about relative magnitudes, not about edge.
- **15 assets is a thin cross-section** for four dollar-neutral sleeves.
- **The value sleeve is a proxy.** Long-horizon reversal is what cross-asset
  factors use when a book value does not exist for gold, but it is not
  fundamental value and should not be read as a test of value investing.
- **No shorting frictions.** Borrow costs and short availability are ignored,
  which flatters the four dollar-neutral sleeves.
- **The commodity carry signal is not real carry.** DBC's roll is already inside
  its price series and cannot be separated without futures-curve data.
