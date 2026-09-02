# Engineering log

The defects found and fixed while building this, kept out of the README so the
research reads as research. Every item was measured on the real book, and
several were introduced by the fix for the item above them -- which is the
honest shape of this kind of work.

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

