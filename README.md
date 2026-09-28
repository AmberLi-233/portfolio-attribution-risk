# Portfolio Attribution and Risk

Two layers of the same question, in Python.

**Layer 1 — where did the return come from?** Brinson-Fachler attribution with
Cariño geometric linking.

**Layer 2 — where does the risk come from?** A Barra-style cross-sectional
factor risk model: exposures, factor covariance, `V = XFX' + D`, tracking
error decomposition and marginal contribution to risk.

The layers are built to meet. Brinson attributes active return to sectors and
reports whatever is left as stock selection — which quietly absorbs any style
tilt the manager was running. Layer 2 re-runs the same period against a factor
model and shows how much of that "selection" was actually a value, momentum or
size bet. The worked example below is a portfolio whose intended bet was
momentum and whose largest single risk driver turned out to be size.

```bash
pip install -r requirements.txt
python examples/run_demo.py             # layer 1
python examples/run_riskmodel_demo.py   # layer 2
python -m pytest tests/ -q              # 33 tests
```

---

# Layer 1 — Brinson-Fachler attribution

For each sector *s*, with portfolio weight `w_p`, benchmark weight `w_b`,
portfolio sector return `r_p`, benchmark sector return `r_b`, and total
benchmark return `R_B = Σ w_b · r_b`:

| Effect | Formula | Reads as |
|---|---|---|
| Allocation | `(w_p − w_b) · (r_b − R_B)` | Did we overweight sectors that beat the benchmark overall? |
| Selection | `w_b · (r_p − r_b)` | Did we pick better names inside each sector? |
| Interaction | `(w_p − w_b) · (r_p − r_b)` | Did we size up the sectors where our picks worked? |

The three sum, across sectors, to the arithmetic active return `R_P − R_B`.
That identity is enforced as a test, not assumed.

**Why Fachler rather than Brinson-Hood-Beebower.** The `− R_B` term benchmarks
an overweight against the benchmark's own total return instead of against
zero. Without it, overweighting any sector with a positive return scores
positive, even one that badly lagged the index. In a rising market the two
conventions can disagree on the sign of the allocation effect for most
sectors.

### Multi-period linking

Arithmetic effects do not sum across periods, because returns compound. Naive
addition leaves a residual that grows with the horizon and has to be dumped
into an unexplained "other" line. Cariño (1999) scales each period's effects
by `k_t / K`, where

```
k_t = [ln(1+R_P,t) − ln(1+R_B,t)] / (R_P,t − R_B,t)
K   = [ln(1+R_P)   − ln(1+R_B)]   / (R_P   − R_B)
```

After scaling, effects sum exactly to the compounded active return. Where a
period's active return is zero the expression is 0/0 and the limit
`1/(1+R_P,t)` is used. `link="none"` is retained only so the residual can be
measured against the linked result.

### Usage

```python
import pandas as pd
from attribution import single_period_attribution, multi_period_attribution

df = pd.read_csv("data/sample_sector_data.csv")
single_period_attribution(df[df["period"] == "2024-01"].drop(columns=["period"]))
multi_period_attribution(df, link="carino")
```

One row per sector per period, columns `sector, w_p, w_b, r_p, r_b`, returns
as decimals. Column presence, NaNs, duplicate sectors and weight sums are
validated before any arithmetic runs: bad attribution output is almost always
a data problem, not a formula problem, so the check belongs at the boundary.

### Worked example

Six sectors, January 2024, effects in bp:

```
     sector  allocation  selection  interaction  total_effect
 Technology         8.9        7.6          1.1          17.6
 Financials        -0.3       -4.8         -1.4          -6.4
Health Care         3.9        4.3         -1.0           7.3
     Energy        10.3       -3.1          1.0           8.2
Industrials         0.1        2.0          0.2           2.3
   Consumer         1.4       -7.2          0.7          -5.1
      Total        24.3       -1.2          0.7          23.8
```

January's +23.8bp is almost entirely allocation, with selection flat. The
single largest contributor is the Energy underweight: Energy fell 180bp against
a benchmark that rose 162bp, so being 3pp light there earned +10.3bp with no
stock picking at all.

Linked across three months:

```
     sector  allocation  selection  interaction  total_effect
 Technology        -1.7       30.2          3.4          31.9
 Financials        -7.5        8.7          2.5           3.7
Health Care        -0.3       11.8         -2.7           8.8
     Energy         3.0       -4.8          1.6          -0.2
Industrials         0.2       -3.0          0.5          -2.3
   Consumer         1.9       -3.6          0.5          -1.2
      Total        -4.4       39.2          5.8          40.6
```

The quarter tells a different story from the month: +40.6bp, of which selection
is +39.2bp and allocation is slightly negative. Naive summation gives 39.73bp
against the linked 40.63bp — a 0.90bp residual on one quarter of modest
returns, which is the error Cariño removes and which scales with the window.

---

# Layer 2 — Barra-style factor risk model

## The problem it solves

Portfolio risk needs the asset covariance matrix: `σ²_p = w'Vw`. For 3,000
stocks, `V` holds roughly 4.5 million independent parameters, estimated from
maybe 500 days of data. The sample covariance matrix has rank at most 500 and
is mostly noise — the standard error of a pairwise correlation is about
`1/√T ≈ 0.045`, so among 4.5 million pairs, tens of thousands show spurious
correlations above 0.09 purely by chance.

A factor model imposes structure instead:

```
r_i = Σ_k X_ik f_k + e_i        V = X F X' + D
```

with specific returns uncorrelated with the factors and with each other. That
replaces 4.5 million free parameters with `K(K+1)/2` in `F` plus `N` specific
variances — a few thousand. The trade is explicit: estimation error falls
sharply, model error is introduced. Omit a genuine common driver (an oil
factor for energy names) and the model systematically understates those
correlations, and no amount of extra data fixes it.

## Estimation: cross-sectional, not time-series

This is where Barra differs from Fama-French, and it is the part worth being
precise about.

| | Fama-French | Barra-style (here) |
|---|---|---|
| Regression | Time-series, per stock | Cross-sectional, per day |
| Observable | Factor returns `f` | Exposures `X` |
| Estimated | Exposures `β` | Factor returns `f` |
| Exposure updates | Slowly, needs a window | Immediately, when the characteristic changes |
| Observations per fit | ~500 days | ~3,000 stocks |

Exposures are computed, not fitted: style exposures are winsorised,
cap-weighted z-scores of company characteristics (size, value, momentum,
volatility), and industry exposures are 0/1 dummies. Each day, one weighted
least-squares regression across the whole universe estimates that day's factor
returns:

```
f(t) = (X'WX)⁻¹ X'W r(t)
```

Regression weights are square-root market cap — Barra's convention, which
downweights the noisy small-cap tail without letting mega-caps dictate the fit.
`F` is then an EWMA covariance of the estimated factor return history (90-day
half-life) and `D` an EWMA variance of the residuals (60-day half-life,
shorter because specific volatility is more regime-dependent and has no
cross-sectional structure to stabilise it).

**A specification choice worth flagging:** there is no separate market factor.
Industry dummies sum to 1 for every stock, so adding an intercept would make
the design matrix singular. Barra resolves this with a constrained regression
(cap-weighted industry factor returns forced to sum to zero); this
implementation instead lets the industry block absorb the market, which makes
industry factor returns total rather than market-relative. Fitted values and
residuals are identical either way — only the attribution of the market return
between factors changes.

## Validation against a known truth

Real return data has no answer key. If a fitted covariance matrix is wrong
there is no way to tell from the data alone, which is why a factor model
fitted only on live data cannot demonstrate that its estimator works.

`riskmodel/simulate.py` generates a market from known `X`, `F` and `D`, and the
tests check the estimator recovers them. On a 500-stock, 750-day panel:

```
                  true_vol  estimated_vol  error_pct
size                0.1200         0.1140     -5.11
value               0.1200         0.1210      0.59
momentum            0.1200         0.1150     -3.86
volatility          0.1200         0.1200     -0.09
ind_consumer        0.1920         0.1930      0.64
ind_energy          0.1920         0.1980      3.24
ind_financials      0.1920         0.1840     -3.98
ind_health          0.1920         0.1900     -1.23
ind_industrials     0.1920         0.1910     -0.46
ind_tech            0.1920         0.1950      1.58
```

Industry factors are recovered less precisely than styles, and the error
shrinks as the universe grows: each industry factor is identified only by the
stocks in it, so with 500 names and six industries roughly 80 stocks carry each
industry factor against all 500 for a style. On a 100-stock universe the
industry volatilities come out around 10% high.

This validates the estimation machinery, not the modelling assumptions — the
simulated market obeys the factor model by construction. Real markets do not:
fat tails, specific returns correlated within supply chains, exposures that
drift intraquarter.

## Risk decomposition

With `x = X'w`:

```
σ²_p = x' F x  +  w' D w
       ^^^^^^^     ^^^^^^^
    common factor   specific
```

The specific term carries `w_i²`, so it collapses as holdings are added. The
factor term does not, because every stock loads on the same factors. Equally
weighted portfolios from the simulated universe:

```
 n_holdings  total_vol  factor_vol  specific_vol  factor_share
         10     0.2068      0.1749        0.1104        0.7152
         25     0.1522      0.1354        0.0696        0.7911
         50     0.1497      0.1421        0.0470        0.9016
        100     0.1575      0.1541        0.0324        0.9577
        250     0.1481      0.1467        0.0203        0.9813
        500     0.1470      0.1463        0.0145        0.9903
```

Specific volatility falls almost exactly as `1/√N` — 11.0% to 1.5% across a
50-fold increase in holdings — while factor volatility barely moves. Past a
hundred names, essentially all the risk is factor risk, which is why the
interesting risk question for a diversified book is always "which factors" and
never "which stocks".

## The unintended tilt

Take the 100 highest-momentum names, equally weighted, against a cap-weighted
benchmark. The intended bet is momentum. The active exposures:

```
size                 -1.577
momentum              1.421
ind_tech             -0.091
ind_industrials       0.046
ind_energy            0.041
value                -0.029
volatility           -0.009
```

Equal-weighting out of a cap-weighted benchmark bought a 1.58 standard
deviation small-cap bet that nobody chose. Decomposing the 32.4% tracking
error by factor:

```
                  exposure     pct of factor variance
size                 -1.58                      53.3%
momentum              1.42                      48.3%
ind_tech             -0.09                       0.7%
```

**Size contributes more tracking error than momentum does.** Stock selection
accounts for 4.3% of the 32.4% total; 98% of the risk is factor risk. A
Brinson report on this portfolio would have shown a large "selection" effect
and said nothing about size — which is precisely the gap Layer 2 exists to
close.

Marginal contribution to risk then answers what to do about it:
`MCR_i = (Vw)_i / σ_p`, computed without forming `V`. By Euler's theorem
`Σ w_i · MCR_i = σ_p` exactly, which is asserted in the tests and is what makes
contribution-to-risk a decomposition rather than a heuristic. The position to
trim is the one with the highest contribution, not the largest weight or the
most volatile stock.

## Usage

```python
from riskmodel import FactorModel, build_exposures, risk_decomposition
from riskmodel import active_exposure, factor_risk_contributions

X = build_exposures(characteristics, industry, cap=market_cap)
model = FactorModel.fit(returns, X, regression_weights=market_cap**0.5)

risk_decomposition(portfolio, model, benchmark_weights=benchmark)
active_exposure(portfolio, benchmark, model)
factor_risk_contributions(portfolio, model, benchmark)
```

`characteristics` is a DataFrame of raw company characteristics, one row per
stock; `returns` is a T×N daily panel. Any data source works — the simulated
market is used in the tests and the demo because it comes with a known answer.

---

## Limitations

**Layer 1**
- Sector returns taken as given; no intra-period trading or transaction costs
- Equity-style decomposition; fixed income needs duration/curve/spread instead
- Cariño linking implemented; Menchero and GRAP are not

**Layer 2**
- Exposures held fixed across the estimation window rather than rebuilt daily.
  Reasonable over a few months, wrong across a year, and it is what makes the
  estimator testable against a fixed truth
- No Newey-West or Bayesian shrinkage on `F`; a production model needs both,
  plus volatility-regime scaling
- Specific risk is a pure EWMA with no structural model and no shrinkage
  toward a cross-sectional mean, so per-name estimates are noisy
- Specific returns assumed uncorrelated. They are not, in supply chains and in
  merger pairs, and the model understates risk exactly there
- No currency or country factors, so the model is single-market only

## Handling derivatives overlays (design note)

A portfolio running index futures or total return swaps breaks both layers in
the same way, and the fix is not cosmetic.

1. **Work on economic exposure, not cash weights.** A fully invested equity
   sleeve with a short futures overlay has cash weights summing to 1 and an
   economic equity exposure well below it. Cash weights attribute return and
   risk to exposures the portfolio does not have. Restate on a delta-adjusted
   basis before anything else.
2. **Carve the overlay out as its own sleeve.** Hedge P&L is not stock
   selection and must not land in a manager's selection effect. Attribute the
   overlay separately against its own objective — beta target, hedge ratio.
3. **Report hedged and unhedged views.** Which is "the" answer depends on
   whether the overlay decision belonged to the manager or to the allocator.

This is where attribution most often goes wrong in practice, because the
arithmetic still runs and still reconciles — it just attributes to the wrong
place.

## Roadmap

- [ ] Constrained cross-sectional regression with an explicit market factor
- [ ] Newey-West adjustment and eigenvalue shrinkage on `F`
- [ ] Structural specific-risk model
- [ ] Bridge report: run Brinson and factor attribution on the same period side
      by side, and quantify how much "selection" was style
- [ ] Currency attribution for multi-currency portfolios








