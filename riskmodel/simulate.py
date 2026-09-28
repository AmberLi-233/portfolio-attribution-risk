"""Synthetic market with known parameters.

Real return data has no answer key: if a fitted covariance matrix is wrong
there is no way to tell from the data alone. Simulating a market whose X, F
and D are known makes the estimator falsifiable — feed it data generated from
a truth and check it recovers that truth within sampling error.

The generated panel follows the model exactly:

    r(t) = X f(t) + e(t),   f(t) ~ N(0, F),   e_i(t) ~ N(0, d_i)

so it validates the estimation machinery, not the modelling assumptions. Real
markets violate those assumptions — fat tails, correlated specific returns
inside supply chains, exposures that drift — which is what the roadmap in the
README is about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskmodel.exposures import build_exposures

__all__ = ["SyntheticMarket", "simulate_market"]

INDUSTRIES = ["tech", "financials", "energy", "health", "industrials", "consumer"]


@dataclass
class SyntheticMarket:
    """A simulated panel together with the parameters that generated it."""

    returns: pd.DataFrame          # T x N
    exposures: pd.DataFrame        # N x K
    true_factor_cov: pd.DataFrame  # K x K
    true_specific_var: pd.Series   # N
    true_factor_returns: pd.DataFrame  # T x K
    market_cap: pd.Series          # N


def simulate_market(
    n_assets: int = 200,
    n_periods: int = 750,
    seed: int = 42,
    factor_vol_annual: float = 0.12,
    specific_vol_annual: tuple[float, float] = (0.18, 0.45),
    trading_days: int = 252,
) -> SyntheticMarket:
    """Generate a market that obeys the factor model exactly.

    Parameters
    ----------
    n_assets, n_periods
        Universe size and length of the daily return panel.
    factor_vol_annual
        Typical annualised volatility of a factor; industry factors are set
        higher than styles, as in real estimates.
    specific_vol_annual
        Range of annualised specific volatilities drawn across the universe.

    Returns
    -------
    SyntheticMarket with the panel and every true parameter.
    """
    rng = np.random.default_rng(seed)

    assets = [f"STK{i:04d}" for i in range(n_assets)]
    industry = pd.Series(rng.choice(INDUSTRIES, size=n_assets), index=assets)

    # Log-normal caps give the fat right tail of a real universe.
    market_cap = pd.Series(
        np.exp(rng.normal(loc=9.0, scale=1.4, size=n_assets)), index=assets
    )

    characteristics = pd.DataFrame(
        {
            "size": np.log(market_cap.to_numpy()),
            "value": rng.normal(size=n_assets),
            "momentum": rng.normal(size=n_assets),
            "volatility": rng.normal(size=n_assets),
        },
        index=assets,
    )

    X = build_exposures(characteristics, industry, cap=market_cap)
    factors = list(X.columns)
    k = len(factors)

    # Factor covariance: correlated block with industries more volatile than
    # styles, built from a random correlation matrix via a low-rank + diagonal
    # construction so it is positive definite by construction.
    daily = factor_vol_annual / np.sqrt(trading_days)
    vols = np.array(
        [daily if f in characteristics.columns else daily * 1.6 for f in factors]
    )
    loadings = rng.normal(size=(k, 3)) * 0.45
    # Low-rank plus a strictly positive diagonal is positive definite for any
    # loadings; the floor matters because three random loadings can sum to
    # more than unit variance, which would leave a negative residual.
    residual = np.maximum(1.0 - np.sum(loadings**2, axis=1), 0.15)
    corr = loadings @ loadings.T + np.diag(residual)
    dsqrt = np.sqrt(np.diag(corr))
    corr = corr / np.outer(dsqrt, dsqrt)
    corr = 0.5 * (corr + corr.T)
    F = np.outer(vols, vols) * corr

    f_draws = rng.multivariate_normal(np.zeros(k), F, size=n_periods)
    factor_returns = pd.DataFrame(f_draws, columns=factors)

    lo, hi = specific_vol_annual
    specific_vol_daily = rng.uniform(lo, hi, size=n_assets) / np.sqrt(trading_days)
    specific = rng.normal(scale=specific_vol_daily, size=(n_periods, n_assets))

    returns = pd.DataFrame(
        f_draws @ X.to_numpy(dtype=float).T + specific, columns=assets
    )

    return SyntheticMarket(
        returns=returns,
        exposures=X,
        true_factor_cov=pd.DataFrame(F, index=factors, columns=factors),
        true_specific_var=pd.Series(specific_vol_daily**2, index=assets),
        true_factor_returns=factor_returns,
        market_cap=market_cap,
    )
