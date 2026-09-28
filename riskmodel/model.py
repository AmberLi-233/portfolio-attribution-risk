"""Barra-style cross-sectional factor risk model.

The model for stock i on day t is

    r_i(t) = sum_k X_ik(t) f_k(t) + e_i(t)

with two assumptions that do all the work: specific returns are uncorrelated
with the factors, and uncorrelated with each other. Under those,

    V = X F X' + D

where F is the K x K factor covariance matrix and D is diagonal with the
specific variances. Estimating V directly for N = 3000 means roughly 4.5m
parameters against maybe 500 days of data; the factor form needs K(K+1)/2 plus
N instead, which is why it is usable at all.

Estimation runs in the opposite direction to Fama-French. There, factor
returns are observable and exposures are regression coefficients. Here
exposures are computed from company characteristics (see exposures.py) and the
factor returns are what the regression estimates, one cross-sectional
regression per day across every stock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["FactorModel", "fit_factor_returns", "ewma_covariance", "ewma_variance"]

TRADING_DAYS = 252


def _half_life_to_lambda(half_life: float) -> float:
    """Decay per period implied by a half-life expressed in periods."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return float(0.5 ** (1.0 / half_life))


def _ewma_weights(n: int, half_life: float) -> np.ndarray:
    """Normalised EWMA weights, most recent observation last."""
    lam = _half_life_to_lambda(half_life)
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = lam**ages
    return w / w.sum()


def fit_factor_returns(
    returns: pd.Series,
    exposures: pd.DataFrame,
    regression_weights: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """One cross-sectional regression: estimate factor returns for one date.

    Solves the weighted least squares problem

        f = (X' W X)^-1 X' W r

    across all stocks on a single date. Barra weights by square-root market
    cap, which downweights the long tail of small names whose returns are
    noisiest without letting the mega-caps determine the fit outright.

    Parameters
    ----------
    returns
        Stock returns on this date, indexed by stock.
    exposures
        N x K exposure matrix for the same date and the same index.
    regression_weights
        Regression weights per stock, typically sqrt(market cap). Equal
        weights if None.

    Returns
    -------
    (factor_returns, specific_returns)
        factor_returns is length K; specific_returns is the residual r - Xf,
        length N.
    """
    idx = exposures.index
    r = returns.reindex(idx).astype(float)
    X = exposures.astype(float)

    if r.isna().any():
        raise ValueError("returns contain NaN for stocks present in exposures")

    if regression_weights is None:
        w = np.ones(len(idx))
    else:
        w = regression_weights.reindex(idx).astype(float).to_numpy()
        if np.any(w <= 0):
            raise ValueError("regression weights must be strictly positive")
    w = w / w.sum()

    Xv = X.to_numpy()
    rv = r.to_numpy()
    XtW = Xv.T * w

    gram = XtW @ Xv
    # Industry dummies plus styles can be near-singular when an industry is
    # thinly populated; lstsq degrades gracefully where a plain inverse would
    # raise or return garbage.
    f, *_ = np.linalg.lstsq(gram, XtW @ rv, rcond=None)

    factor_returns = pd.Series(f, index=X.columns, name=returns.name)
    specific = pd.Series(rv - Xv @ f, index=idx, name=returns.name)
    return factor_returns, specific


def ewma_covariance(panel: pd.DataFrame, half_life: float = 90.0) -> pd.DataFrame:
    """Exponentially weighted covariance of a T x K panel of factor returns.

    Recent observations carry more weight because volatility and correlation
    move. A 90-day half-life is a common choice for a daily factor covariance;
    shorter reacts faster and is noisier.
    """
    if len(panel) < 2:
        raise ValueError("need at least two observations")

    Z = panel.to_numpy(dtype=float)
    w = _ewma_weights(len(panel), half_life)

    mu = w @ Z
    centred = Z - mu
    cov = (centred * w[:, None]).T @ centred
    cov = 0.5 * (cov + cov.T)  # kill floating-point asymmetry

    return pd.DataFrame(cov, index=panel.columns, columns=panel.columns)


def ewma_variance(panel: pd.DataFrame, half_life: float = 60.0) -> pd.Series:
    """Exponentially weighted variance of each column of a T x N panel.

    Used for specific variances. A shorter half-life than the factor
    covariance is usual: specific volatility is more regime-dependent, and
    there is no cross-sectional structure to stabilise it.
    """
    if len(panel) < 2:
        raise ValueError("need at least two observations")

    Z = panel.to_numpy(dtype=float)
    w = _ewma_weights(len(panel), half_life)

    mu = w @ Z
    centred = Z - mu
    var = w @ (centred**2)

    return pd.Series(var, index=panel.columns)


@dataclass
class FactorModel:
    """A fitted factor risk model for one date.

    Attributes
    ----------
    exposures
        N x K exposure matrix, X.
    factor_cov
        K x K factor covariance, F, in the frequency of the input returns.
    specific_var
        Length N specific variances, the diagonal of D.
    factor_returns
        T x K history of estimated factor returns, kept for attribution.
    specific_returns
        T x N history of specific returns.
    """

    exposures: pd.DataFrame
    factor_cov: pd.DataFrame
    specific_var: pd.Series
    factor_returns: pd.DataFrame | None = None
    specific_returns: pd.DataFrame | None = None

    @property
    def factors(self) -> list[str]:
        return list(self.exposures.columns)

    @property
    def assets(self) -> list[str]:
        return list(self.exposures.index)

    def covariance(self) -> pd.DataFrame:
        """Full asset covariance matrix V = X F X' + D.

        Materialising V is fine for a few hundred names and pointless for a
        few thousand: every downstream quantity (portfolio variance, tracking
        error, marginal contributions) can be computed from X, F and D without
        forming an N x N matrix. This exists for testing the identity and for
        small universes.
        """
        X = self.exposures.to_numpy(dtype=float)
        F = self.factor_cov.to_numpy(dtype=float)
        common = X @ F @ X.T
        V = common + np.diag(self.specific_var.to_numpy(dtype=float))
        return pd.DataFrame(V, index=self.exposures.index, columns=self.exposures.index)

    @classmethod
    def fit(
        cls,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        regression_weights: pd.Series | None = None,
        factor_half_life: float = 90.0,
        specific_half_life: float = 60.0,
    ) -> "FactorModel":
        """Estimate the model from a return panel and a fixed exposure matrix.

        Parameters
        ----------
        returns
            T x N panel of stock returns, columns matching the exposure index.
        exposures
            N x K exposure matrix, held fixed across the estimation window.
            Real implementations re-derive exposures every day; holding them
            fixed here keeps the estimator testable against a known truth and
            is a reasonable approximation over a window of a few months.
        regression_weights
            Per-stock regression weights, typically sqrt(market cap).

        Returns
        -------
        A fitted FactorModel. F and D are in the frequency of `returns`;
        annualise with the helpers in decompose.py.
        """
        common = exposures.index.intersection(returns.columns)
        if len(common) < len(exposures.columns) + 1:
            raise ValueError(
                "need more stocks than factors for the cross-sectional regression"
            )

        X = exposures.loc[common]
        R = returns[common]

        f_rows, e_rows = [], []
        for date, row in R.iterrows():
            f, e = fit_factor_returns(row, X, regression_weights)
            f.name = date
            e.name = date
            f_rows.append(f)
            e_rows.append(e)

        factor_returns = pd.DataFrame(f_rows)
        specific_returns = pd.DataFrame(e_rows)

        return cls(
            exposures=X,
            factor_cov=ewma_covariance(factor_returns, factor_half_life),
            specific_var=ewma_variance(specific_returns, specific_half_life),
            factor_returns=factor_returns,
            specific_returns=specific_returns,
        )
