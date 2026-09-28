"""Portfolio risk decomposition on top of a fitted factor model.

Everything here follows from one identity. With portfolio weights w and
portfolio factor exposure x = X'w,

    sigma_p^2 = w' V w = x' F x  +  w' D w
                        ^^^^^^^     ^^^^^^^
                     common factor   specific

The specific term carries w_i squared, so it collapses as holdings are added:
100 equally weighted names put w_i^2 = 1e-4 on each. The factor term does not
collapse, because every stock loads on the same factors. That asymmetry is the
whole reason a diversified portfolio's risk is almost entirely factor risk,
and why the interesting risk question for such a portfolio is always "which
factors", never "which stocks".

Substituting active weights w_p - w_b gives tracking error instead of total
risk; nothing else changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskmodel.model import TRADING_DAYS, FactorModel

__all__ = [
    "portfolio_exposure",
    "active_exposure",
    "risk_decomposition",
    "factor_risk_contributions",
    "marginal_contribution_to_risk",
    "attribute_returns",
]


def _align(weights: pd.Series, model: FactorModel) -> np.ndarray:
    w = weights.reindex(model.exposures.index).fillna(0.0).astype(float)
    extra = set(weights.index) - set(model.exposures.index)
    if extra:
        raise ValueError(f"weights reference assets not in the model: {sorted(extra)}")
    return w.to_numpy()


def portfolio_exposure(weights: pd.Series, model: FactorModel) -> pd.Series:
    """x = X'w, the portfolio's factor exposures.

    A portfolio's exposure is the weighted average of its holdings' exposures.
    Because style exposures are cap-weighted z-scores, a value of +0.5 reads
    as half a standard deviation of tilt against the market.
    """
    w = _align(weights, model)
    return pd.Series(model.exposures.to_numpy(dtype=float).T @ w, index=model.factors)


def active_exposure(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    model: FactorModel,
) -> pd.Series:
    """Portfolio exposure minus benchmark exposure.

    This is the number a manager gets asked about. A manager claiming to be
    style-neutral is making a testable assertion: every style entry here
    should be near zero.
    """
    return portfolio_exposure(portfolio_weights, model) - portfolio_exposure(
        benchmark_weights, model
    )


def risk_decomposition(
    weights: pd.Series,
    model: FactorModel,
    benchmark_weights: pd.Series | None = None,
    annualise: bool = True,
) -> dict:
    """Split portfolio variance into common-factor and specific parts.

    Pass `benchmark_weights` to decompose active risk (tracking error) instead
    of total risk; the arithmetic is identical on active weights.

    Returns
    -------
    dict with keys total_vol, factor_vol, specific_vol, factor_share.
    Volatilities are annualised by default (sqrt of 252).
    """
    w = _align(weights, model)
    if benchmark_weights is not None:
        w = w - _align(benchmark_weights, model)

    X = model.exposures.to_numpy(dtype=float)
    F = model.factor_cov.to_numpy(dtype=float)
    d = model.specific_var.to_numpy(dtype=float)

    x = X.T @ w
    factor_var = float(x @ F @ x)
    specific_var = float(np.sum(w**2 * d))
    total_var = factor_var + specific_var

    scale = TRADING_DAYS if annualise else 1.0

    return {
        "total_vol": float(np.sqrt(total_var * scale)),
        "factor_vol": float(np.sqrt(factor_var * scale)),
        "specific_vol": float(np.sqrt(specific_var * scale)),
        "factor_share": float(factor_var / total_var) if total_var > 0 else np.nan,
    }


def factor_risk_contributions(
    weights: pd.Series,
    model: FactorModel,
    benchmark_weights: pd.Series | None = None,
    annualise: bool = True,
) -> pd.DataFrame:
    """Variance contribution of each factor, summing to the common-factor part.

    Factor k contributes x_k * (F x)_k to variance. Contributions can be
    negative: a factor that hedges the rest of the portfolio reduces total
    variance, and the decomposition shows that honestly rather than taking
    absolute values.

    This is the report that answers "our tracking error is 4%, where is it
    coming from" — the usual answer being one or two style tilts nobody
    intended to take.
    """
    w = _align(weights, model)
    if benchmark_weights is not None:
        w = w - _align(benchmark_weights, model)

    X = model.exposures.to_numpy(dtype=float)
    F = model.factor_cov.to_numpy(dtype=float)

    x = X.T @ w
    contrib = x * (F @ x)
    scale = TRADING_DAYS if annualise else 1.0
    total = contrib.sum()

    return pd.DataFrame(
        {
            "exposure": x,
            "variance_contribution": contrib * scale,
            "pct_of_factor_variance": contrib / total if total != 0 else np.nan,
        },
        index=model.factors,
    ).sort_values("variance_contribution", ascending=False)


def marginal_contribution_to_risk(
    weights: pd.Series,
    model: FactorModel,
    benchmark_weights: pd.Series | None = None,
    annualise: bool = True,
) -> pd.DataFrame:
    """Sensitivity of portfolio volatility to each holding's weight.

    MCR_i = d(sigma_p) / d(w_i) = (V w)_i / sigma_p, computed without forming
    V: (V w) = X F (X'w) + D w.

    By Euler's theorem sigma_p is homogeneous of degree one in w, so
    sum_i w_i * MCR_i = sigma_p exactly. That identity is asserted in the
    tests, and it is what makes "contribution to risk" a decomposition rather
    than a heuristic.

    The practical use is triage: to cut risk fastest, trim the position with
    the highest MCR, not the largest position or the most volatile stock.
    """
    w_total = _align(weights, model)
    w = w_total - _align(benchmark_weights, model) if benchmark_weights is not None else w_total

    X = model.exposures.to_numpy(dtype=float)
    F = model.factor_cov.to_numpy(dtype=float)
    d = model.specific_var.to_numpy(dtype=float)

    Vw = X @ (F @ (X.T @ w)) + d * w
    var = float(w @ Vw)
    if var <= 0:
        raise ValueError("portfolio variance is not positive; check the weights")

    scale = np.sqrt(TRADING_DAYS) if annualise else 1.0
    sigma = np.sqrt(var)
    mcr = Vw / sigma

    return pd.DataFrame(
        {
            "weight": w,
            "mcr": mcr * scale,
            "contribution": w * mcr * scale,
        },
        index=model.exposures.index,
    ).sort_values("contribution", ascending=False)


def attribute_returns(
    weights: pd.Series,
    model: FactorModel,
    benchmark_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Decompose realised return into factor and specific components.

    Active return over the window is x_a' * (cumulative factor returns) plus
    the active-weighted specific returns. This is the bridge to the Brinson
    layer: Brinson attributes to sectors and reports the remainder as stock
    selection, which quietly includes any style tilt. Running the same period
    through this function shows how much of that "selection" was a value or
    momentum bet.

    Returns
    -------
    DataFrame with one row per factor plus a "specific" row and a "total" row,
    in the return units of the estimation window (summed, not compounded).
    """
    if model.factor_returns is None or model.specific_returns is None:
        raise ValueError("model was not fitted with return histories")

    w = _align(weights, model)
    if benchmark_weights is not None:
        w = w - _align(benchmark_weights, model)

    x = model.exposures.to_numpy(dtype=float).T @ w
    cumulative_f = model.factor_returns.sum(axis=0).to_numpy(dtype=float)
    factor_part = x * cumulative_f

    specific_part = float(
        model.specific_returns.sum(axis=0).reindex(model.exposures.index).to_numpy() @ w
    )

    rows = pd.Series(factor_part, index=model.factors)
    rows["specific"] = specific_part
    rows["total"] = rows.sum()

    return rows.to_frame("return_contribution")
