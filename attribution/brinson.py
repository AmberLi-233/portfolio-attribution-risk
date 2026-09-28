"""Brinson-Fachler performance attribution.

Single-period decomposition of active return into allocation, selection and
interaction effects, plus Carino multi-period linking.

Notation, per sector s:
    w_p  portfolio weight
    w_b  benchmark weight
    r_p  portfolio return within the sector
    r_b  benchmark return within the sector
    R_B  total benchmark return = sum_s w_b * r_b

Effects (Brinson-Fachler, 1985):
    allocation  = (w_p - w_b) * (r_b - R_B)
    selection   =  w_b       * (r_p - r_b)
    interaction = (w_p - w_b) * (r_p - r_b)

These sum, across sectors, to the arithmetic active return R_P - R_B.
The Fachler refinement is the `- R_B` term in the allocation effect: it scores
an overweight against the benchmark's own total return rather than against
zero, so overweighting a sector is only rewarded when that sector beat the
benchmark overall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["sector", "w_p", "w_b", "r_p", "r_b"]

__all__ = [
    "single_period_attribution",
    "multi_period_attribution",
    "carino_factors",
]


def _validate(df: pd.DataFrame, weight_tol: float = 1e-6) -> None:
    """Raise if the frame is missing columns, has NaNs, or weights do not sum to 1."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        bad = df[df[REQUIRED_COLUMNS].isna().any(axis=1)]["sector"].tolist()
        raise ValueError(f"NaN values in sectors: {bad}")

    if df["sector"].duplicated().any():
        dupes = df.loc[df["sector"].duplicated(), "sector"].tolist()
        raise ValueError(f"duplicate sectors in a single period: {dupes}")

    for col in ("w_p", "w_b"):
        total = df[col].sum()
        if abs(total - 1.0) > weight_tol:
            raise ValueError(f"{col} sums to {total:.6f}, expected 1.0")


def single_period_attribution(
    df: pd.DataFrame,
    validate: bool = True,
) -> pd.DataFrame:
    """Decompose one period's active return into Brinson-Fachler effects.

    Parameters
    ----------
    df
        One row per sector with columns sector, w_p, w_b, r_p, r_b.
        Returns are simple (not log) and expressed as decimals: 0.023 = 2.3%.
    validate
        Check columns, NaNs, duplicate sectors and that both weight vectors
        sum to one. Turn off only for speed in a loop over pre-checked data.

    Returns
    -------
    DataFrame
        One row per sector plus a final "Total" row, with columns
        allocation, selection, interaction, total_effect.
        The Total row's total_effect equals R_P - R_B by construction.
    """
    if validate:
        _validate(df)

    out = df[["sector"]].copy()

    active_weight = df["w_p"] - df["w_b"]
    active_return = df["r_p"] - df["r_b"]
    benchmark_total = float((df["w_b"] * df["r_b"]).sum())

    out["allocation"] = active_weight * (df["r_b"] - benchmark_total)
    out["selection"] = df["w_b"] * active_return
    out["interaction"] = active_weight * active_return
    out["total_effect"] = out[["allocation", "selection", "interaction"]].sum(axis=1)

    return _append_total_row(out)


def _append_total_row(out: pd.DataFrame) -> pd.DataFrame:
    """Append a Total row, preserving float dtypes on the effect columns."""
    effects = ["allocation", "selection", "interaction", "total_effect"]
    total = pd.DataFrame(
        {"sector": ["Total"], **{c: [float(out[c].sum())] for c in effects}}
    )
    return pd.concat([out[["sector"] + effects], total], ignore_index=True)


def carino_factors(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[pd.Series, float]:
    """Carino (1999) smoothing factors for geometric multi-period linking.

    Arithmetic effects do not sum across periods, because returns compound.
    Carino's fix scales each period's effects by k_t / K, where

        k_t = [ln(1+R_P,t) - ln(1+R_B,t)] / (R_P,t - R_B,t)
        K   = [ln(1+R_P)   - ln(1+R_B)]   / (R_P   - R_B)

    with R_P, R_B the geometrically compounded returns over the full window.
    When a period's active return is zero the expression is 0/0; the limit is
    1 / (1 + R_P,t), which is what is used there.

    Returns
    -------
    (k, K)
        k is a Series of per-period factors indexed like the inputs;
        K is the scalar full-period factor.
    """
    rp = portfolio_returns.astype(float)
    rb = benchmark_returns.astype(float)

    active = rp - rb
    k = np.where(
        np.abs(active) < 1e-12,
        1.0 / (1.0 + rp),
        (np.log1p(rp) - np.log1p(rb)) / np.where(active == 0, 1.0, active),
    )
    k = pd.Series(k, index=rp.index)

    total_p = float(np.prod(1.0 + rp) - 1.0)
    total_b = float(np.prod(1.0 + rb) - 1.0)
    total_active = total_p - total_b

    if abs(total_active) < 1e-12:
        big_k = 1.0 / (1.0 + total_p)
    else:
        big_k = (np.log1p(total_p) - np.log1p(total_b)) / total_active

    return k, float(big_k)


def multi_period_attribution(
    df: pd.DataFrame,
    period_col: str = "period",
    link: str = "carino",
) -> pd.DataFrame:
    """Attribute across several periods, linking effects so they sum correctly.

    Parameters
    ----------
    df
        One row per sector per period, with the single-period columns plus
        `period_col`. Periods are processed in order of first appearance.
    link
        "carino"  geometric linking; effects sum to the compounded active
                  return (1+R_P)/(1+R_B) - 1 style difference R_P - R_B.
        "none"    naive summation of arithmetic effects. Kept for comparison
                  only; the residual it leaves is the compounding error.

    Returns
    -------
    DataFrame
        One row per sector plus a Total row, with linked allocation,
        selection, interaction and total_effect.
    """
    if link not in ("carino", "none"):
        raise ValueError(f"unknown link method: {link}")

    periods = list(dict.fromkeys(df[period_col]))
    per_period: dict = {}
    rp_list, rb_list = [], []

    for p in periods:
        sub = df[df[period_col] == p]
        per_period[p] = single_period_attribution(sub)
        rp_list.append(float((sub["w_p"] * sub["r_p"]).sum()))
        rb_list.append(float((sub["w_b"] * sub["r_b"]).sum()))

    rp = pd.Series(rp_list, index=periods)
    rb = pd.Series(rb_list, index=periods)

    if link == "carino":
        k, big_k = carino_factors(rp, rb)
        scale = {p: k[p] / big_k for p in periods}
    else:
        scale = {p: 1.0 for p in periods}

    effects = ["allocation", "selection", "interaction", "total_effect"]
    stacked = []
    for p in periods:
        block = per_period[p]
        block = block[block["sector"] != "Total"].copy()
        block[effects] = block[effects] * scale[p]
        stacked.append(block)

    combined = pd.concat(stacked, ignore_index=True)
    out = combined.groupby("sector", as_index=False, sort=False)[effects].sum()

    return _append_total_row(out)
