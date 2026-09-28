"""Factor exposure construction.

Barra-style models do not estimate exposures by time-series regression. They
compute them directly from company characteristics, then cross-sectionally
standardise, so that an exposure reads as "this many standard deviations above
the market average" and updates the moment a characteristic changes.

Style exposures here are winsorised then z-scored against the cap-weighted
market mean. Industry exposures are 0/1 dummies.

A deliberate specification choice: there is no separate market/intercept
factor. Industry dummies sum to 1 for every stock, so an intercept would make
the design matrix singular. Barra resolves this with a constrained regression
(cap-weighted industry factor returns forced to sum to zero); the simpler
route taken here is to let the industry block absorb the market, which leaves
industry factor returns interpretable as total industry returns rather than
as returns relative to the market. Both are defensible; they differ in how the
market return is attributed, not in the fitted values or the residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["standardise", "build_exposures", "STYLE_FACTORS"]

STYLE_FACTORS = ["size", "value", "momentum", "volatility"]


def standardise(
    raw: pd.Series,
    cap: pd.Series | None = None,
    winsor: float = 3.0,
) -> pd.Series:
    """Cross-sectionally z-score one characteristic.

    Parameters
    ----------
    raw
        The characteristic for every stock on one date (log market cap, book
        to price, trailing 12-1 return, and so on).
    cap
        Market caps, used to centre on the cap-weighted mean rather than the
        equal-weighted mean. With cap weights the benchmark's exposure is zero
        by construction, which is what makes active exposure equal to raw
        portfolio exposure. Pass None for an equal-weighted centring.
    winsor
        Clip at this many standard deviations before standardising, so a
        single outlier cannot dominate the scale.

    Returns
    -------
    Series of z-scores, mean zero (cap-weighted) and unit dispersion.
    """
    x = raw.astype(float).copy()

    mu = x.mean()
    sd = x.std(ddof=0)
    if sd > 0:
        x = x.clip(mu - winsor * sd, mu + winsor * sd)

    if cap is None:
        centre = x.mean()
    else:
        w = cap.astype(float).reindex(x.index)
        centre = float((w * x).sum() / w.sum())

    spread = x.std(ddof=0)
    if spread == 0:
        return pd.Series(0.0, index=x.index, name=raw.name)

    return ((x - centre) / spread).rename(raw.name)


def build_exposures(
    characteristics: pd.DataFrame,
    industry: pd.Series,
    cap: pd.Series | None = None,
    style_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the N x K exposure matrix X for one date.

    Parameters
    ----------
    characteristics
        Raw characteristics, one row per stock, one column per style.
    industry
        Industry label per stock.
    cap
        Market caps for cap-weighted centring of the styles.
    style_columns
        Which columns of `characteristics` to treat as styles. Defaults to
        every column present that appears in STYLE_FACTORS.

    Returns
    -------
    DataFrame indexed by stock, with standardised style columns followed by
    industry dummy columns prefixed "ind_".
    """
    if style_columns is None:
        style_columns = [c for c in STYLE_FACTORS if c in characteristics.columns]
    missing = [c for c in style_columns if c not in characteristics.columns]
    if missing:
        raise ValueError(f"characteristics is missing style columns: {missing}")

    styles = pd.DataFrame(
        {c: standardise(characteristics[c], cap) for c in style_columns},
        index=characteristics.index,
    )

    dummies = pd.get_dummies(industry.reindex(characteristics.index), prefix="ind")
    dummies = dummies.astype(float)

    return pd.concat([styles, dummies], axis=1)
