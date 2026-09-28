import numpy as np
import pandas as pd
import pytest

from riskmodel import (
    FactorModel,
    active_exposure,
    attribute_returns,
    build_exposures,
    factor_risk_contributions,
    marginal_contribution_to_risk,
    portfolio_exposure,
    risk_decomposition,
    simulate_market,
    standardise,
)
from riskmodel.model import TRADING_DAYS


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=300, n_periods=750, seed=11)


@pytest.fixture(scope="module")
def model(market):
    return FactorModel.fit(
        market.returns,
        market.exposures,
        regression_weights=market.market_cap**0.5,
    )


def equal_weights(model, n=None):
    assets = model.assets if n is None else model.assets[:n]
    return pd.Series(1.0 / len(assets), index=assets)


def cap_weights(market):
    return market.market_cap / market.market_cap.sum()


# ---------------------------------------------------------------- exposures


def test_standardise_is_cap_weighted_mean_zero():
    rng = np.random.default_rng(0)
    raw = pd.Series(rng.normal(size=500))
    cap = pd.Series(np.exp(rng.normal(size=500)))
    z = standardise(raw, cap)
    assert float((cap * z).sum() / cap.sum()) == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=0) == pytest.approx(1.0, rel=0.01)


def test_standardise_clips_outliers():
    raw = pd.Series([0.0] * 99 + [1000.0])
    z = standardise(raw, cap=None, winsor=3.0)
    assert z.abs().max() < 20


def test_industry_dummies_sum_to_one(market):
    ind_cols = [c for c in market.exposures.columns if c.startswith("ind_")]
    row_sums = market.exposures[ind_cols].sum(axis=1)
    assert row_sums.min() == pytest.approx(1.0)
    assert row_sums.max() == pytest.approx(1.0)


def test_build_exposures_rejects_missing_style():
    chars = pd.DataFrame({"value": [0.1, 0.2]}, index=["A", "B"])
    ind = pd.Series(["tech", "energy"], index=["A", "B"])
    with pytest.raises(ValueError, match="missing style columns"):
        build_exposures(chars, ind, style_columns=["value", "momentum"])


# ------------------------------------------------------------- estimation


def test_cross_sectional_regression_recovers_factor_returns(market, model):
    """The estimator must reproduce the factor returns that generated the data."""
    for f in model.factors:
        corr = np.corrcoef(model.factor_returns[f], market.true_factor_returns[f])[0, 1]
        assert corr > 0.80, f"{f} recovered with corr {corr:.3f}"


def test_factor_volatilities_recovered_within_tolerance(market, model):
    true_vol = np.sqrt(np.diag(market.true_factor_cov.to_numpy()) * TRADING_DAYS)
    est_vol = np.sqrt(np.diag(model.factor_cov.to_numpy()) * TRADING_DAYS)
    relative_error = np.abs(est_vol / true_vol - 1.0)
    assert relative_error.max() < 0.25
    assert relative_error.mean() < 0.12


def test_specific_variance_recovered(market, model):
    true_vol = np.sqrt(market.true_specific_var * TRADING_DAYS)
    est_vol = np.sqrt(model.specific_var * TRADING_DAYS)
    # Per-name specific vol is noisy; the cross-sectional average should not be.
    assert float(est_vol.mean() / true_vol.mean()) == pytest.approx(1.0, rel=0.10)


def test_residuals_are_orthogonal_to_exposures(market, model):
    """A least-squares residual is orthogonal to the design matrix by construction.

    This is not a modelling assumption being tested, it is a check that the
    regression is actually solving what it claims to solve.
    """
    w = (market.market_cap**0.5).reindex(model.assets).to_numpy()
    w = w / w.sum()
    X = model.exposures.to_numpy(dtype=float)
    e = model.specific_returns.to_numpy(dtype=float)
    weighted = (X.T * w) @ e.T
    assert np.abs(weighted).max() < 1e-10


def test_needs_more_stocks_than_factors(market):
    tiny = market.exposures.iloc[:3]
    with pytest.raises(ValueError, match="more stocks than factors"):
        FactorModel.fit(market.returns[tiny.index], tiny)


# --------------------------------------------------------- the V identity


def test_covariance_matches_xfx_plus_d(model):
    X = model.exposures.to_numpy(dtype=float)
    F = model.factor_cov.to_numpy(dtype=float)
    D = np.diag(model.specific_var.to_numpy(dtype=float))
    assert np.allclose(model.covariance().to_numpy(), X @ F @ X.T + D, atol=1e-14)


def test_portfolio_variance_agrees_with_full_covariance(model):
    w = equal_weights(model)
    V = model.covariance().to_numpy()
    direct = float(w.to_numpy() @ V @ w.to_numpy() * TRADING_DAYS)
    decomposed = risk_decomposition(w, model)["total_vol"] ** 2
    assert decomposed == pytest.approx(direct, rel=1e-10)


# ------------------------------------------------------------ decomposition


def test_factor_and_specific_variance_sum_to_total(model):
    d = risk_decomposition(equal_weights(model), model)
    assert d["factor_vol"] ** 2 + d["specific_vol"] ** 2 == pytest.approx(
        d["total_vol"] ** 2, rel=1e-12
    )


def test_specific_risk_diversifies_away_but_factor_risk_does_not(model):
    """The asymmetry that makes factor models worth building.

    Specific variance carries w_i squared and collapses roughly as 1/N.
    Factor risk does not, because every holding loads on the same factors.
    """
    small = risk_decomposition(equal_weights(model, 10), model)
    large = risk_decomposition(equal_weights(model, 250), model)

    ratio = (small["specific_vol"] / large["specific_vol"]) ** 2
    assert ratio == pytest.approx(25.0, rel=0.6)  # 250/10, sampling noise aside

    assert large["factor_vol"] > 0.5 * small["factor_vol"]
    assert large["factor_share"] > small["factor_share"]
    assert large["factor_share"] > 0.9


def test_factor_contributions_sum_to_factor_variance(model):
    w = equal_weights(model)
    d = risk_decomposition(w, model)
    contrib = factor_risk_contributions(w, model)
    assert contrib["variance_contribution"].sum() == pytest.approx(
        d["factor_vol"] ** 2, rel=1e-10
    )


def test_euler_identity_for_marginal_contributions(model):
    """sum_i w_i * MCR_i = sigma_p, exactly. Without this, "contribution to
    risk" would be a story rather than a decomposition."""
    w = equal_weights(model)
    mcr = marginal_contribution_to_risk(w, model)
    sigma = risk_decomposition(w, model)["total_vol"]
    assert mcr["contribution"].sum() == pytest.approx(sigma, rel=1e-10)


def test_tracking_error_is_zero_when_portfolio_matches_benchmark(market, model):
    b = cap_weights(market)
    te = risk_decomposition(b, model, benchmark_weights=b)
    assert te["total_vol"] == pytest.approx(0.0, abs=1e-12)


def test_benchmark_has_zero_active_exposure_to_itself(market, model):
    b = cap_weights(market)
    xa = active_exposure(b, b, model)
    assert xa.abs().max() == pytest.approx(0.0, abs=1e-12)


def test_cap_weighted_benchmark_is_style_neutral(market, model):
    """Style exposures are cap-weighted z-scores, so the cap-weighted
    benchmark sits at zero on every style by construction. This is what makes
    a portfolio's raw style exposure readable as an active bet."""
    x = portfolio_exposure(cap_weights(market), model)
    styles = [f for f in model.factors if not f.startswith("ind_")]
    assert x[styles].abs().max() < 1e-10


def test_deliberate_tilt_shows_up_as_active_exposure(market, model):
    """Tilt toward high-momentum names and the momentum exposure must move."""
    mom = model.exposures["momentum"]
    top = mom.sort_values(ascending=False).index[:60]
    tilted = pd.Series(1.0 / len(top), index=top)

    xa = active_exposure(tilted, cap_weights(market), model)
    assert xa["momentum"] > 0.8

    contrib = factor_risk_contributions(tilted, model, cap_weights(market))
    assert contrib.loc["momentum", "variance_contribution"] > 0


# -------------------------------------------------------------- attribution


def test_return_attribution_reconciles_to_realised_active_return(market, model):
    """Factor contributions plus specific must equal the realised active return."""
    p = equal_weights(model)
    b = cap_weights(market)

    active_w = (p.reindex(model.assets).fillna(0.0) - b.reindex(model.assets)).to_numpy()
    realised = float((market.returns[model.assets].to_numpy() @ active_w).sum())

    attributed = attribute_returns(p, model, benchmark_weights=b)
    assert attributed.loc["total", "return_contribution"] == pytest.approx(
        realised, rel=1e-8
    )


def test_attribution_requires_fitted_histories(model):
    bare = FactorModel(
        exposures=model.exposures,
        factor_cov=model.factor_cov,
        specific_var=model.specific_var,
    )
    with pytest.raises(ValueError, match="return histories"):
        attribute_returns(equal_weights(model), bare)


def test_unknown_asset_in_weights_raises(model):
    w = equal_weights(model)
    w["NOT_A_STOCK"] = 0.1
    with pytest.raises(ValueError, match="assets not in the model"):
        risk_decomposition(w, model)
