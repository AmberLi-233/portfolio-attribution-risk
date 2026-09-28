"""Barra-style cross-sectional factor risk model."""

from riskmodel.decompose import (
    active_exposure,
    attribute_returns,
    factor_risk_contributions,
    marginal_contribution_to_risk,
    portfolio_exposure,
    risk_decomposition,
)
from riskmodel.exposures import build_exposures, standardise
from riskmodel.model import FactorModel, ewma_covariance, ewma_variance, fit_factor_returns
from riskmodel.simulate import simulate_market

__all__ = [
    "FactorModel",
    "fit_factor_returns",
    "ewma_covariance",
    "ewma_variance",
    "build_exposures",
    "standardise",
    "simulate_market",
    "portfolio_exposure",
    "active_exposure",
    "risk_decomposition",
    "factor_risk_contributions",
    "marginal_contribution_to_risk",
    "attribute_returns",
]
