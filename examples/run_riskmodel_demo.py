"""Layer 2 worked example: fit a factor risk model and decompose a portfolio.

Run from the repository root:
    python examples/run_riskmodel_demo.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskmodel import (  # noqa: E402
    FactorModel,
    active_exposure,
    factor_risk_contributions,
    marginal_contribution_to_risk,
    risk_decomposition,
    simulate_market,
)
from riskmodel.model import TRADING_DAYS  # noqa: E402


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:>10.4f}")

    market = simulate_market(n_assets=500, n_periods=750, seed=2024)
    model = FactorModel.fit(
        market.returns,
        market.exposures,
        regression_weights=market.market_cap**0.5,
    )
    print(f"universe {len(model.assets)} stocks, {len(model.factors)} factors, "
          f"{len(market.returns)} daily observations")

    print("\n\n1. Does the estimator recover the truth?")
    true_vol = np.sqrt(np.diag(market.true_factor_cov.to_numpy()) * TRADING_DAYS)
    est_vol = np.sqrt(np.diag(model.factor_cov.to_numpy()) * TRADING_DAYS)
    recovery = pd.DataFrame(
        {
            "true_vol": true_vol,
            "estimated_vol": est_vol,
            "error_pct": (est_vol / true_vol - 1) * 100,
        },
        index=model.factors,
    )
    print(recovery.round(3).to_string())

    benchmark = market.market_cap / market.market_cap.sum()

    print("\n\n2. Risk decomposition, equally weighted portfolios")
    rows = []
    for n in (10, 25, 50, 100, 250, 500):
        w = pd.Series(1.0 / n, index=model.assets[:n])
        d = risk_decomposition(w, model)
        rows.append({"n_holdings": n, **d})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("specific risk falls roughly as 1/sqrt(N); factor risk does not move")

    print("\n\n3. A momentum-tilted portfolio against the cap-weighted benchmark")
    top = model.exposures["momentum"].sort_values(ascending=False).index[:100]
    tilted = pd.Series(1.0 / len(top), index=top)

    xa = active_exposure(tilted, benchmark, model)
    print("\nactive exposures")
    print(xa.round(3).to_string())

    te = risk_decomposition(tilted, model, benchmark_weights=benchmark)
    print(f"\ntracking error      {te['total_vol'] * 100:.2f}%")
    print(f"  from factors      {te['factor_vol'] * 100:.2f}%")
    print(f"  from stock picks  {te['specific_vol'] * 100:.2f}%")
    print(f"  factor share      {te['factor_share'] * 100:.1f}%")

    print("\nwhere the tracking error comes from")
    contrib = factor_risk_contributions(tilted, model, benchmark)
    contrib = contrib.assign(
        pct=lambda d: d["pct_of_factor_variance"] * 100
    )[["exposure", "pct"]]
    print(contrib.head(5).round(2).to_string())

    print("\n\n4. Which position to trim first")
    mcr = marginal_contribution_to_risk(tilted, model, benchmark)
    print(mcr.head(5).round(4).to_string())
    print("highest contribution, not highest weight, is the one to cut")


if __name__ == "__main__":
    main()
