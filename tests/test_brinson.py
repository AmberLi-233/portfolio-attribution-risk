import numpy as np
import pandas as pd
import pytest

from attribution import (
    multi_period_attribution,
    single_period_attribution,
)


def simple_frame():
    return pd.DataFrame(
        {
            "sector": ["Tech", "Energy", "Financials", "Health"],
            "w_p": [0.40, 0.10, 0.30, 0.20],
            "w_b": [0.30, 0.20, 0.25, 0.25],
            "r_p": [0.050, -0.020, 0.030, 0.010],
            "r_b": [0.040, -0.030, 0.035, 0.015],
        }
    )


def active_return(df):
    return float((df["w_p"] * df["r_p"]).sum() - (df["w_b"] * df["r_b"]).sum())


def test_effects_sum_to_active_return():
    df = simple_frame()
    result = single_period_attribution(df)
    total = result.loc[result["sector"] == "Total", "total_effect"].iloc[0]
    assert total == pytest.approx(active_return(df), abs=1e-12)


def test_identical_portfolio_and_benchmark_gives_zero():
    df = simple_frame()
    df["w_p"] = df["w_b"]
    df["r_p"] = df["r_b"]
    result = single_period_attribution(df)
    assert result["total_effect"].abs().max() == pytest.approx(0.0, abs=1e-15)


def test_pure_allocation_no_selection():
    """Same returns as the benchmark everywhere: only allocation can be nonzero."""
    df = simple_frame()
    df["r_p"] = df["r_b"]
    result = single_period_attribution(df)
    assert result["selection"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert result["interaction"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert result.loc[result["sector"] == "Total", "allocation"].iloc[0] != 0


def test_pure_selection_no_allocation():
    """Benchmark weights everywhere: only selection can be nonzero."""
    df = simple_frame()
    df["w_p"] = df["w_b"]
    result = single_period_attribution(df)
    assert result["allocation"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert result["interaction"].abs().max() == pytest.approx(0.0, abs=1e-15)


def test_fachler_sign_convention():
    """Overweighting a sector that lags the total benchmark is penalised.

    Energy returns -3% while the benchmark as a whole is positive, so an
    overweight in Energy must produce a negative allocation effect. Under the
    older Brinson-Hood-Beebower form (no -R_B term) the same overweight would
    also score negative here, so the test uses a sector that is positive but
    still below the benchmark total: Health at +1.5%.
    """
    df = simple_frame()
    df.loc[df["sector"] == "Health", "w_p"] = 0.35
    df.loc[df["sector"] == "Tech", "w_p"] = 0.25
    result = single_period_attribution(df)
    health = result.loc[result["sector"] == "Health"].iloc[0]
    benchmark_total = float((df["w_b"] * df["r_b"]).sum())

    assert df.loc[df["sector"] == "Health", "r_b"].iloc[0] > 0
    assert df.loc[df["sector"] == "Health", "r_b"].iloc[0] < benchmark_total
    assert health["allocation"] < 0


def test_weights_must_sum_to_one():
    df = simple_frame()
    df.loc[0, "w_p"] = 0.60
    with pytest.raises(ValueError, match="sums to"):
        single_period_attribution(df)


def test_missing_column_raises():
    df = simple_frame().drop(columns=["r_b"])
    with pytest.raises(ValueError, match="missing required columns"):
        single_period_attribution(df)


def test_duplicate_sector_raises():
    df = pd.concat([simple_frame(), simple_frame().head(1)], ignore_index=True)
    df["w_p"] = df["w_p"] / df["w_p"].sum()
    df["w_b"] = df["w_b"] / df["w_b"].sum()
    with pytest.raises(ValueError, match="duplicate sectors"):
        single_period_attribution(df)


def multi_period_frame():
    frames = []
    rng = np.random.default_rng(0)
    base = simple_frame()
    for period in ["2024-01", "2024-02", "2024-03"]:
        f = base.copy()
        f["period"] = period
        f["r_p"] = base["r_p"] + rng.normal(0, 0.01, len(base))
        f["r_b"] = base["r_b"] + rng.normal(0, 0.01, len(base))
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def compounded_active(df):
    rp, rb = [], []
    for p in dict.fromkeys(df["period"]):
        sub = df[df["period"] == p]
        rp.append(float((sub["w_p"] * sub["r_p"]).sum()))
        rb.append(float((sub["w_b"] * sub["r_b"]).sum()))
    total_p = float(np.prod([1 + x for x in rp]) - 1)
    total_b = float(np.prod([1 + x for x in rb]) - 1)
    return total_p - total_b


def test_carino_effects_sum_to_compounded_active_return():
    df = multi_period_frame()
    result = multi_period_attribution(df, link="carino")
    total = result.loc[result["sector"] == "Total", "total_effect"].iloc[0]
    assert total == pytest.approx(compounded_active(df), abs=1e-10)


def test_naive_summation_leaves_a_residual():
    """The whole point of Carino: naive summation does not reconcile."""
    df = multi_period_frame()
    naive = multi_period_attribution(df, link="none")
    total = naive.loc[naive["sector"] == "Total", "total_effect"].iloc[0]
    assert abs(total - compounded_active(df)) > 1e-8


def test_single_period_multi_period_agree():
    """With one period, Carino linking must reduce to the single-period result."""
    df = simple_frame()
    df["period"] = "2024-01"
    single = single_period_attribution(simple_frame())
    linked = multi_period_attribution(df, link="carino")
    for col in ["allocation", "selection", "interaction"]:
        assert linked[col].to_numpy() == pytest.approx(
            single[col].to_numpy(), abs=1e-12
        )
