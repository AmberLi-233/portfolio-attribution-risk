"""Worked example: three months of six-sector equity attribution.

Run from the repository root:
    python examples/run_demo.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution import multi_period_attribution, single_period_attribution  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_sector_data.csv"
BP = 10_000  # decimals -> basis points


def as_bp(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["allocation", "selection", "interaction", "total_effect"]
    out = frame.copy()
    out[cols] = (out[cols] * BP).round(1)
    return out


def main() -> None:
    df = pd.read_csv(DATA)
    pd.set_option("display.float_format", lambda v: f"{v:>8.1f}")

    first = df[df["period"] == "2024-01"].drop(columns=["period"])
    print("Single period (2024-01), effects in bp")
    print(as_bp(single_period_attribution(first)).to_string(index=False))

    r_p = (first["w_p"] * first["r_p"]).sum()
    r_b = (first["w_b"] * first["r_b"]).sum()
    print(f"\nportfolio {r_p * BP:.1f} bp, benchmark {r_b * BP:.1f} bp, "
          f"active {(r_p - r_b) * BP:.1f} bp")

    print("\n\nThree periods linked (Carino), effects in bp")
    linked = multi_period_attribution(df, link="carino")
    print(as_bp(linked).to_string(index=False))

    naive = multi_period_attribution(df, link="none")
    linked_total = linked.loc[linked["sector"] == "Total", "total_effect"].iloc[0]
    naive_total = naive.loc[naive["sector"] == "Total", "total_effect"].iloc[0]

    print(f"\nlinked total    {linked_total * BP:.2f} bp")
    print(f"naive total     {naive_total * BP:.2f} bp")
    print(f"residual        {(naive_total - linked_total) * BP:.2f} bp "
          "(the compounding error Carino removes)")


if __name__ == "__main__":
    main()
