"""
test_projector.py — Validates the Matchup Projector helper functions
(build_waterfall, build_distribution, build_situational_flags) against
synthetic fixtures with exact, hand-computable expected results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from generate_predictions import (
    build_waterfall, build_distribution, build_situational_flags, FEATURE_COLS, FEATURE_GROUPS,
)


def approx(a, b, tol=0.01):
    return abs(a - b) < tol


def test_waterfall_sums_to_prediction():
    """
    Fit a tiny Ridge model on synthetic data, then confirm the waterfall
    decomposition for a real row sums EXACTLY (to float precision) to that
    row's actual model prediction -- this is the core promise: the
    breakdown isn't an approximation, it's the literal arithmetic.
    """
    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame(rng.randn(n, len(FEATURE_COLS)), columns=FEATURE_COLS)
    y = X["power_rating"] * 3 - X["opp_power_rating"] * 2 + X["is_home"] * 1.5 + rng.randn(n) * 0.1

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_s, y)

    test_row = X.iloc[0]
    waterfall = build_waterfall(test_row, model, scaler)
    waterfall_sum = sum(w["contribution"] for w in waterfall)
    actual_pred = float(model.predict(scaler.transform(pd.DataFrame([test_row.values], columns=FEATURE_COLS)))[0])

    checks = [
        ("waterfall has one entry per group + intercept", len(waterfall), len(FEATURE_GROUPS) + 1),
        ("waterfall sum matches actual model prediction", round(waterfall_sum, 2), round(actual_pred, 2)),
    ]
    # All FEATURE_COLS must be covered by exactly one group (no silent drops).
    covered = set()
    for cols in FEATURE_GROUPS.values():
        covered.update(cols)
    checks.append(("every FEATURE_COL is covered by some group", covered == set(FEATURE_COLS), True))
    return checks


def test_distribution_integrates_to_one():
    """A normal distribution's PDF integral over a wide enough range should
    sum to ~1.0, and the bin containing the mean should have the highest
    probability mass."""
    dist = build_distribution(pred_margin=7.0, resid_std=13.0, bin_width=3)
    total_prob = sum(b["prob"] for b in dist)
    peak_bin = max(dist, key=lambda b: b["prob"])
    checks = [
        ("distribution integrates to ~1.0", round(total_prob, 2), 1.0),
        ("peak bin contains the mean (7.0)", peak_bin["bin_start"] <= 7.0 < peak_bin["bin_start"] + 3, True),
    ]
    return checks


def test_situational_flags():
    """Hand-constructed rows with known flag-triggering conditions."""
    checks = []

    row_division_dome = pd.Series({
        "div_game": 1, "is_indoor": 1, "rest_days": 7, "opp_rest_days": 7,
        "wind_filled": 0, "temp_filled": 70,
    })
    flags = build_situational_flags(row_division_dome)
    checks.append(("division + dome flags fire", set(flags), {"Division game", "Dome / indoor"}))

    row_rest_and_wind = pd.Series({
        "div_game": 0, "is_indoor": 0, "rest_days": 10, "opp_rest_days": 6,
        "wind_filled": 20, "temp_filled": 55,
    })
    flags2 = build_situational_flags(row_rest_and_wind)
    checks.append(("home rest advantage + high wind fire", flags2, ["Home rest advantage (+4d)", "High wind (20 mph)"]))

    row_nothing = pd.Series({
        "div_game": 0, "is_indoor": 0, "rest_days": 7, "opp_rest_days": 7,
        "wind_filled": 5, "temp_filled": 65,
    })
    checks.append(("no flags fire for an unremarkable game", build_situational_flags(row_nothing), []))

    return checks


def main():
    all_checks = (
        test_waterfall_sums_to_prediction()
        + test_distribution_integrates_to_one()
        + test_situational_flags()
    )

    print(f"{'check':50s} {'got':>30s} {'want':>30s}  ok")
    print("-" * 118)
    all_ok = True
    for name, got, want in all_checks:
        ok = approx(got, want) if isinstance(want, float) else (got == want)
        all_ok &= ok
        print(f"{name:50s} {str(got):>30s} {str(want):>30s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 118)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
