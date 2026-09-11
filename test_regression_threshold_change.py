"""
test_regression_threshold_change.py — Validates two things about the
blueprint_qualification.py change:
  1. The new REGRESSION_RATIO (2.0, was 1.5) actually changes real outcomes
     the way intended -- a player at 1.6x (would have failed the OLD
     threshold) now PASSES this specific check, while a player at 2.5x
     (still well over the NEW threshold) is still correctly flagged. This
     isn't a no-op change or an accidental removal of the check.
  2. regression_ratio is present and correct in ALL THREE return paths
     (market=None, hard_fail, and the success/tier path) -- this is the
     actual point of the change: track the real ratio for every candidate
     every week, regardless of outcome, so the threshold itself can be
     re-tested against real logged results later.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import blueprint_qualification as bq


def base_row(**overrides):
    """A row that clears every OTHER hard gate cleanly, so only the
    regression check is ever the variable under test."""
    row = {
        "asof_roll4_snap_share": 0.85,
        "asof_roll4_rz_target_share": 0.20,
        "asof_roll4_inside5_carry_share": 0.0,
        "asof_roll4_actual_tds": None,
        "asof_roll4_expected_tds": None,
    }
    row.update(overrides)
    return row


GOOD_MARKET = {"edge": 0.10, "ev": 0.20, "anytime_td_price": 200}


def main():
    checks = []

    # ---- The constant itself actually changed ----
    checks.append(("REGRESSION_RATIO is now 2.0, not the old 1.5", bq.REGRESSION_RATIO, 2.0))

    # ---- A player at 1.6x: FAILED the old 1.5x bar, must now PASS ----
    # actual=3.2, expected=2.0 -> ratio = 1.6
    row_16x = base_row(asof_roll4_actual_tds=3.2, asof_roll4_expected_tds=2.0)
    result_16x = bq.qualify(row_16x, model_prob=0.35, market=GOOD_MARKET)
    checks.append(("1.6x ratio: regression check no longer fires (was a hard-fail at 1.5x)",
                    any("regression risk" in r for r in result_16x["reason_codes"]), False))
    checks.append(("1.6x ratio: now genuinely qualifies (nothing else was wrong with this row)",
                    result_16x["qualifies"], True))
    checks.append(("1.6x ratio: regression_ratio reports the real computed value",
                    result_16x["regression_ratio"], 1.6))

    # ---- A player at 2.5x: still well over the NEW 2.0x bar, must still fail ----
    # actual=5.0, expected=2.0 -> ratio = 2.5
    row_25x = base_row(asof_roll4_actual_tds=5.0, asof_roll4_expected_tds=2.0)
    result_25x = bq.qualify(row_25x, model_prob=0.35, market=GOOD_MARKET)
    checks.append(("2.5x ratio: regression check STILL fires (this isn't a no-op change)",
                    any("regression risk" in r for r in result_25x["reason_codes"]), True))
    checks.append(("2.5x ratio: correctly does not qualify", result_25x["qualifies"], False))
    checks.append(("2.5x ratio: regression_ratio still reports the real value even though it failed",
                    result_25x["regression_ratio"], 2.5))

    # ---- Exactly at the new boundary (2.0x) should NOT fail (strictly greater-than) ----
    row_exactly_2x = base_row(asof_roll4_actual_tds=4.0, asof_roll4_expected_tds=2.0)
    result_2x = bq.qualify(row_exactly_2x, model_prob=0.35, market=GOOD_MARKET)
    checks.append(("exactly 2.0x (the new boundary itself) does not trip the check",
                    any("regression risk" in r for r in result_2x["reason_codes"]), False))

    # ---- regression_ratio present in ALL THREE return paths ----
    # Path 1: market is None
    row_market_none = base_row(asof_roll4_actual_tds=3.2, asof_roll4_expected_tds=2.0)
    result_no_market = bq.qualify(row_market_none, model_prob=0.35, market=None)
    checks.append(("regression_ratio present when market=None", result_no_market["regression_ratio"], 1.6))

    # Path 2: hard_fail via a DIFFERENT check (snap share), with regression data also present
    row_hard_fail = base_row(asof_roll4_snap_share=0.50, asof_roll4_actual_tds=3.2, asof_roll4_expected_tds=2.0)
    result_hard_fail = bq.qualify(row_hard_fail, model_prob=0.35, market=GOOD_MARKET)
    checks.append(("regression_ratio present on a hard_fail from an UNRELATED check (snap share)",
                    result_hard_fail["regression_ratio"], 1.6))
    checks.append(("that row correctly fails for snap share, not regression",
                    any("Snap share" in r for r in result_hard_fail["reason_codes"]), True))

    # Path 3: success/tier path (already covered by the 1.6x case above, result_16x)
    checks.append(("regression_ratio present on the success path", "regression_ratio" in result_16x, True))

    # ---- None handling: missing actual/expected data must not crash, and
    # must report regression_ratio as None (not 0, not a fabricated value) ----
    row_missing = base_row()  # actual_tds/expected_tds both None by default
    result_missing = bq.qualify(row_missing, model_prob=0.35, market=GOOD_MARKET)
    checks.append(("missing actual/expected TDs -> regression_ratio is None, not a crash or fake 0",
                    result_missing["regression_ratio"], None))
    checks.append(("missing data doesn't spuriously fail the regression check",
                    any("regression risk" in r for r in result_missing["reason_codes"]), False))

    # ---- expected_tds = 0 (real edge case: a player with zero projected
    # opportunity who somehow scored) must not divide by zero ----
    row_zero_expected = base_row(asof_roll4_actual_tds=2.0, asof_roll4_expected_tds=0.0)
    result_zero = bq.qualify(row_zero_expected, model_prob=0.35, market=GOOD_MARKET)
    checks.append(("expected_tds=0 doesn't crash with a ZeroDivisionError", result_zero["regression_ratio"], None))

    print(f"{'check':82s} {'got':>8s} {'want':>8s}  ok")
    print("-" * 108)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:82s} {str(got):>8s} {str(want):>8s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 108)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
