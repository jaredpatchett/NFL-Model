"""
test_log_regression_ratio.py -- Confirms regression_ratio flows correctly
from the REAL blueprint_qualification.qualify() (not a stand-in) into the
exact log-line dict construction copied from generate_player_predictions.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import blueprint_qualification as bq


def build_log_line(p: dict) -> dict:
    """Copied exactly from the real edit in generate_player_predictions.py's
    log-write loop (the fields relevant to this change)."""
    q = p["qualification"]
    return {
        "qualifies": q["qualifies"], "blueprint_tier": q["tier"],
        "regression_ratio": q.get("regression_ratio"),
    }


def main():
    checks = []

    row = {
        "asof_roll4_snap_share": 0.80,
        "asof_roll4_rz_target_share": 0.15,
        "asof_roll4_inside5_carry_share": 0.0,
        "asof_roll4_actual_tds": 3.6,
        "asof_roll4_expected_tds": 2.0,
    }
    market = {"edge": 0.08, "ev": 0.15, "anytime_td_price": 180}
    qual = bq.qualify(row, model_prob=0.40, market=market)

    fake_player = {"qualification": qual}
    log_line = build_log_line(fake_player)

    checks.append(("real qualify() call computes the expected 1.8x ratio", qual["regression_ratio"], 1.8))
    checks.append(("log line carries the real regression_ratio through, not a stand-in value",
                    log_line["regression_ratio"], 1.8))
    checks.append(("log line's qualifies/tier match the real qualify() output",
                    (log_line["qualifies"], log_line["blueprint_tier"]), (qual["qualifies"], qual["tier"])))

    row_no_data = {
        "asof_roll4_snap_share": 0.80,
        "asof_roll4_rz_target_share": 0.15,
        "asof_roll4_inside5_carry_share": 0.0,
        "asof_roll4_actual_tds": None,
        "asof_roll4_expected_tds": None,
    }
    qual_no_data = bq.qualify(row_no_data, model_prob=0.40, market=market)
    log_line_no_data = build_log_line({"qualification": qual_no_data})
    checks.append(("player with no trailing TD data logs regression_ratio as None, not a crash",
                    log_line_no_data["regression_ratio"], None))

    import json
    try:
        json.dumps(log_line)
        json.dumps(log_line_no_data)
        serializes_ok = True
    except Exception:
        serializes_ok = False
    checks.append(("both log lines are valid JSON (what actually gets written to the log file)",
                    serializes_ok, True))

    print(f"{'check':78s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 105)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:78s} {str(got):>10s} {str(want):>10s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 105)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
