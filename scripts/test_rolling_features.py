"""
test_rolling_features.py — Validate rolling_features.py against synthetic
multi-week data with hand-computable answers. No network needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from rolling_features import build_asof_player_features, PLAYER_COUNT_COLS, TEAM_COUNT_COLS


def make_synthetic_player_week():
    """
    One team (BUF), one RB (RB1), 3 weeks. Also a teammate (RB2) each week so
    team totals != player totals (otherwise shares are trivially 1.0 always).
    Hand-known values:
      RB1 rush_opp: wk1=10, wk2=14, wk3=8
      RB2 rush_opp: wk1=5,  wk2=6,  wk3=10
      team_rush_opp = RB1+RB2 each week: wk1=15, wk2=20, wk3=18
    """
    rows = []

    def row(week, player, rush_opp, team_rush_opp):
        rows.append({
            "season": 2024, "week": week, "posteam": "BUF", "player_id": player,
            "rush_opp": rush_opp, "rush_td": 0, "rush_inside5": 0, "rush_inside10": 0,
            "rush_rz": 0, "rush_field": rush_opp, "pass_opp": 0, "pass_td": 0,
            "pass_inside5": 0, "pass_inside10": 0, "pass_rz": 0, "pass_field": 0,
            "expected_tds": 0.0, "actual_tds": 0, "scored_td": 0,
            "team_rush_opp": team_rush_opp, "team_pass_opp": 1, "team_rush_inside5": 1,
            "team_rush_inside10": 1, "team_rush_rz": 1, "team_pass_rz": 1,
            "team_pass_inside10": 1,
        })

    row(1, "RB1", 10, 15); row(1, "RB2", 5, 15)
    row(2, "RB1", 14, 20); row(2, "RB2", 6, 20)
    row(3, "RB1", 8, 18);  row(3, "RB2", 10, 18)

    return pd.DataFrame(rows)


def approx(a, b, tol=1e-6):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) < tol


def main():
    pw = make_synthetic_player_week()
    asof = build_asof_player_features(pw, windows=(4,))

    rb1 = asof[asof["player_id"] == "RB1"].set_index("week")

    checks = []

    # Week 1: no prior games -> everything trailing should be NaN
    checks.append(("RB1 wk1 asof_cum_rush_opp is NaN", pd.isna(rb1.loc[1, "asof_cum_rush_opp"]), True))
    checks.append(("RB1 wk1 asof_roll4_rush_opp is NaN", pd.isna(rb1.loc[1, "asof_roll4_rush_opp"]), True))
    checks.append(("RB1 wk1 games_played_prior", rb1.loc[1, "asof_player_games_prior"], 0))

    # Week 2: trailing = week 1 only (10 carries out of 15 team carries)
    checks.append(("RB1 wk2 asof_cum_rush_opp", rb1.loc[2, "asof_cum_rush_opp"], 10))
    checks.append(("RB1 wk2 asof_roll4_rush_opp", rb1.loc[2, "asof_roll4_rush_opp"], 10))
    checks.append(("RB1 wk2 asof_cum_carry_share", rb1.loc[2, "asof_cum_carry_share"], 10 / 15))
    checks.append(("RB1 wk2 games_played_prior", rb1.loc[2, "asof_player_games_prior"], 1))

    # Week 3: trailing = weeks 1+2 (10+14=24 carries out of 15+20=35 team carries)
    checks.append(("RB1 wk3 asof_cum_rush_opp", rb1.loc[3, "asof_cum_rush_opp"], 24))
    checks.append(("RB1 wk3 asof_roll4_rush_opp", rb1.loc[3, "asof_roll4_rush_opp"], 24))
    checks.append(("RB1 wk3 asof_cum_carry_share", rb1.loc[3, "asof_cum_carry_share"], 24 / 35))
    checks.append(("RB1 wk3 games_played_prior", rb1.loc[3, "asof_player_games_prior"], 2))

    # Critical leakage check: week 3's trailing carries must NOT include week 3's
    # own 8 carries. If it did, asof_cum_rush_opp would be 32, not 24.
    checks.append(("RB1 wk3 trailing EXCLUDES own week's carries",
                    rb1.loc[3, "asof_cum_rush_opp"] != 10 + 14 + 8, True))

    print(f"{'check':45s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 75)
    all_ok = True
    for name, got, want in checks:
        got_f = float(got) if not isinstance(got, bool) else float(got)
        want_f = float(want) if not isinstance(want, bool) else float(want)
        ok = approx(got_f, want_f)
        all_ok &= ok
        print(f"{name:45s} {got_f:10.5f} {want_f:10.5f}  {'PASS' if ok else 'FAIL'}")

    print("-" * 75)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
