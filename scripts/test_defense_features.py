"""
test_defense_features.py — Validates defense_features.py: raw defensive
count extraction, leakage-safe trailing (reusing rolling_features.py's
tested engine), and league-average normalization, against a small synthetic
schedule + pbp fixture with hand-computable expected results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from defense_features import _raw_defense_counts, _team_week_long, build_matchup_ratings
from unittest.mock import patch


def make_synthetic_pbp():
    """
    2 defenses (A, B), 1 week. Hand-known:
      Team A defense faces: 4 rush plays (1 TD), 4 pass plays (2 TD)
      Team B defense faces: 4 rush plays (0 TD), 4 pass plays (0 TD)
    """
    rows = []

    def play(defteam, play_type, yl, rush_td=0, pass_td=0):
        rows.append({
            "season": 2024, "week": 1, "defteam": defteam, "posteam": "OPP",
            "play_type": play_type, "yardline_100": yl,
            "rush_touchdown": rush_td, "pass_touchdown": pass_td,
            "touchdown": max(rush_td, pass_td),
        })

    for i in range(4):
        play("A", "run", 50, rush_td=1 if i == 0 else 0)
    for i in range(4):
        play("A", "pass", 50, pass_td=1 if i < 2 else 0)
    for i in range(4):
        play("B", "run", 50)
    for i in range(4):
        play("B", "pass", 50)

    return pd.DataFrame(rows)


def make_synthetic_schedule():
    return pd.DataFrame([
        {"season": 2024, "week": 1, "home_team": "A", "away_team": "B", "home_score": 20, "away_score": 10},
        {"season": 2024, "week": 2, "home_team": "A", "away_team": "B", "home_score": None, "away_score": None},
    ])


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def main():
    checks = []

    # ---- Raw counts ----
    pbp = make_synthetic_pbp()
    raw = _raw_defense_counts(pbp)
    a_row = raw[raw["defteam"] == "A"].iloc[0]
    b_row = raw[raw["defteam"] == "B"].iloc[0]

    checks.append(("Team A rush_opp_faced", a_row["rush_opp_faced"], 4))
    checks.append(("Team A rush_td_allowed", a_row["rush_td_allowed"], 1))
    checks.append(("Team A pass_opp_faced", a_row["pass_opp_faced"], 4))
    checks.append(("Team A pass_td_allowed", a_row["pass_td_allowed"], 2))
    checks.append(("Team B rush_td_allowed (stingy defense)", b_row["rush_td_allowed"], 0))
    checks.append(("Team B pass_td_allowed (stingy defense)", b_row["pass_td_allowed"], 0))

    # ---- Team-week long format includes future/unplayed weeks ----
    sched = make_synthetic_schedule()
    long = _team_week_long(sched)
    checks.append(("team_week_long includes week 2 (unplayed)",
                    len(long[long["week"] == 2]), 2))

    # ---- Full pipeline: leakage safety + league normalization ----
    with patch("defense_features.load_schedules", return_value=sched), \
         patch("defense_features.load_pbp", return_value=pbp):
        result = build_matchup_ratings([2024])

    wk2_a = result[(result["week"] == 2) & (result["defteam"] == "A")].iloc[0]
    wk2_b = result[(result["week"] == 2) & (result["defteam"] == "B")].iloc[0]

    # Week 2's trailing rate must come from week 1's REAL data (leakage check:
    # week 2 itself has no plays, so if trailing were reading week 2's own
    # stats it would be NaN/0, not week 1's actual 1/4 and 2/4 rates).
    checks.append(("wk2 Team A trailing rush TD rate = wk1's real 1/4", wk2_a["asof_roll4_rush_td_rate_allowed"], 0.25))
    checks.append(("wk2 Team A trailing pass TD rate = wk1's real 2/4", wk2_a["asof_roll4_pass_td_rate_allowed"], 0.5))
    checks.append(("wk2 Team B trailing rush TD rate = wk1's real 0/4", wk2_b["asof_roll4_rush_td_rate_allowed"], 0.0))

    # League average (2 teams) pass TD rate = (0.5 + 0.0) / 2 = 0.25.
    # Team A's matchup rating = 0.5 / 0.25 = 2.0 (twice as generous as average).
    checks.append(("Team A pass matchup rating = 2x league avg", round(wk2_a["matchup_rating_roll4_pass"], 4), 2.0))
    checks.append(("Team B pass matchup rating = 0x league avg (stingy)", round(wk2_b["matchup_rating_roll4_pass"], 4), 0.0))

    # Week 1 itself: no prior games, trailing should be NaN.
    wk1_a = result[(result["week"] == 1) & (result["defteam"] == "A")].iloc[0]
    checks.append(("wk1 Team A trailing rate is NaN (no prior games)",
                    pd.isna(wk1_a["asof_roll4_rush_td_rate_allowed"]), True))

    print(f"{'check':60s} {'got':>12s} {'want':>12s}  ok")
    print("-" * 90)
    all_ok = True
    for name, got, want in checks:
        got_f = float(got) if not isinstance(got, bool) else float(got)
        want_f = float(want) if not isinstance(want, bool) else float(want)
        ok = approx(got_f, want_f)
        all_ok &= ok
        print(f"{name:60s} {got_f:12.4f} {want_f:12.4f}  {'PASS' if ok else 'FAIL'}")
    print("-" * 90)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
