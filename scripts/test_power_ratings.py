"""
test_power_ratings.py — Validate power_ratings.py: the SRS solve itself
(hand-computable on a tiny fully-determined system) and leakage safety of
the week-by-week builder. No network needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from power_ratings import _solve_srs, build_power_ratings


def approx(a, b, tol=0.05):
    return abs(a - b) < tol


def test_solve_srs_hand_computable():
    """
    Two teams play twice (home and away swapped):
      Game 1: A home vs B away, margin (A-B) = +10
      Game 2: B home vs A away, margin (B-A) = +4
    System: rA - rB + HFA = 10 ; rB - rA + HFA = 4
    => HFA = 7, rA - rB = 3 => (centered) rA = 1.5, rB = -1.5, HFA = 7
    Use near-zero alpha so ridge shrinkage doesn't meaningfully perturb this.
    """
    games = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_score": 20, "away_score": 10},
        {"home_team": "B", "away_team": "A", "home_score": 17, "away_score": 13},
    ])
    ratings = _solve_srs(games, teams=["A", "B"], alpha=0.01)
    return [
        ("rating_A ~= 1.5", ratings["A"], 1.5),
        ("rating_B ~= -1.5", ratings["B"], -1.5),
        ("HFA ~= 7.0", ratings["__HFA__"], 7.0),
    ]


def test_leakage_and_cold_start():
    """
    3 teams, 2 seasons, 3 weeks each. Verify:
      - Week 1 of season 1 has zero games_used (true cold start, no history at all)
      - Week 3's rating does NOT reflect week 3's own result (leakage check)
    """
    rows = []

    def g(season, week, home, away, hs, as_):
        rows.append({"season": season, "week": week, "home_team": home,
                      "away_team": away, "home_score": hs, "away_score": as_})

    # Season 1
    g(2023, 1, "A", "B", 24, 10)
    g(2023, 1, "C", "A", 14, 14)  # bye-less round robin isn't needed; just need games
    g(2023, 2, "B", "C", 20, 17)
    g(2023, 2, "A", "C", 30, 10)
    g(2023, 3, "A", "B", 21, 20)
    g(2023, 3, "B", "C", 24, 24)

    sched = pd.DataFrame(rows)
    ratings = build_power_ratings(sched)

    checks = []
    wk1 = ratings[(ratings["season"] == 2023) & (ratings["week"] == 1)]
    checks.append(("season1 week1 games_used == 0 (true cold start)",
                    (wk1["games_used"] == 0).all(), True))

    # Leakage check: compute week-3 ratings two ways -- once via the normal
    # builder (which should use only weeks 1-2), and once by manually solving
    # with week 3's game INCLUDED. They must differ if leakage is absent.
    wk3_normal = ratings[(ratings["season"] == 2023) & (ratings["week"] == 3) & (ratings["team"] == "A")]
    rating_a_normal = wk3_normal["power_rating"].iloc[0]

    all_games_incl_wk3 = sched[(sched["season"] == 2023) & (sched["week"] <= 3)]
    solved_with_leakage = _solve_srs(all_games_incl_wk3, teams=["A", "B", "C"])
    rating_a_with_leakage = solved_with_leakage["A"]

    checks.append(("week3 rating EXCLUDES week3's own game (differs from leaky version)",
                    rating_a_normal != rating_a_with_leakage, True))

    return checks


def main():
    checks = test_solve_srs_hand_computable() + test_leakage_and_cold_start()

    print(f"{'check':60s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 90)
    all_ok = True
    for name, got, want in checks:
        got_f = float(got)
        want_f = float(want)
        ok = approx(got_f, want_f) if not isinstance(got, bool) else (got == want)
        all_ok &= ok
        print(f"{name:60s} {got_f:10.4f} {want_f:10.4f}  {'PASS' if ok else 'FAIL'}")

    print("-" * 90)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
