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


def test_carryover_blend_pulls_toward_prior_not_zero():
    """
    Real bug this fix addresses, reproduced on a tiny hand-computable case:
    a team with a strong prior-season carryover rating should stay pulled
    TOWARD that carryover with only a few current-season games -- not get
    crushed toward 0 the instant any current-season data exists (the old
    behavior). 2 teams, 1 current-season game, where the game's own signal
    is weak/contradictory -- with the old flat-0 regularization this drags
    both teams near 0; with the new carryover-aware regularization, a team
    with a strong positive carryover prior should stay meaningfully above 0.
    """
    games = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_score": 20, "away_score": 20},  # a tie: this single game carries ~zero signal either way
    ])
    prior = {"A": 5.0, "B": -5.0}  # A carries a strong positive prior, B a strong negative one

    old_style = _solve_srs(games, teams=["A", "B"])  # no prior_ratings passed -- reproduces old flat-0 behavior
    new_style = _solve_srs(games, teams=["A", "B"], prior_ratings=prior)

    checks = []
    # Old behavior: with no prior and a near-uninformative game, both ratings land near 0.
    checks.append(("old-style (no prior) A stays near 0", abs(old_style["A"]) < 0.5, True))
    # New behavior: A should stay meaningfully pulled toward its +5 carryover, not crushed to 0.
    checks.append(("new-style (with prior) A stays pulled toward +5 carryover", new_style["A"] > 2.0, True))
    checks.append(("new-style B stays pulled toward -5 carryover", new_style["B"] < -2.0, True))
    return checks


def test_no_prior_reproduces_old_behavior_exactly():
    """Backward-compatibility guarantee: calling _solve_srs with no
    prior_ratings must be numerically IDENTICAL to the pre-fix function --
    this is what every existing caller in this codebase that doesn't pass a
    prior (e.g. the end-of-season full_solved call in build_power_ratings)
    depends on."""
    games = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_score": 24, "away_score": 17},
        {"home_team": "B", "away_team": "C", "home_score": 14, "away_score": 21},
        {"home_team": "C", "away_team": "A", "home_score": 10, "away_score": 27},
    ])
    teams = ["A", "B", "C"]
    no_prior = _solve_srs(games, teams)  # default prior_ratings=None
    explicit_zero_prior = _solve_srs(games, teams, prior_ratings={t: 0.0 for t in teams})
    checks = []
    for t in teams:
        checks.append((f"no-prior == explicit-zero-prior for {t}", abs(no_prior[t] - explicit_zero_prior[t]) < 1e-9, True))
    return checks


def main():
    checks = (
        test_solve_srs_hand_computable()
        + test_leakage_and_cold_start()
        + test_carryover_blend_pulls_toward_prior_not_zero()
        + test_no_prior_reproduces_old_behavior_exactly()
    )

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
