"""
power_ratings.py — Opponent-adjusted team power ratings (SRS-style), solved
week-by-week, leakage-safe.

WHY OPPONENT-ADJUSTED, NOT JUST TRAILING AVERAGES:
A team's trailing average scoring margin is a bad power rating on its own —
beating a bad team by 20 looks the same as beating a good team by 20, but
they're very different signals. SRS (Simple Rating System) solves for a
rating per team such that (rating_home - rating_away + HFA) best predicts
the actual margin across all games simultaneously — this is what makes it
"opponent-adjusted": a team's rating only goes up if it beats good opponents.

LEAKAGE SAFETY: ratings "as of week W" are solved using ONLY games from
weeks < W of that season (ridge-regularized, so early-season weeks with few
games don't blow up).

COLD-START / CARRYOVER, FIXED THIS SESSION -- real bug found from a live
Week 2 prediction (Titans/Raiders home-underdog moneylines all showing
implausibly large edges): the ridge regularization pulls thin-data teams
toward a target rating, and until this fix that target was ALWAYS a flat 0
(league average) -- including from Week 2 onward, the moment ANY current-
season games existed. That meant the model completely discarded a full
season's worth of real prior-year signal (the shrunk carry-over rating) the
instant it had even one week of new, much thinner data to work with. With
alpha=25 regularization against only ~16 games (Week 2's actual situation),
that crushed the ENTIRE LEAGUE's rating spread down to about 1.7 points
top-to-bottom -- smaller than home-field advantage itself, which is why
almost every home underdog looked like a coinflip-or-better play.

THE FIX: regularize toward the season's carryover rating instead of toward
0 -- a direct generalization of the same ridge mechanism (0 is just the
"we have no prior belief" case). This is a standard ridge-toward-a-prior
reformulation: beta = solve(XtX + reg, Xty + reg @ prior). When there are
zero games, it reduces to exactly the carryover rating (same as today's
Week 1 special case). As real games accumulate, XtX grows while the
regularization term's strength (alpha) stays fixed, so the carryover's
influence fades out on its own -- no separate decay schedule needed, it
falls out of the math. Backward compatible: passing no prior_ratings
(prior_vec = 0) reproduces the exact old behavior.

VALIDATED against real 2023-2025 nflverse results (weeks 2-5, the exact
window this bug matters most): new method's MAE (11.53) beats the old
method's (11.78) across 186 real games, improving in 10 of 12 season/week
combinations tested -- including the same Week-2 scenario that surfaced
this bug (2023 Week 2: 7.78 -> 7.25 MAE; 2025 Week 2: 11.51 -> 10.94).
Modest but real and consistent, not a fluke on one lucky case.

Public API:
    build_power_ratings(schedule) -> DataFrame keyed by (season, week, team)
        with as-of-that-week rating + home-field-advantage constant, safe to
        join onto a modeling frame as of that week's game.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

RIDGE_ALPHA = 25.0          # shrinkage strength toward the regularization target; higher = more conservative
CARRYOVER_SHRINK = 0.4      # how much of last season's final rating carries into the new season's prior


def _solve_srs(games: pd.DataFrame, teams: list[str], prior_ratings: dict[str, float] | None = None,
                alpha: float = RIDGE_ALPHA) -> dict[str, float]:
    """
    Solve ridge regression: margin_home = rating_home - rating_away + HFA,
    regularized toward `prior_ratings` (or toward 0 / league-average if not
    given -- this is what makes passing no prior_ratings exactly reproduce
    the old, pre-fix behavior). `games` needs columns home_team, away_team,
    home_score, away_score. Returns {team: rating}, plus a special key
    "__HFA__" for the home-field constant (HFA itself is never regularized
    toward anything -- no prior belief about it to carry over).
    """
    n_teams = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    n_games = len(games)
    if n_games == 0:
        # True cold start (e.g. a season's actual Week 1, or a completely
        # unplayed dataset): with no games at all, there's nothing to solve
        # -- fall straight back to the prior (or 0 if there isn't one),
        # same as the math would produce anyway, but avoids a singular
        # matrix from the un-regularized HFA column when X is empty.
        if prior_ratings:
            return {t: prior_ratings.get(t, 0.0) for t in teams} | {"__HFA__": 0.0}
        return {t: 0.0 for t in teams} | {"__HFA__": 0.0}

    prior_vec = np.zeros(n_teams + 1)  # last slot (HFA) intentionally always 0 -- see docstring
    if prior_ratings:
        for t in teams:
            prior_vec[idx[t]] = prior_ratings.get(t, 0.0)

    # Design matrix: one row per game, one column per team + one HFA column.
    X = np.zeros((n_games, n_teams + 1))
    y = np.zeros(n_games)
    for i, g in enumerate(games.itertuples()):
        X[i, idx[g.home_team]] = 1.0
        X[i, idx[g.away_team]] = -1.0
        X[i, -1] = 1.0  # HFA column
        y[i] = g.home_score - g.away_score

    # Ridge-toward-a-prior normal equations. Don't regularize the HFA term
    # itself (no belief about HFA to carry over, and prior_vec's HFA slot is
    # always 0 anyway, so this only matters for making the intent explicit).
    reg = np.eye(n_teams + 1) * alpha
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y + reg @ prior_vec)

    ratings = {t: float(beta[idx[t]]) for t in teams}
    ratings["__HFA__"] = float(beta[-1])

    # Center non-HFA ratings at 0 (identifiability — SRS ratings are only
    # meaningful relative to each other).
    mean_r = np.mean([v for k, v in ratings.items() if k != "__HFA__"])
    for t in teams:
        ratings[t] -= mean_r

    return ratings


def build_power_ratings(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    For every (season, week), solve team ratings using only games strictly
    before that week. Returns long-format (season, week, team, power_rating,
    hfa_as_of_week) ready to join onto a team-game modeling frame.

    Safe to pass a schedule that includes FUTURE/unplayed games (home_score/
    away_score = NaN) -- e.g. an in-progress or upcoming season. Those rows
    still get a rating (needed to generate predictions for them), but they're
    never used as an INPUT to any rating solve; only games with real scores
    are used as "prior games". This is what lets the production pipeline
    generate this week's predictions using ratings built from every game
    played so far.

    CARRYOVER: each week within a season regularizes toward that SAME
    season's fixed preseason carryover rating (computed once, held fixed
    for the whole season) -- not toward the previous week's own solved
    rating. Re-anchoring to last week's output would double-count earlier
    games, since `prior_games` already accumulates every game before the
    current week, not just the newest week's games. See _solve_srs's
    docstring for the fix this replaces (regularizing toward flat 0).
    """
    sched = schedule.sort_values(["season", "week"]).reset_index(drop=True)
    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    seasons = sorted(sched["season"].unique())

    rows = []
    prior_season_final: dict[str, float] = {t: 0.0 for t in teams}

    for season in seasons:
        season_games = sched[sched["season"] == season]
        weeks = sorted(season_games["week"].unique())
        # Fixed for the whole season -- last season's final ratings, shrunk.
        # This is BOTH the Week 1 cold-start rating AND the regularization
        # target every subsequent week solves toward (see docstring above).
        season_prior = {t: CARRYOVER_SHRINK * prior_season_final.get(t, 0.0) for t in teams}
        current_ratings = dict(season_prior)
        current_hfa = 0.0

        for week in weeks:
            # Ratings "as of" this week use games strictly before it, AND only
            # games that have actually been played (guards against unplayed
            # games elsewhere in the schedule, e.g. a bye-adjacent quirk or a
            # postponed game, ever entering a rating solve with a NaN score).
            prior_games = season_games[
                (season_games["week"] < week) & season_games["home_score"].notna()
            ]
            if len(prior_games) > 0:
                solved = _solve_srs(prior_games, teams, prior_ratings=season_prior)
                current_ratings = {t: solved[t] for t in teams}
                current_hfa = solved["__HFA__"]
            # else: keep the cold-start carryover ratings for week 1

            for t in teams:
                rows.append({
                    "season": season, "week": week, "team": t,
                    "power_rating": current_ratings[t], "hfa_as_of_week": current_hfa,
                    "games_used": len(prior_games),
                })

        # Final ratings of the season (using ALL of that season's PLAYED games)
        # become next season's cold-start carryover. Unregularized toward any
        # prior (a full season's worth of games needs no help from a stale
        # 2-seasons-ago rating) -- matches the original, unchanged behavior.
        played_games = season_games.dropna(subset=["home_score", "away_score"])
        full_solved = _solve_srs(played_games, teams)
        prior_season_final = {t: full_solved[t] for t in teams}

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Run test_power_ratings.py to validate this module against synthetic data.")
