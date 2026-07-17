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
games don't blow up — the regularization pulls thin-data teams toward
league-average 0 rather than an unstable extreme rating). Week 1 of a season
has zero prior games, so it falls back to a regressed carry-over of the
team's final rating from the previous season (heavy shrinkage) — this is the
cold-start bridge Layer 2 didn't have.

Public API:
    build_power_ratings(schedule) -> DataFrame keyed by (season, week, team)
        with as-of-that-week rating + home-field-advantage constant, safe to
        join onto a modeling frame as of that week's game.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

RIDGE_ALPHA = 25.0          # shrinkage toward 0 (league average); higher = more conservative
CARRYOVER_SHRINK = 0.4      # how much of last season's final rating carries into week 1


def _solve_srs(games: pd.DataFrame, teams: list[str], alpha: float = RIDGE_ALPHA) -> dict[str, float]:
    """
    Solve ridge regression: margin_home = rating_home - rating_away + HFA.
    `games` needs columns home_team, away_team, home_score, away_score.
    Returns {team: rating}, plus a special key "__HFA__" for the home-field constant.
    """
    n_teams = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    n_games = len(games)
    if n_games == 0:
        return {t: 0.0 for t in teams} | {"__HFA__": 0.0}

    # Design matrix: one row per game, one column per team + one HFA column.
    X = np.zeros((n_games, n_teams + 1))
    y = np.zeros(n_games)
    for i, g in enumerate(games.itertuples()):
        X[i, idx[g.home_team]] = 1.0
        X[i, idx[g.away_team]] = -1.0
        X[i, -1] = 1.0  # HFA column
        y[i] = g.home_score - g.away_score

    # Ridge normal equations. Don't regularize the HFA term itself.
    reg = np.eye(n_teams + 1) * alpha
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)

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
    """
    sched = schedule.sort_values(["season", "week"]).reset_index(drop=True)
    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    seasons = sorted(sched["season"].unique())

    rows = []
    prior_season_final: dict[str, float] = {t: 0.0 for t in teams}

    for season in seasons:
        season_games = sched[sched["season"] == season]
        weeks = sorted(season_games["week"].unique())
        # Week-1 cold start: shrink last season's final ratings toward 0.
        current_ratings = {t: CARRYOVER_SHRINK * prior_season_final.get(t, 0.0) for t in teams}
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
                solved = _solve_srs(prior_games, teams)
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
        # become next season's cold-start carryover.
        played_games = season_games.dropna(subset=["home_score", "away_score"])
        full_solved = _solve_srs(played_games, teams)
        prior_season_final = {t: full_solved[t] for t in teams}

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Run test_power_ratings.py to validate this module against synthetic data.")
