"""
team_snapshot.py — Trailing-window team radar stats for the dashboard's
Matchup Detail view (last 6 / 12 / 18 games).

SEPARATE from the production prediction pipeline: power_ratings.py's
build_power_ratings() (season-cumulative, "as of this week") and
game_features.py's trailing 4/8-game windows are what the Ridge models
actually train on and are already validated -- this module doesn't touch
either. It exists purely to feed the dashboard's radar-chart visualization
with a genuinely different question ("how has this team played over their
last N games, opponent-adjusted") than what the prediction model asks
("what's the best current estimate of this team's true strength as of this
exact week"). Changing this file can never change a prediction.

It DOES reuse power_ratings.py's `_solve_srs` -- the same tested,
ridge-regularized opponent-adjustment math -- just called on a sliding
window of each team's most recent games instead of "every game so far this
season". That's a deliberate choice: re-deriving the same kind of rating
with different (untested) machinery would be a real risk for no benefit.

FIVE REAL, DIRECTLY-COMPUTED AXES. The reference image this was modeled on
showed six (including one labeled "PULL"); nothing in this codebase computes
anything that honestly maps to that, so it's left out rather than guessed at:

  SCORING  - trailing avg points scored/game
  DEFENSE  - trailing avg points allowed/game (lower is better; the
             percentile rank is inverted so a stingy defense still shows a
             HIGH percentile, even though the raw number is small)
  NET      - opponent-adjusted SRS rating, solved on the UNION of every
             team's last-N games in this window (SRS needs many teams'
             games solved together, not one team in isolation)
  FORM     - trend within the window: SRS rating solved on just the more
             recent half of each team's games, minus SRS solved on just the
             older half. Positive = playing better lately than earlier in
             the same window.
  CEILING  - single-game highest point total within the window

Each axis also gets a 0-100 percentile rank against every other team with
data in that same window, and a single 1-99 "strength" summary (the NET
percentile, rescaled) mirroring the reference's single-number badge.

A team with fewer than N games available (early season, or window reaching
past available history) gets whatever it actually has -- "games" reports
the real count used rather than silently padding to N.

Public API:
    build_team_snapshots(schedule, as_of_season, as_of_week, windows=(6,12,18))
        -> {window_n: {team: {...} | None}}
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from power_ratings import _solve_srs


def _played_games_before(schedule: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """All played games strictly before (season, week), across all seasons
    present in `schedule` -- so a trailing window can reach back into last
    season near the start of a new one. Same cold-start philosophy as
    power_ratings.py: real games only, never a game missing a final score."""
    played = schedule.dropna(subset=["home_score", "away_score"])
    before = played[
        (played["season"] < season) | ((played["season"] == season) & (played["week"] < week))
    ]
    return before.sort_values(["season", "week"]).reset_index(drop=True)


def _team_game_log(games: pd.DataFrame, team: str) -> pd.DataFrame:
    """One row per game `team` played (home or away), with team_score/
    opp_score from that team's own perspective, oldest -> newest."""
    home = games[games["home_team"] == team].rename(
        columns={"home_score": "team_score", "away_score": "opp_score"})
    away = games[games["away_team"] == team].rename(
        columns={"away_score": "team_score", "home_score": "opp_score"})
    both = pd.concat([home, away], ignore_index=True)
    return both.sort_values(["season", "week"]).reset_index(drop=True)


def _percentile_rank(values: dict[str, float], invert: bool = False) -> dict[str, float]:
    """0-100 percentile rank of each team's value among the teams given.
    invert=True: a LOWER raw value earns a HIGHER percentile (used for
    defense, where fewer points allowed is better)."""
    teams = list(values.keys())
    if len(teams) <= 1:
        return {t: 50.0 for t in teams}
    arr = np.array([values[t] for t in teams], dtype=float)
    if invert:
        arr = -arr
    order = arr.argsort().argsort()  # 0 = lowest raw value in `arr`
    pct = order / (len(teams) - 1) * 100
    return {t: float(round(p, 1)) for t, p in zip(teams, pct)}


def _snapshot_for_window(teams: list[str], prior_games: pd.DataFrame, n: int) -> dict:
    logs = {}
    for team in teams:
        log = _team_game_log(prior_games, team).tail(n)
        logs[team] = log if len(log) > 0 else None

    # NET: SRS needs many teams' games solved together, not one team's games
    # in isolation -- solve on the UNION of every team's last-N games
    # actually played (a shared game between two in-scope teams counts once).
    def game_set(logs_dict):
        ids = set()
        for log in logs_dict.values():
            if log is not None:
                ids.update(log["game_id"])
        return ids

    net_games = prior_games[prior_games["game_id"].isin(game_set(logs))]
    net_solved = _solve_srs(net_games, teams) if len(net_games) else {t: 0.0 for t in teams} | {"__HFA__": 0.0}

    # FORM: same SRS solve, independently on each team's more-recent half vs
    # older half of their own window.
    recent_logs, older_logs = {}, {}
    for team, log in logs.items():
        if log is None:
            recent_logs[team] = None
            older_logs[team] = None
            continue
        half = max(1, len(log) // 2)
        older_logs[team] = log.head(len(log) - half)
        recent_logs[team] = log.tail(half)

    older_games = prior_games[prior_games["game_id"].isin(game_set(older_logs))]
    recent_games = prior_games[prior_games["game_id"].isin(game_set(recent_logs))]
    older_solved = _solve_srs(older_games, teams) if len(older_games) else {t: 0.0 for t in teams} | {"__HFA__": 0.0}
    recent_solved = _solve_srs(recent_games, teams) if len(recent_games) else {t: 0.0 for t in teams} | {"__HFA__": 0.0}

    out = {}
    for team in teams:
        log = logs[team]
        if log is None:
            out[team] = None
            continue
        out[team] = {
            "games": int(len(log)),
            "scoring": round(float(log["team_score"].mean()), 1),
            "defense": round(float(log["opp_score"].mean()), 1),
            "net": round(net_solved[team], 2),
            "form": round(recent_solved[team] - older_solved[team], 2),
            "ceiling": round(float(log["team_score"].max()), 1),
        }

    have_data = {t: v for t, v in out.items() if v is not None}
    pctiles = {
        "scoring_pctile": _percentile_rank({t: v["scoring"] for t, v in have_data.items()}),
        "defense_pctile": _percentile_rank({t: v["defense"] for t, v in have_data.items()}, invert=True),
        "net_pctile": _percentile_rank({t: v["net"] for t, v in have_data.items()}),
        "form_pctile": _percentile_rank({t: v["form"] for t, v in have_data.items()}),
        "ceiling_pctile": _percentile_rank({t: v["ceiling"] for t, v in have_data.items()}),
    }
    for t, v in have_data.items():
        for key, ranks in pctiles.items():
            v[key] = ranks[t]
        # Single-number summary, mirroring the reference's "strength X/99"
        # badge -- driven by the NET percentile (the most complete single
        # signal we compute: opponent-adjusted, not just raw scoring/defense).
        v["strength"] = int(round(1 + v["net_pctile"] / 100 * 98))

    return out


def build_team_snapshots(
    schedule: pd.DataFrame, as_of_season: int, as_of_week: int,
    windows: tuple[int, ...] = (6, 12, 18),
) -> dict[int, dict]:
    """Returns {window_n: {team: {...} | None}} for each requested window,
    using only games strictly before (as_of_season, as_of_week)."""
    teams = sorted(set(schedule["home_team"]) | set(schedule["away_team"]))
    prior_games = _played_games_before(schedule, as_of_season, as_of_week)
    return {n: _snapshot_for_window(teams, prior_games, n) for n in windows}


if __name__ == "__main__":
    print("Run test_team_snapshot.py to validate this module against synthetic data.")
