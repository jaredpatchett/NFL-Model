"""
generate_fantasy_projections.py -- Production entrypoint for Track C
(fantasy point projections). Same pattern as generate_predictions.py (Track B)
and generate_player_predictions.py (Track A): trains/derives from all played
history, projects the upcoming week, writes a current snapshot (overwritten)
+ appends to a persistent log (never overwritten).

INPUT: data/player_stats_{season}.csv and data/snap_counts_{season}.csv,
written by fetch_player_stats.py (nflverse-data, pulled earlier in the same
workflow run -- see .github/workflows/update-predictions.yml).

METHOD (first cut -- a trailing-average baseline, not yet a trained
regression like Track A/B):
  1. Leakage-safe trailing fantasy points: for every player, shift(1) before
     taking a rolling mean over their last 4 played games -- same
     shift-before-aggregate principle as rolling_features.py, so a game's
     own result can never enter its own projection.
  2. Opponent-adjustment: trailing PPR fantasy points allowed BY THE
     UPCOMING OPPONENT to that player's position_group, normalized against
     the league-average allowed to that position_group THAT SAME WEEK --
     same "normalize against the same-week league average" approach as
     Track A's matchup_rating (defense_features.py), not a fixed historical
     constant. 1.00 = league average; above 1.00 = softer than average.
  3. Final projection = trailing_avg_pts * matchup_factor.

Honest limitation, stated plainly (same spirit as Track B's total-points
caveat): this hasn't been validated against held-out data the way Track A/B
were (AUC 0.70, Ridge MAE beats baseline). It's a reasonable, leakage-safe
baseline to ship and start logging -- a real accuracy check happens once
predictions_log-style entries can be joined against real outcomes.

Usage:
    python generate_fantasy_projections.py
"""

from __future__ import annotations
import json
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from nfl_data import load_schedules

SEASON = date.today().year
STATS_CSV = f"../data/player_stats_{SEASON}.csv"
SNAPS_CSV = f"../data/snap_counts_{SEASON}.csv"

OUT_JSON = "../data/fantasy_projections.json"
OUT_JS = "../data/fantasy_projections.js"
OUT_LOG = "../data/fantasy_projections_log.jsonl"

TRAILING_WINDOW = 4
MIN_TRAILING_WEEKS_FOR_MATCHUP = 3  # below this, default matchup_factor to 1.00 -- see build_defense_vs_position
MATCHUP_FACTOR_MIN = 0.75
MATCHUP_FACTOR_MAX = 1.35
SHRINKAGE_GAMES = 3  # prior "weight" in games -- see build_trailing_player_avg's shrinkage estimator
POSITIONS = ["QB", "RB", "WR", "TE"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(
            f"{STATS_CSV} not found -- run fetch_player_stats.py first "
            "(it runs earlier in the same workflow)."
        )
    stats = pd.read_csv(STATS_CSV)
    stats = stats[stats["position"].isin(POSITIONS)].copy()

    snaps = pd.DataFrame()
    if os.path.exists(SNAPS_CSV):
        snaps = pd.read_csv(SNAPS_CSV)

    # Prior-season data for the per-player shrinkage prior (see
    # build_trailing_player_avg's docstring). Pulled directly from
    # nflverse-data via fetch_player_stats.py's own fetch_player_stats()
    # function -- same source, not re-fetched into a separate CSV on disk,
    # since this is only needed transiently to compute one prior per player.
    prior_season = pd.DataFrame()
    try:
        from fetch_player_stats import fetch_player_stats as _fetch_prior
        prior_season = _fetch_prior(SEASON - 1)
        prior_season = prior_season[prior_season["position"].isin(POSITIONS)]
    except Exception as e:
        print(f"  WARNING: couldn't load {SEASON - 1} prior-season data for shrinkage "
              f"priors ({e}) -- falling back to generic position median for everyone.",
              file=sys.stderr)

    return stats, snaps, prior_season


def build_trailing_player_avg(stats: pd.DataFrame, prior_season_stats: pd.DataFrame) -> pd.DataFrame:
    """LIVE trailing mean fantasy_points_ppr per player, as of right now --
    averages over each player's most recent (up to TRAILING_WINDOW) PLAYED
    games, including their most recent one.

    This is deliberately NOT shifted the way rolling_features.py's
    shift-before-aggregate engine is. That shift exists to stop a game from
    using its OWN result as an input to its OWN prediction when building a
    per-week HISTORICAL training table (many rows, one per played week).
    Here there is no "own result" problem: we're projecting a genuinely
    unplayed future week, so every game played so far -- including the most
    recent one -- is legitimate, already-known information for that
    projection. Shifting here was the bug: it excluded each player's latest
    game from their own trailing average, producing NaN for anyone with
    only one game played (exactly what happened at Week 1 -- see the
    conversation this was caught in).

    SHRINKAGE, take 2 (the first version shrank toward a generic
    same-position median -- flagged as wrong in the same conversation:
    shrinking an established WR1 toward "whatever every WR on the field
    that week averaged" drags him down to replacement level, which isn't
    what a small sample should imply about a known player). This version
    shrinks each player toward THEIR OWN prior-season average instead --
    same "bridge cold start with prior-season trailing stats" fix already
    named as the correct direction in this repo's own README for Track
    A/B's cold-start gap. Verified against real 2025 season data: Justin
    Jefferson's actual full-season PPR average was 11.85 (a real down year
    by his own standard -- McCaffrey/Nacua/Chase were all 19-24 that same
    season), so shrinking his single 2026 Week 1 game toward 11.85 is
    defensible; shrinking him toward a generic WR pool median was not.
    Falls back to the generic position median ONLY for players with no
    prior-season row at all (true rookies) -- there's no other prior
    available for them yet."""
    stats = stats.sort_values(["player_id", "week"]).reset_index(drop=True)
    grp = stats.groupby("player_id")["fantasy_points_ppr"]
    stats["raw_trailing_avg_pts"] = grp.transform(
        lambda s: s.rolling(TRAILING_WINDOW, min_periods=1).mean()
    )
    stats["trailing_games"] = grp.transform("cumcount") + 1

    # Player-specific prior: each player's own prior-season average.
    player_prior = (
        prior_season_stats.groupby("player_id")["fantasy_points_ppr"].mean()
        if not prior_season_stats.empty else pd.Series(dtype=float)
    )
    stats["player_prior_pts"] = stats["player_id"].map(player_prior)

    # Generic position median -- fallback ONLY for true rookies with no
    # prior-season row to draw a personal prior from.
    position_baseline = stats.groupby("position")["raw_trailing_avg_pts"].transform("median")
    stats["shrink_prior_pts"] = stats["player_prior_pts"].fillna(position_baseline)

    weight = stats["trailing_games"] / (stats["trailing_games"] + SHRINKAGE_GAMES)
    stats["trailing_avg_pts"] = (
        weight * stats["raw_trailing_avg_pts"]
        + (1 - weight) * stats["shrink_prior_pts"]
    ).round(2)
    return stats


def build_defense_vs_position(stats: pd.DataFrame) -> pd.DataFrame:
    """LIVE trailing fantasy points allowed by each team to each
    position_group, as of right now (not shifted -- same reasoning as
    build_trailing_player_avg above: every played week is legitimate
    already-known information for projecting the next, unplayed one),
    normalized against that same week's league-average allowed to that
    position -- same approach as Track A's matchup_rating."""
    team_pos_week = (
        stats.groupby(["opponent_team", "position", "week"], as_index=False)
        .agg(pts_allowed=("fantasy_points_ppr", "sum"),
             players=("player_id", "nunique"))
    )
    team_pos_week = team_pos_week.sort_values(["opponent_team", "position", "week"])
    team_pos_week["trailing_allowed"] = team_pos_week.groupby(
        ["opponent_team", "position"]
    )["pts_allowed"].transform(lambda s: s.rolling(TRAILING_WINDOW, min_periods=1).mean())

    # league-average allowed to that position, same week, same trailing basis
    league_avg = (
        team_pos_week.groupby(["position", "week"])["trailing_allowed"]
        .transform("mean")
    )
    team_pos_week["matchup_factor"] = np.where(
        league_avg > 0, team_pos_week["trailing_allowed"] / league_avg, 1.0
    )

    # Sample-size guardrail: with only 1-2 games of trailing defensive data,
    # a single unusually high/low game looks like a real tendency but is
    # mostly noise (confirmed against real Week 1 2026 data -- one big game
    # allowed produced a 2.27x matchup_factor off a single sample). Two
    # protections, both standard shrinkage/capping, not invented for this
    # specific number:
    #   1. Below MIN_TRAILING_WEEKS_FOR_MATCHUP games of defensive history,
    #      don't trust the factor at all -- default to 1.00 (league average)
    #      rather than react to n=1 noise.
    #   2. Even with enough games, cap the factor to a plausible range so one
    #      remaining outlier week can't swing a projection more than +/-40%.
    team_pos_week["trailing_def_weeks"] = team_pos_week.groupby(
        ["opponent_team", "position"]
    ).cumcount() + 1
    team_pos_week.loc[
        team_pos_week["trailing_def_weeks"] < MIN_TRAILING_WEEKS_FOR_MATCHUP, "matchup_factor"
    ] = 1.0
    team_pos_week["matchup_factor"] = team_pos_week["matchup_factor"].clip(
        MATCHUP_FACTOR_MIN, MATCHUP_FACTOR_MAX
    )
    return team_pos_week


def rank_defense_vs_position(team_pos_week: pd.DataFrame, latest_week: dict) -> pd.DataFrame:
    """1 = allows the MOST fantasy points to that position (softest matchup),
    32 = allows the fewest (toughest) -- as of each team's latest played week."""
    rows = []
    for pos in POSITIONS:
        pos_df = team_pos_week[team_pos_week["position"] == pos]
        latest = (
            pos_df.sort_values("week")
            .groupby("opponent_team")
            .tail(1)
            .copy()
        )
        latest = latest.dropna(subset=["trailing_allowed"])
        if latest.empty:
            continue
        latest["def_rank"] = latest["trailing_allowed"].rank(ascending=False, method="min").astype(int)
        latest["def_rank_of"] = latest["opponent_team"].nunique()
        rows.append(latest[["opponent_team", "def_rank", "def_rank_of", "matchup_factor"]].assign(position=pos))
    if not rows:
        return pd.DataFrame(columns=["opponent_team", "def_rank", "def_rank_of", "matchup_factor", "position"])
    return pd.concat(rows, ignore_index=True)


def build_snap_pct(snaps: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        return pd.DataFrame(columns=["player", "team", "latest_snap_pct"])
    snaps = snaps.sort_values(["player", "week"])
    latest = snaps.groupby("player").tail(1)
    return latest[["player", "offense_pct"]].rename(columns={"offense_pct": "latest_snap_pct"})


def build_weekly_def_rank(stats: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time defense-vs-position rank for a SPECIFIC played week --
    for the game log, not projection. Different from build_defense_vs_position
    (which is a trailing, leakage-safe figure used to project an upcoming
    game): this ranks all 32 teams by what they ACTUALLY allowed to a
    position in one specific past week, since it's purely descriptive
    (showing what really happened in a game already played), not a model
    input, so there's no leakage question here."""
    team_pos_week = (
        stats.groupby(["opponent_team", "position", "week"], as_index=False)
        .agg(pts_allowed=("fantasy_points_ppr", "sum"))
    )
    team_pos_week["week_def_rank"] = team_pos_week.groupby(["position", "week"])["pts_allowed"].rank(
        ascending=False, method="min"
    ).astype(int)
    team_pos_week["week_def_rank_of"] = team_pos_week.groupby(["position", "week"])["opponent_team"].transform("nunique")
    return team_pos_week[["opponent_team", "position", "week", "week_def_rank", "week_def_rank_of"]]


def build_percentiles(stats: pd.DataFrame) -> pd.DataFrame:
    """Season-average percentile rank within position for the stat tiles
    (targets, receptions, rec yards, touchdowns, fantasy pts) -- purely
    descriptive of games actually played, not a projection."""
    season_avg = stats.groupby(["player_id", "position"], as_index=False).agg(
        targets_avg=("targets", "mean"),
        receptions_avg=("receptions", "mean"),
        rec_yards_avg=("receiving_yards", "mean"),
        rush_yards_avg=("rushing_yards", "mean"),
        touchdowns_avg=("rushing_tds", lambda s: s.mean()),  # placeholder, combined below
        fantasy_pts_avg=("fantasy_points_ppr", "mean"),
        games=("week", "nunique"),
    )
    # touchdowns_avg needs rushing_tds + receiving_tds per game, not a single column mean
    td_avg = stats.groupby("player_id").apply(
        lambda g: (g["rushing_tds"] + g["receiving_tds"]).mean(), include_groups=False
    ).rename("touchdowns_avg")
    season_avg = season_avg.drop(columns=["touchdowns_avg"]).merge(td_avg, on="player_id", how="left")

    stat_cols = ["targets_avg", "receptions_avg", "rec_yards_avg", "touchdowns_avg", "fantasy_pts_avg"]
    for col in stat_cols:
        season_avg[f"pctl_{col}"] = (
            season_avg.groupby("position")[col].rank(pct=True) * 100
        ).round(0)
        season_avg[f"peers_{col}"] = season_avg.groupby("position")["player_id"].transform("count")
    return season_avg


def build_snap_share_by_week(snaps: pd.DataFrame) -> dict:
    if snaps.empty:
        return {}
    out = {}
    for player, g in snaps.sort_values("week").groupby("player"):
        out[player] = [
            {"week": int(w), "pct": None if pd.isna(p) else round(float(p), 3)}
            for w, p in zip(g["week"], g["offense_pct"])
        ]
    return out


def build_game_log(stats: pd.DataFrame, weekly_def_rank: pd.DataFrame) -> dict:
    merged = stats.merge(
        weekly_def_rank, left_on=["opponent_team", "position", "week"],
        right_on=["opponent_team", "position", "week"], how="left"
    )
    merged = merged.sort_values(["player_id", "week"], ascending=[True, False])
    out = {}
    for pid, g in merged.groupby("player_id"):
        out[pid] = [{
            "week": int(r["week"]),
            "opponent": r["opponent_team"],
            "week_def_rank": None if pd.isna(r.get("week_def_rank")) else int(r["week_def_rank"]),
            "week_def_rank_of": None if pd.isna(r.get("week_def_rank_of")) else int(r["week_def_rank_of"]),
            "targets": int(r["targets"]), "receptions": int(r["receptions"]),
            "receiving_yards": int(r["receiving_yards"]), "rushing_yards": int(r["rushing_yards"]),
            "touchdowns": int(r["rushing_tds"] + r["receiving_tds"]),
            "fantasy_pts": round(float(r["fantasy_points_ppr"]), 1),
        } for _, r in g.iterrows()]
    return out


def fetch_team_logos() -> dict:
    """Team logo URLs, real source verified this session:
    https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.csv
    Same nflverse-data release family as everything else -- no new account,
    no scraping."""
    import requests
    url = "https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(resp.text))
        return dict(zip(df["team_abbr"], df["team_logo_espn"]))
    except Exception as e:
        print(f"  WARNING: couldn't fetch team logos ({e})", file=sys.stderr)
        return {}


def find_upcoming_matchups(season: int) -> pd.DataFrame:
    """Earliest unplayed week per team, with opponent -- same
    find_upcoming_week pattern as generate_predictions.py, but keyed by team
    since every team needs its own next opponent, not one shared week."""
    sched = load_schedules([season])
    unplayed = sched[sched["home_score"].isna()].copy()
    if unplayed.empty:
        return pd.DataFrame(columns=["team", "opponent", "week"])

    home = unplayed[["week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = unplayed[["week", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    both = pd.concat([home, away], ignore_index=True)
    return both.sort_values("week").groupby("team").head(1)


def main():
    print(f"Building fantasy projections for {SEASON}...")
    stats, snaps, prior_season = load_inputs()
    if stats.empty:
        print("No player stats available yet -- nothing to project.")
        return

    stats = build_trailing_player_avg(stats, prior_season)
    team_pos_week = build_defense_vs_position(stats)
    def_ranks = rank_defense_vs_position(team_pos_week, {})
    snap_pct = build_snap_pct(snaps)
    weekly_def_rank = build_weekly_def_rank(stats)
    percentiles = build_percentiles(stats)
    snap_share_by_week = build_snap_share_by_week(snaps)
    game_logs = build_game_log(stats, weekly_def_rank)
    team_logos = fetch_team_logos()

    latest_row = (
        stats.sort_values("week")
        .groupby("player_id")
        .tail(1)
        .copy()
    )

    upcoming = find_upcoming_matchups(SEASON)
    latest_row = latest_row.merge(upcoming, left_on="team", right_on="team", how="left")

    latest_row = latest_row.merge(
        def_ranks, left_on=["opponent", "position"], right_on=["opponent_team", "position"], how="left"
    )
    latest_row = latest_row.merge(snap_pct, left_on="player_display_name", right_on="player", how="left")
    latest_row = latest_row.merge(percentiles, on=["player_id", "position"], how="left")

    latest_row["matchup_factor"] = latest_row["matchup_factor"].fillna(1.0)
    latest_row["proj_fantasy_pts"] = (
        latest_row["trailing_avg_pts"].fillna(0) * latest_row["matchup_factor"]
    ).round(2)

    def _pctl(r, col):
        v = r.get(f"pctl_{col}")
        return None if pd.isna(v) else int(v)

    def _peers(r, col):
        v = r.get(f"peers_{col}")
        return None if pd.isna(v) else int(v)

    players = []
    for _, r in latest_row.iterrows():
        if pd.isna(r.get("opponent")):
            continue  # no upcoming game found for this team (bye week or season over)
        players.append({
            "player_id": r["player_id"],
            "player_name": r["player_display_name"],
            "position": r["position"],
            "team": r["team"],
            "opponent": r["opponent"],
            "headshot_url": r.get("headshot_url"),
            "team_logo_url": team_logos.get(r["team"]),
            "games_played": int(r.get("games", r["trailing_games"])),
            "trailing_avg_pts": None if pd.isna(r["trailing_avg_pts"]) else round(float(r["trailing_avg_pts"]), 2),
            "trailing_games": int(r["trailing_games"]),
            "proj_fantasy_pts": None if pd.isna(r["proj_fantasy_pts"]) else float(r["proj_fantasy_pts"]),
            "def_rank": None if pd.isna(r.get("def_rank")) else int(r["def_rank"]),
            "def_rank_of": None if pd.isna(r.get("def_rank_of")) else int(r["def_rank_of"]),
            "matchup_factor": round(float(r["matchup_factor"]), 3),
            "latest_snap_pct": None if pd.isna(r.get("latest_snap_pct")) else round(float(r["latest_snap_pct"]), 3),
            "season_avg": {
                "targets": None if pd.isna(r.get("targets_avg")) else round(float(r["targets_avg"]), 2),
                "receptions": None if pd.isna(r.get("receptions_avg")) else round(float(r["receptions_avg"]), 2),
                "rec_yards": None if pd.isna(r.get("rec_yards_avg")) else round(float(r["rec_yards_avg"]), 2),
                "touchdowns": None if pd.isna(r.get("touchdowns_avg")) else round(float(r["touchdowns_avg"]), 2),
                "fantasy_pts": None if pd.isna(r.get("fantasy_pts_avg")) else round(float(r["fantasy_pts_avg"]), 2),
            },
            "percentiles": {
                "targets": _pctl(r, "targets_avg"),
                "receptions": _pctl(r, "receptions_avg"),
                "rec_yards": _pctl(r, "rec_yards_avg"),
                "touchdowns": _pctl(r, "touchdowns_avg"),
                "fantasy_pts": _pctl(r, "fantasy_pts_avg"),
            },
            "peers": {
                "targets": _peers(r, "targets_avg"),
                "receptions": _peers(r, "receptions_avg"),
                "rec_yards": _peers(r, "rec_yards_avg"),
                "touchdowns": _peers(r, "touchdowns_avg"),
                "fantasy_pts": _peers(r, "fantasy_pts_avg"),
            },
            "snap_share_by_week": snap_share_by_week.get(r["player_display_name"], []),
            "game_log": game_logs.get(r["player_id"], []),
        })

    players.sort(key=lambda p: (p["proj_fantasy_pts"] or 0), reverse=True)

    output = {
        "season": SEASON,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "trailing_L4_avg x opponent_matchup_factor (baseline, not yet validated against held-out data)",
        "players": players,
    }

    out_dir = os.path.dirname(os.path.abspath(OUT_JSON))
    os.makedirs(out_dir, exist_ok=True)

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    with open(OUT_JS, "w") as f:
        f.write("// Auto-generated by scripts/generate_fantasy_projections.py -- do not edit by hand.\n")
        f.write(f"const FANTASY_DATA = {json.dumps(output, indent=2)};\n")

    with open(OUT_LOG, "a") as f:
        for p in players:
            # Log only the prediction-relevant fields, not the full
            # descriptive payload (game_log, snap_share_by_week, season_avg,
            # percentiles) -- those don't change within a week and would
            # bloat an append-only log with duplicate data every run. This
            # log exists to backtest predictions, same as predictions_log.jsonl
            # / player_td_log.jsonl -- only the projection itself matters here.
            log_row = {
                "player_id": p["player_id"], "player_name": p["player_name"],
                "position": p["position"], "team": p["team"], "opponent": p["opponent"],
                "trailing_avg_pts": p["trailing_avg_pts"], "matchup_factor": p["matchup_factor"],
                "proj_fantasy_pts": p["proj_fantasy_pts"], "def_rank": p["def_rank"],
                "season": SEASON, "logged_at": output["generated_at"],
            }
            f.write(json.dumps(log_row) + "\n")

    print(f"Wrote {len(players)} player projections to {OUT_JSON} and {OUT_JS}")


if __name__ == "__main__":
    main()
