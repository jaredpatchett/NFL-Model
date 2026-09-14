"""
player_td_features.py — Layer 3/4: leakage-safe player-level feature table
for the anytime-TD probability model.

THE UPCOMING-WEEK PROBLEM (same shape as Track B's pace fix, worse here):
Leakage-safe trailing features only exist for weeks that appear in the
combined player-week table. A genuinely future week (e.g. 2026 week 1, no
plays yet) has no row at all -- there's nothing to attach trailing features
to. Track B solved this by giving the schedule (which DOES have future rows)
raw values to trail from. There is no equivalent "future schedule of player
appearances" -- nflverse doesn't publish 2026 rosters yet (confirmed: 403).

Fix: manufacture one STUB row per CANDIDATE player for the upcoming
(season, week) with all count columns set to NaN (this player's own
upcoming-game stats are genuinely unknown -- that's correct, not a bug).
Appending these stub rows to the real historical table before running the
existing shift-based trailing logic means every candidate player's REAL
prior games get picked up correctly as "prior" to their stub row, exactly
the same mechanism already proven for team pace in game_features.py.

CANDIDATE POOL / ROSTER FRESHNESS CAVEAT: there is no way to know the true
Week 1 53-man roster before final cuts happen. The candidate pool here is
"any player with meaningful trailing usage in the last 8 weeks of the most
recent completed season" -- current team is corrected using nfl_data_py's
ID crosswalk (which IS updated with current-season signings/trades/rookies,
unlike pbp/roster data). This will still miss: rookies with zero prior NFL
usage (no trailing history to build from at all) and any player who
retired/is a free agent. Known, flagged limitation -- gets fixed automatically
once real 2026 games start generating real trailing data.

ADDITIONAL CANDIDATES FROM LIVE ODDS (added after a real Week 1 gap was
found -- see README): the usage-based pool above is a REASONABLE heuristic,
not a guarantee of completeness, and it has a real, confirmed failure mode
-- see _apply_position_fallback()'s docstring. Rather than trust the
heuristic alone, build_player_td_table() also accepts additional_candidate_ids:
any player the sportsbook is actually offering an anytime-TD price on, for
the games in this week's slate, regardless of whether our own usage filter
would have caught them. A live market price is real, external evidence of
relevance that doesn't depend on our own pipeline being bug-free. These
extra players go through the EXACT same trailing-feature and position-
resolution logic as usage-based candidates -- no special-cased, thinner
version of the pipeline for them. A true rookie with no real trailing
history added this way will still correctly end up with NaN features and
get dropped by generate_player_predictions.py's existing missing-features
check, with a printed reason -- that's honest, not a bug this change routes
around.

Public API:
    build_player_td_table(seasons, upcoming_season, upcoming_week,
                           additional_candidate_ids=None) ->
        DataFrame of ALL played historical player-weeks (for training) PLUS
        stub rows for the upcoming week (for prediction), all with leakage-
        safe asof_* features, position, snap_share trailing, and
        implied_team_total attached.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from nfl_data import load_pbp, load_snaps, load_id_crosswalk, load_schedules, load_weekly
from features import build_player_week_features
from rolling_features import build_asof_player_features, ROLL_WINDOWS
from team_features import _schedule_long
from defense_features import build_matchup_ratings

# Team abbreviation mapping: nfl_data_py's ID crosswalk uses a different (and
# inconsistent -- includes legacy/relocated codes) set of team codes than
# nflverse pbp/schedule data. Only current, active franchise codes need
# mapping; legacy codes (OAK, SDC, STL, RAM) belong to old rosters and are
# dropped, not mapped, since they'd never match a current schedule anyway.
IDS_TEAM_MAP = {
    "GBP": "GB", "KCC": "KC", "LVR": "LV", "NEP": "NE", "NOS": "NO",
    "SFO": "SF", "TBB": "TB", "JAC": "JAX",
}
IDS_INACTIVE_CODES = {"FA", "FA*", "OAK", "SDC", "STL", "RAM"}

MIN_TRAILING_SNAP_SHARE = 0.15  # candidate-pool bar: meaningfully used, not garbage-time only
CANDIDATE_LOOKBACK_WEEKS = 8    # "meaningful usage in the last N games of the most recent season"
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}

SNAP_SHARE_WINDOWS = ROLL_WINDOWS


def _map_ids_team(team: str) -> str | None:
    if pd.isna(team) or team in IDS_INACTIVE_CODES:
        return None
    return IDS_TEAM_MAP.get(team, team)


def _snap_share_trailing(df: pd.DataFrame, windows=ROLL_WINDOWS) -> pd.DataFrame:
    """
    snap_share is already a ratio (offense_pct), so trailing it means a
    rolling MEAN of prior weeks, not a rolling SUM like the count columns in
    rolling_features._asof_trailing. Same shift(1)-before-aggregation
    leakage-safety principle, different aggregation. Only rolling-window
    means (roll4/roll8) -- no cumulative-season version, since roll4/roll8
    already capture "recent role" well and a groupby().expanding().mean()
    cumulative version is a common source of subtle index-alignment bugs
    for comparatively little benefit here.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    out = df[["player_id", "season", "week"]].copy()
    shifted = df.groupby("player_id")["snap_share"].shift(1)
    for w in windows:
        out[f"asof_roll{w}_snap_share"] = shifted.groupby(df["player_id"]).transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
    return out


def _build_candidate_pool(player_week_feat: pd.DataFrame, upcoming_season: int) -> pd.DataFrame:
    """
    Players with real, meaningful usage in the last CANDIDATE_LOOKBACK_WEEKS
    weeks of the most recent completed season, one row per player with their
    LAST known team (pre-correction -- corrected against the ID crosswalk
    by the caller).
    """
    most_recent_season = upcoming_season - 1
    season_games = player_week_feat[player_week_feat["season"] == most_recent_season]
    if season_games.empty:
        return pd.DataFrame(columns=["player_id", "posteam"])
    max_week = season_games["week"].max()
    recent = season_games[season_games["week"] > max_week - CANDIDATE_LOOKBACK_WEEKS]

    # "Meaningful usage": snap_share bar OR real opportunity volume (guards
    # against snap_share nulls for players the crosswalk merge missed).
    usage_ok = (recent["snap_share"].fillna(0) >= MIN_TRAILING_SNAP_SHARE) | (
        (recent["rush_opp"] + recent["pass_opp"]) >= 2
    )
    candidates = recent[usage_ok]

    # One row per player: their most recent game in the window (for last-known team).
    last_game = (
        candidates.sort_values(["player_id", "season", "week"])
        .groupby("player_id", as_index=False)
        .last()
    )
    return last_game[["player_id", "posteam"]]


def _load_weekly_resilient(seasons: list[int]) -> pd.DataFrame:
    """
    load_weekly(), but tolerant of a season that 404s. Only used here for
    player_name/position enrichment -- non-critical (a missing season just
    means slightly stale name/position for that season's true rookies, not a
    broken pipeline). Observed in practice: nflverse's weekly aggregation can
    lag behind pbp/snap-count releases by weeks for the most recent season
    even once games are actually complete (2025's pbp/snaps were available,
    its weekly stats were not, at time of writing) -- worth re-checking this
    each time it's touched, since it may just resolve itself over time.

    CONFIRMED REAL-WORLD IMPACT (Week 1 2026): this lag turned out to affect
    far more than just that season's true rookies. A meaningful number of
    established, heavily-used veterans (Jahmyr Gibbs, Derrick Henry, Bijan
    Robinson, and others -- confirmed missing from a real production run)
    had no matching row in `weekly` at all, which silently dropped them
    later at the position filter in build_player_td_table() -- see
    _apply_position_fallback() for the fix.
    """
    frames = []
    for s in seasons:
        try:
            frames.append(load_weekly([s]))
        except Exception as e:
            print(f"WARNING: load_weekly([{s}]) failed ({e}) -- skipping that season "
                  f"for name/position lookup only, does not affect core features.")
    if not frames:
        return pd.DataFrame(columns=["player_id", "player_name", "position"])
    return pd.concat(frames, ignore_index=True)


def _apply_position_fallback(asof: pd.DataFrame, xwalk: pd.DataFrame) -> pd.DataFrame:
    """
    FIX for the confirmed Week 1 2026 gap (see _load_weekly_resilient()'s
    docstring): `weekly` is the ONLY source of `position` AND `player_name`
    before this function ran, and a left-merge against a source with real
    coverage gaps means real, active players can end up with position=NaN
    (and, confirmed separately in production, player_name=NaN too -- a raw
    Python float('nan') for a string field serializes as a bare, unquoted
    NaN token, which is valid JavaScript but not valid JSON; the dashboard
    then displays the literal string "NaN" when it interpolates that real
    NaN value into a template) through no fault of their own usage or the
    crosswalk's team-resolution step. The position filter right after this
    (`asof["position"].isin(VALID_POSITIONS)`) then silently drops the
    position-less rows -- NaN.isin(...) is False, with no warning printed
    anywhere. That silence is exactly how Gibbs/Henry/etc. went unnoticed
    until a real week of outcomes was checked by hand; the name gap was
    caught the same way, from a real screenshot of "NaN" rows in production.

    Fix: fall back to the ID crosswalk's OWN `position` and `name` fields
    (already loaded elsewhere in this module for team correction -- not a
    new data source, not a new API call) for any row where the weekly-
    sourced value is missing. And instead of a silent drop, print exactly
    who still has no position after BOTH sources are tried, so a gap like
    this is visible the next time it happens rather than discovered a week
    later from real game outcomes.
    """
    xwalk_lookup = xwalk[["gsis_id", "position", "name"]].rename(
        columns={"gsis_id": "player_id", "position": "crosswalk_position", "name": "crosswalk_name"}
    ).drop_duplicates(subset=["player_id"])
    asof = asof.merge(xwalk_lookup, on="player_id", how="left")
    asof["position"] = asof["position"].where(asof["position"].notna(), asof["crosswalk_position"])
    asof["player_name"] = asof["player_name"].where(asof["player_name"].notna(), asof["crosswalk_name"])
    asof = asof.drop(columns=["crosswalk_position", "crosswalk_name"])

    still_missing_pos = asof[asof["position"].isna()]["player_id"].unique()
    if len(still_missing_pos):
        print(f"WARNING: {len(still_missing_pos)} player_id(s) have no position from EITHER "
              f"weekly data or the ID crosswalk, and will be dropped by the position filter "
              f"below (real, not usage-related -- likely a genuine data gap for that player_id, "
              f"worth a manual check if any of these look like they should be active):")
        print(f"  {list(still_missing_pos)}")
    still_missing_name = asof[asof["player_name"].isna()]["player_id"].unique()
    if len(still_missing_name):
        print(f"WARNING: {len(still_missing_name)} player_id(s) have no player_name from EITHER "
              f"source -- these will still get a real prediction (name is cosmetic, unlike "
              f"position which is used as a real feature), but will display as a literal 'NaN' "
              f"or blank in any downstream UI until this resolves itself. Worth a manual check:")
        print(f"  {list(still_missing_name)}")
    return asof


def build_player_td_table(
    seasons: list[int], upcoming_season: int, upcoming_week: int,
    additional_candidate_ids: set[str] | None = None,
) -> pd.DataFrame:
    # Same "no data for a season with zero games played" guard as Track B's
    # game_features.py (2026 pbp/weekly/snaps all 404 right now, correctly --
    # verify this against nfl_data.py's docstring if it ever looks wrong).
    sched_all = load_schedules(seasons)
    played_seasons = [
        s for s in seasons
        if sched_all.loc[sched_all["season"] == s, "home_score"].notna().any()
    ]

    pbp = load_pbp(played_seasons)
    snaps = load_snaps(played_seasons)
    xwalk = load_id_crosswalk()
    weekly = _load_weekly_resilient(played_seasons)

    player_feat = build_player_week_features(pbp, snaps, id_crosswalk=xwalk)
    del pbp  # large multi-season frame, not needed again -- build_matchup_ratings()
             # loads its own pbp copy below; freeing this first avoids holding two
             # full-size copies in memory at once (this crashed the process once
             # already before this fix -- see README for the concrete OOM story).

    # ---- Candidate pool + team correction for the upcoming week's stub rows ----
    candidates = _build_candidate_pool(player_feat, upcoming_season)
    xwalk_teams = xwalk[["gsis_id", "team"]].rename(columns={"gsis_id": "player_id"})
    # CONFIRMED REAL BUG (Week 1 2026): load_id_crosswalk() only deduplicates
    # by pfr_id, not gsis_id/player_id -- if the same player has multiple
    # pfr_id-distinct rows sharing one gsis_id (real, observed in the ID
    # crosswalk data), the merge below fans candidates out into duplicate
    # rows for the same player, each potentially predicted from a slightly
    # different feature snapshot. Confirmed in production: S.Barkley alone
    # had 4 duplicate rows with 4 different probabilities. Deduplicating
    # HERE, at the source, protects every downstream use of xwalk_teams
    # (both the original usage-based path and the additional-candidates
    # path below), rather than patching each merge site separately.
    xwalk_teams = xwalk_teams.drop_duplicates(subset=["player_id"], keep="first")
    xwalk_teams["current_team"] = xwalk_teams["team"].apply(_map_ids_team)
    candidates = candidates.merge(xwalk_teams[["player_id", "current_team"]], on="player_id", how="left")
    # Fall back to last-known team if the crosswalk has no current entry (e.g. missed merge).
    candidates["resolved_team"] = candidates["current_team"].fillna(candidates["posteam"])
    candidates = candidates.dropna(subset=["resolved_team"])
    # Defensive final dedup -- xwalk_teams is now deduplicated at the source
    # above, but this stays as a safety net: a candidate table should never
    # have two rows for the same player_id, and this guarantees it regardless
    # of whether some other, not-yet-found path could also produce a dupe.
    candidates = candidates.drop_duplicates(subset=["player_id"], keep="first")

    # ---- Additional candidates from live sportsbook odds (see module
    # docstring) -- a player with a real anytime-TD price is real evidence
    # of relevance independent of our own usage-heuristic's blind spots.
    # These go through the identical stub-row/trailing-feature/position-
    # resolution path below as usage-based candidates; nothing thinner.
    if additional_candidate_ids:
        already_included = set(candidates["player_id"])
        new_ids = set(additional_candidate_ids) - already_included
        if new_ids:
            new_teams = xwalk_teams[xwalk_teams["player_id"].isin(new_ids)][["player_id", "current_team"]]
            new_teams = new_teams.rename(columns={"current_team": "resolved_team"}).dropna(subset=["resolved_team"])
            found_ids = set(new_teams["player_id"])
            unresolved = new_ids - found_ids
            if unresolved:
                print(f"WARNING: {len(unresolved)} player_id(s) had a live odds price but no "
                      f"resolvable current team in the ID crosswalk, so they can't be added as "
                      f"candidates (no team means no schedule/matchup context to build features "
                      f"from): {list(unresolved)}")
            print(f"Adding {len(found_ids)} additional candidate(s) from live odds coverage "
                  f"that the usage-based pool alone did not include.")
            candidates = pd.concat([candidates[["player_id", "resolved_team"]], new_teams], ignore_index=True)

    stub_rows = pd.DataFrame({
        "season": upcoming_season, "week": upcoming_week,
        "posteam": candidates["resolved_team"], "player_id": candidates["player_id"],
    })
    # All count/outcome columns unknown for a future game -- NaN, not 0. This
    # is what makes the shift-based trailing logic treat this row correctly
    # as "nothing happened yet" rather than "this player did nothing."
    count_cols = [
        "rush_opp", "rush_td", "rush_inside5", "rush_inside10", "rush_rz", "rush_field",
        "pass_opp", "pass_td", "pass_inside5", "pass_inside10", "pass_rz", "pass_field",
        "expected_tds", "actual_tds", "scored_td", "snap_share",
        "carry_share", "target_share", "inside5_carry_share", "rz_carry_share",
        "rz_target_share", "inside10_opp_share",
        "team_rush_opp", "team_pass_opp", "team_rush_inside5", "team_rush_inside10",
        "team_rush_rz", "team_pass_rz", "team_pass_inside10",
    ]
    for c in count_cols:
        if c in player_feat.columns:
            stub_rows[c] = np.nan

    combined = pd.concat([player_feat, stub_rows], ignore_index=True, sort=False)

    # ---- Leakage-safe trailing (player shares + team context), including stubs ----
    asof = build_asof_player_features(combined)
    snap_trailing = _snap_share_trailing(combined)
    asof = asof.merge(snap_trailing, on=["player_id", "season", "week"], how="left")

    # ---- Position / player name: most recent known value, carried forward ----
    names = (
        weekly[["player_id", "player_name", "position"]]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"], keep="last")
    )
    asof = asof.merge(names, on="player_id", how="left")
    # FIX (see _apply_position_fallback docstring): weekly-only position
    # lookup has confirmed real coverage gaps -- fall back to the crosswalk
    # before filtering on position, and print anyone still unresolved.
    asof = _apply_position_fallback(asof, xwalk)

    # ---- Implied team total: legitimate, un-lagged (set before kickoff, not
    # a leakage risk the way trailing stats are) -- same formula as Track B.
    sched_ctx = _schedule_long(sched_all)[
        ["season", "week", "posteam", "opponent", "implied_team_total", "is_home"]
    ]
    asof = asof.merge(sched_ctx, on=["season", "week", "posteam"], how="left")

    # ---- Opponent matchup rating: position-specific TD-rate-allowed, real
    # from play-by-play, not a single blended points-allowed number. See
    # defense_features.py's docstring for the full formula and reasoning.
    # Informational only right now (surfaced in output, not yet a hard
    # qualifier in blueprint_qualification.py) -- new and not yet validated
    # against a real season of results.
    matchup = build_matchup_ratings(seasons)
    matchup_cols = matchup[[
        "season", "week", "defteam",
        "matchup_rating_roll4_rush", "matchup_rating_roll4_pass",
        "asof_roll4_rush_td_rate_allowed", "asof_roll4_pass_td_rate_allowed",
    ]].rename(columns={"defteam": "opponent"})
    asof = asof.merge(matchup_cols, on=["season", "week", "opponent"], how="left")

    # Position-appropriate single number: RB -> rush matchup, WR/TE/QB -> pass
    # matchup (QBs mostly score via passing TDs to others, but their OWN
    # rushing TDs use the rush rating -- approximated here with pass as the
    # default since most QB scoring in this model's positions list is via
    # designed/scramble runs that don't cleanly map to either category; worth
    # revisiting once there's real data to check which fits QBs better).
    asof["matchup_rating"] = np.where(
        asof["position"] == "RB",
        asof["matchup_rating_roll4_rush"],
        asof["matchup_rating_roll4_pass"],
    )

    # Skill positions only -- OL/DL/specialists essentially never score
    # offensive TDs and just add noise; a missing position is either a stale/
    # unmatched crosswalk entry or a season load_weekly() failed to cover
    # (see _load_weekly_resilient) -- neither is usable here. Position
    # fallback above means a NaN here now genuinely means "unresolvable in
    # either source", already warned about, not a silent, avoidable drop.
    asof = asof[asof["position"].isin(list(VALID_POSITIONS))].reset_index(drop=True)

    return asof


if __name__ == "__main__":
    print("Run test_player_td_features.py to validate this module.")
