INTEGRATION.md -- wiring team_snapshot.py into generate_predictions.py
=========================================================================

team_snapshot.py is additive only. It takes the same schedule DataFrame
generate_predictions.py already loads, plus the same (season, week) it
already computes -- nothing new to fetch, nothing existing to change.

1. Add the import near the top, alongside the other local imports:

       from team_snapshot import build_team_snapshots

2. In main(), after `full = build_game_model_table(ALL_SEASONS)` and after
   `season, week = target` are known, build the snapshots. This needs the
   raw schedule (not the team-game-long table build_game_model_table
   produces), so load it once more -- nfl_data.py caches to parquet, so
   this is a cheap local read, not a second network fetch:

       from nfl_data import load_schedules
       sched_for_snapshots = load_schedules(ALL_SEASONS)
       team_snapshots = build_team_snapshots(sched_for_snapshots, season, week)

3. Add it to the `output` dict, alongside "games" and "power_ratings":

       output = {
           ...,
           "games": games,
           "power_ratings": power_ratings_out,
           "team_snapshots": team_snapshots,   # {6: {...}, 12: {...}, 18: {...}}
       }

That's the whole integration. Nothing else in generate_predictions.py needs
to change -- this only adds a new key to the JSON/JS output.

Output shape reference (per team, per window):
    {
      "games": 6,
      "scoring": 24.0, "scoring_pctile": 78.5,
      "defense": 18.8, "defense_pctile": 62.0,
      "net": 3.4,      "net_pctile": 81.0,
      "form": 2.1,     "form_pctile": 70.0,
      "ceiling": 40.0, "ceiling_pctile": 90.0,
      "strength": 80
    }
A team with fewer than N games available (early season) reports its real
"games" count -- never silently padded -- and a team with literally zero
games in scope is simply absent from that window's dict.
