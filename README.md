# NFL Anytime-Touchdown Betting Model

Standalone project — a separate repo, separate Cloudflare Pages deploy, and
separate GitHub Actions cron from the wc-7-dashboard soccer project. Same
general static-site + scheduled-job *pattern* (Python in `scripts/` fetches
data + runs the model, writes to `data/`, a static site reads the output,
cron reruns + redeploys) but intentionally not merged or cross-linked with
soccer — no shared toggle, no shared data format, no shared deploy.

## Current status (updated this session)

DONE and TESTED against REAL 2021-2024 nflverse data (not just synthetic):

- `scripts/nfl_data.py` — nflverse ingestion + local parquet caching.
  **Fixed this session:** `load_schedules()` was hitting
  `http://www.habitatring.com/games.csv` via `nfl_data_py`'s `import_schedules()`,
  which returns HTTP 403 from any restricted-egress environment (confirmed in
  sandbox testing). Swapped to the GitHub-hosted mirror
  (`raw.githubusercontent.com/nflverse/nfldata/.../games.csv`), which is
  reachable everywhere GitHub is. **Added:** `load_id_crosswalk()`.
- `scripts/features.py` — play-by-play -> within-week opportunity features
  (shares, xTD). **Fixed this session:** `snap_share` was silently null for
  100% of real rows. Root cause: nflverse snap counts are keyed by PFR ID
  (`"BrowSp00"`), play-by-play/weekly data is keyed by GSIS ID
  (`"00-0034796"`) — two different ID namespaces that were being merged
  directly. The synthetic unit test didn't catch it because the fixture
  coincidentally used the same fake ID for both. Now routes through
  `load_id_crosswalk()`; regression-tested with deliberately mismatched
  synthetic IDs so this can't silently reappear. Verified on real 2024 data:
  5,424 / 5,434 rows (99.8%) now resolve correctly.
- `scripts/rolling_features.py` — **new this session.** Leakage-safe "as-of-
  kickoff" trailing features (season-to-date cumulative + rolling 4/8-game
  windows), computed via `shift(1)` before any aggregation so a player/team's
  own current-week stats can never enter their own features. This matters:
  `features.py`'s shares are computed FROM that week's own plays, which is
  fine for backtesting "what happened" but is hindsight leakage if fed into
  a model predicting that same week's outcome. Regression-tested with a
  synthetic 3-week fixture that explicitly checks week 3's trailing total
  excludes week 3's own carries.
- `scripts/team_features.py` — **new this session.** Team-week modeling
  table: trailing team opportunity stats + schedule game context (home/away,
  rest days, div game) + Vegas-implied team total from `spread_line` /
  `total_line` (sign convention verified against actual 2024 results —
  positive `spread_line` = home team favored).
- `scripts/team_td_model.py` — **new this session. Layer 2 baseline is
  built and validated.** Predicts team offensive TDs per game. Time-based
  split: train on 2021-2023, validate on the fully held-out 2024 season
  (2,278 team-games pulled). Results on held-out 2024:
    - Baseline (predict training mean): MAE 1.13
    - Poisson GLM: MAE 1.00, beats baseline
    - GBM (Poisson loss): MAE 1.05
  `implied_team_total` (the Vegas signal) dominates feature importance by a
  wide margin, as expected — the market already prices in injuries/weather/
  matchup we haven't modeled yet. Calibration by implied-total bucket is
  reasonably monotonic (see script output), with a small ~-0.13 TD
  underprediction bias across the board worth investigating further.
- `scripts/test_features.py`, `scripts/test_rolling_features.py` —
  synthetic-data unit tests, all passing (10/10 and 12/12), run offline.

KNOWN CAVEATS / NEXT STEPS (in blueprint build order):
- **Cold start**: rows from a team's/player's first ~3 games of a season have
  thin or absent trailing features. Current baseline just drops them from
  train/eval. Production needs to bridge this with prior-season trailing
  stats or a league-average prior.
- **Postseason weeks**: `team_features.py` currently includes playoff games
  in the team-week table undifferentiated from regular season. Playoff
  sample is small and dynamics differ (only good teams present, different
  game script incentives) — worth filtering to REG season only, or adding a
  `season_type` flag, before this becomes a real signal source.
- Player TD allocation model (Layer 3: share of team TDs) — next up, and
  `rolling_features.build_asof_player_features()` is already built for this.
- Direct player probability (Layer 4: calibrated classifier)
- Calibration + ensemble (Layer 5: isotonic / Platt)
- Pricing/edge engine (implied prob, fair odds, edge, EV)
- Odds ingestion (The Odds API) — anytime-TD props, timestamped
- Backtest + calibration dashboard
- `data/nfl_data.js` output for this project's own static dashboard
  (independent of the soccer project's data format).

## Track B: Spreads / Totals / Moneylines (new this session)

Separate model track from the TD props above — same data foundation
(`nfl_data.py`), different targets. Built and validated this session:

- `scripts/power_ratings.py` — opponent-adjusted SRS-style team power
  ratings, solved week-by-week via ridge regression using ONLY games strictly
  before that week (leakage-safe). Cold start at each season's week 1 uses a
  shrunk carryover of the team's final rating from the prior season.
  Sanity-checked against real 2021-2024 results: end-of-2024 ratings had
  Detroit #1, Buffalo #2, Philadelphia #3 (won the Super Bowl), Carolina/
  Cleveland/Giants at the bottom — matches reality. Solved home-field
  advantage: ~1.6 points, in line with modern NFL estimates. Regression-
  tested (hand-computable 2-team system + explicit leakage check).
- `scripts/game_features.py` — team-game modeling table: power ratings +
  trailing points scored/allowed (leakage-safe) + rest/home/division context.
  Market lines (moneyline, spread, total) are carried through UNCHANGED for
  comparison only — never used as model inputs.
- `scripts/game_lines_model.py` — margin, total, and moneyline models,
  same time-based validation as Layer 2 (train 2021-2023, validate held-out
  2024):
    - **Margin (spread)**: Ridge MAE 10.27 vs baseline 11.30 — beats baseline,
      correctly-signed coefficients (own rating +, opponent rating -, home
      field +).
    - **Total points**: Ridge/GBM MAE ~10.1 vs baseline 10.14 — barely beats
      baseline. Total points is a genuinely hard target (weather, game
      script); this is a known limitation of power-rating approaches, not a
      bug, and is a good candidate for future improvement (pace/EPA features,
      weather data).
    - **Moneyline**: win probability via normal CDF of projected margin
      (residual std ~13.2 pts), converted to fair odds. Lands slightly more
      conservative than market-implied probability, as expected (market
      prices in injury news / info this model doesn't see).

**CLOSING-LINE CAVEAT**: the schedule data's lines are closing lines
(nflverse/nfldata), not time-stamped opening/live lines. Fine for validating
model calibration, NOT valid for a real profitability backtest — that needs
The Odds API's time-stamped historical lines, same constraint noted for TD
props above.

**Data source advantage**: unlike TD props (needs The Odds API's paid plan
for historical player-prop odds), moneyline/spread/total odds are available
on The Odds API's free/cheap tier — cheaper path to live odds for this track.

Next steps for Track B: improve the total-points model (weather, pace/EPA
features), pull live odds via The Odds API, build the edge/pricing
comparison, backtest with genuinely time-stamped (not closing) lines.

## Automation & dashboard (new this session)

- `scripts/generate_predictions.py` — production entrypoint. Trains margin
  and total Ridge models on ALL available played history (not held out —
  the held-out validation already happened in `game_lines_model.py`; this
  script's job is to produce the best live prediction, not report accuracy),
  finds the earliest upcoming week that has posted market lines, predicts
  it, and writes `data/nfl_lines.json` + `data/nfl_lines.js`. Verified
  end-to-end against the real, already-published 2026 Week 1 schedule
  (season hasn't started yet, but 2026 lines are live — 16 games, all
  predicted correctly with sane spread/total/moneyline edges).
- `.github/workflows/update-predictions.yml` — cron (Tue-Fri, 13:00 UTC;
  adjust as needed), installs deps, runs `generate_predictions.py`, commits
  `data/nfl_lines.json`/`.js` if changed, pushes. Standard `workflow_dispatch`
  manual trigger included too. **Independent workflow, independent repo** —
  no shared cron with `wc-7-dashboard`.
- `index.html` — static dashboard reading `data/nfl_lines.js`. Dark
  data-terminal aesthetic (this is a solo quant tool, not a consumer app):
  a "largest spread edge" callout up top, full sortable-by-edge game table
  below. No build step — Cloudflare Pages can serve this directory as-is.

**One real production wrinkle already handled**: play-by-play data doesn't
exist for a season with zero games played (confirmed — 2026 pbp pull 404s
right now, correctly). The pipeline was patched to only pull pbp for seasons
with at least one played game, while still solving power ratings and
predicting the upcoming week using the schedule's already-published lines.
Worth remembering if this breaks again after a data-source change: pbp and
schedule data are NOT available on the same timeline.

**To deploy**: push this repo, connect a new (separate) Cloudflare Pages
project pointed at it, build output directory = repo root, no build command
needed (static HTML). The GitHub Actions workflow's default `GITHUB_TOKEN`
permissions are already set (`contents: write`) — no extra secrets needed
for this workflow as written.

## Architecture

Static-site + scheduled-job pattern: Python scripts in `scripts/` fetch data
+ run the model, write to `data/`; a static site reads the data file (no
live compute); GitHub Actions cron reruns scripts, commits output,
Cloudflare Pages redeploys. This is its own independent deploy — separate
repo, separate Pages project, separate cron — not part of the soccer
project's site or data pipeline.

Source of truth = nflverse fetch + parquet cache, not hand-fed data (unlike
soccer repo's hand-appended JSON).

## Setup

    pip install -r requirements.txt

(If `pip install pandas` tries to build from source and fails on
`ModuleNotFoundError: No module named 'pkg_resources'`, run
`pip install setuptools` first, then retry with `--only-binary=:all:`.)

## What to run

    python scripts/nfl_data.py 2024          # confirms network pull + caching
    python scripts/test_features.py          # offline, should be 10/10
    python scripts/test_rolling_features.py  # offline, should be 12/12
    python scripts/test_power_ratings.py     # offline, should be 5/5
    python scripts/team_features.py 2024     # builds team-week model table
    python scripts/team_td_model.py          # trains + validates Layer 2 (TD props)
    python scripts/game_features.py 2024     # builds team-game model table (Track B)
    python scripts/game_lines_model.py       # trains + validates margin/total/moneyline
    python scripts/generate_predictions.py   # production: predicts upcoming week, writes data/

## Data sources decided

- Player/team stats: nflverse (nfl_data_py) — FREE. Foundation. Red-zone /
  inside-10 / inside-5 usage is DERIVED from play-by-play
  (yardline_100 <= 20 / 10 / 5), not purchased.
- Schedules + Vegas lines (spread/total): nflverse's GitHub-hosted
  `nfldata/games.csv` mirror (NOT `nfl_data_py.import_schedules()`, which
  hits a domain that 403s from restricted-egress environments).
- Historical anytime-TD odds: The Odds API paid plan (~2.5 seasons,
  5-min snapshots from May 2023). THE binding constraint for honest
  backtesting. Not yet integrated.
- Later upgrades (only after signal proven): PFF (coverage, O-line),
  OpticOdds/OddsJam (more books, lower latency).

## Key modeling principles (from the blueprint)

- Value != likelihood. Bet only when model prob meaningfully beats the
  price-implied prob.
- Opportunity predicts TDs better than recent TD results.
- Model team TDs first, then allocate to players. More stable.
- Calibrate probabilities (isotonic/Platt) so 35% really wins ~35%.
- Backtest with time-based splits and the ACTUAL price available at
  prediction time. Never closing/hindsight prices. (Layer 2 above already
  follows this — train 2021-2023, validate fully-held-out 2024.)
- Anytime-TD props are heavily juiced (15-25% vig two-way). Edge must
  clear the vig. Much of real profit is in speed + line shopping.

## Core math (implemented in pricing engine, to build)

- Positive American odds -> implied prob = 100 / (odds + 100)
- Negative -> |odds| / (|odds| + 100)
- Model prob p (<50%) -> fair positive odds = 100 * (1 - p) / p
- Edge = model prob - implied prob
- EV (1u at + odds) = p * profit - (1 - p)

## Implied team total (Layer 2 signal)

Verified sign convention against actual 2024 results — `spread_line` here is
a "home margin" convention (positive = home favored), not the traditional
negative-favorite American-odds convention:

    implied_home_total = (total_line + spread_line) / 2
    implied_away_total = (total_line - spread_line) / 2
