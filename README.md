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
    - **Total points**: market-blended, revisited and fixed this session
      (see "Total points: market-blended fix" section below) — was weak,
      now anchored to the market total with a heavily-regularized deviation
      model on top, MAE 9.776 essentially matching market-alone (9.768).
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

## Total points: market-blended fix (new this session)

The original from-scratch total model (predict game_total directly from
power ratings + trailing pace/scoring) never beat a naive mean-guess
baseline by much (MAE ~10.1 vs baseline 10.14). Revisited this session with
two changes:

1. **Added weather/pace features** (dome flag, wind, temp, trailing plays)
   in an earlier pass — helped bias/RMSE, barely moved MAE.
2. **Market-blended residual approach** (this session): instead of
   predicting the total from scratch, anchor to the market's own
   `total_line` (legitimately available pre-kickoff, not leakage) and
   predict the RESIDUAL (deviation from that anchor) using the same
   feature set.

**Honest finding from an alpha (regularization strength) sweep**: the
market's own total_line, used ALONE with zero model input, beats the
residual model at every regularization strength tried — from alpha=5
(MAE 10.00) through alpha=100,000 (MAE 9.770), approaching but never
crossing market-alone's MAE of 9.768. This isn't a tuning problem; it's
evidence this feature set doesn't contain genuine signal beyond what's
already priced into the market total.

**Response**: rather than ship a model that's measurably worse than just
trusting the market, or fabricate a smaller alpha that happened to look
better on this one validation season (overfitting to 2024's noise), the
model now uses heavy regularization (alpha=20,000) — final result:
**MAE 9.776, RMSE 12.586 — statistically indistinguishable from market-
alone (MAE 9.768, RMSE 12.587)**, while still preserving a small,
football-sane directional nudge (wind negative, power rating positive,
trailing pace positive). Production (`generate_predictions.py`) now
computes `pred_total = total_line + model_residual` instead of predicting
the total from scratch.

**What this means for `total_edge` going forward**: treat it as
low-confidence — the model doesn't have a demonstrated edge on totals the
way it does on spreads. Deviations from the market total shown in the
dashboard are now small (typically <1 point) by design, reflecting that
honest uncertainty rather than presenting a fabricated confident number.
Real improvement here would need genuinely new signal (EPA/success-rate
features, or weather forecast data beyond what's already in the schedule
pull) — not more tuning of the current feature set, which has been shown
not to contain it.

Next steps for Track B: find genuinely new total-points signal (EPA/
success-rate features — bigger lift, not yet started), build a proper
edge/pricing backtest once enough logged weeks accumulate, genuinely
time-stamped (not closing) lines for a real profitability check.

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

**CI fix applied after first real deploy**: `nfl_data_py` hard-pins
`pandas<2.0,numpy<2.0` in its own package metadata, which has no prebuilt
wheel for Python 3.12 and forces a broken source build on a fresh GitHub
Actions runner (`ModuleNotFoundError: No module named 'pkg_resources'`).
Confirmed its actual code only needs numpy/pandas/appdirs at import time and
works fine on modern versions (tested end-to-end against a clean venv
matching the runner). Fix: install it with `--no-deps`, then install modern
pandas/numpy/etc. directly — both the workflow and `requirements.txt`/README
setup instructions do this now.

## Prediction logging (new this session)

`generate_predictions.py` now writes THREE things, not two:

- `data/nfl_lines.json` / `.js` — current snapshot, **overwritten** every
  run. This is what the dashboard reads. It only ever shows the latest
  prediction — nothing about last week's version survives here.
- `data/predictions_log.jsonl` — **append-only**, one line per (game, run).
  Every cron run (Tue-Fri in season) adds a fresh timestamped row for every
  game in the upcoming week, even if nothing changed since the last run.
  Nothing already logged is ever modified or removed.

**Why this exists**: every validation done on this project so far (Layer 2,
Track B margin/total) has checked model calibration against CLOSING lines —
useful for sanity-checking the model, but explicitly NOT a valid
profitability backtest per the blueprint's own core principle (backtest with
the price actually available at prediction time, never hindsight/closing
prices). `predictions_log.jsonl` is how that gets fixed: it's a genuine
pre-game record — what the model said, what the market showed, days before
any outcome was known. Once enough weeks accumulate, join this log against
final scores (already in the schedule data once games are played, keyed by
`game_id`) for a real time-stamped edge/ROI analysis. There's no reason to
wait for the regular season to start collecting this — the log already has
its first real entries from Week 1 2026's pre-season market lines.

No backtest/analysis script consumes this log yet — that's the natural next
step once a few weeks of real data accumulate.

## Track A: Anytime-TD player props (new this session — Layers 3/4 built)

Layer 2 (team TD counts) was already validated. This session built the
piece that actually produces a bettable per-player number:

- `scripts/player_td_features.py` — the hard part. Leakage-safe trailing
  opportunity shares per player (carry/target/inside-5/red-zone share, snap
  share) via the same shift-before-aggregate mechanism as `rolling_features.py`.
  **The upcoming-week problem, again, worse this time**: nflverse doesn't
  publish 2026 rosters at all yet (confirmed — 403). Solved the same way as
  Track B's pace fix — manufacture one stub row per CANDIDATE player for the
  upcoming week (all count columns NaN, since a future game's stats are
  genuinely unknown, not zero) and let the existing trailing logic pick up
  real prior games naturally. Candidate pool = players with meaningful
  trailing usage in the last 8 weeks of the most recently completed season,
  with current team corrected against `nfl_data_py`'s ID crosswalk (a
  fantasy-community-maintained database that's updated with trades/signings
  well before nflverse's own roster data exists for a new season).
  **Verified this actually caught real 2025 offseason moves** — Aaron
  Rodgers → PIT, Kirk Cousins → LV both resolved correctly.
- `scripts/player_td_model.py` — Layers 3 (allocate team TDs across players)
  and 4 (calibrated probability) combined into one logistic regression
  rather than two separate stages, given the time budget ahead of Week 1.
  Time-based validation (train 2021-2023, validate held-out 2024): **AUC
  0.70, LogLoss 0.506 vs 0.549 baseline**, calibration tracks reasonably
  well across probability buckets (see script output for the full table).
  Coefficients are football-sane: carry/target share, implied_team_total,
  snap_share all positive; fullback position negative.
- `scripts/generate_player_predictions.py` — production script, same
  pattern as Track B's: trains on all available history, predicts the
  upcoming week, writes `data/player_td.json`/`.js` (overwritten snapshot)
  and appends to `data/player_td_log.jsonl` (persistent history, same
  reasoning as `predictions_log.jsonl` above). **Verified against the real
  2026 Week 1 slate**: Saquon Barkley tops the board at 71% anytime-TD
  probability — correct real-world sanity check for a bell-cow goal-line
  back.
- `td-props.html` — dashboard styled to match a reference design: Player /
  Matchup / Snap% / RZ% / Inside-5% / xTD / Model probability / Tier, with
  position filters. Tier here is a MODEL CONFIDENCE tier (probability
  level), not a value/edge tier like Track B's — there's no market to
  compare against yet, see below.

**NO LIVE ODDS CONNECTED — this is the main honest gap.** Output is model
probability and usage shares only; no BEST PRICE / IMPLIED / EDGE / EV
columns. Player-prop odds specifically need The Odds API's *paid* tier
(unlike game lines, which are on the free tier — see Track B notes above).
Get an Odds API key to unlock this, or keep using model-only output.

**ROSTER FRESHNESS CAVEAT**: the candidate pool is last season's active
players corrected for known trades/signings. True rookies with zero prior
NFL usage (no trailing history to build from at all) and any very recent
retirement the ID crosswalk hasn't caught yet won't appear. This resolves
itself automatically once real 2026 game data starts flowing in — Week 1's
actual snap counts will fold into next week's trailing features the normal
way.

**Test coverage gap, flagged honestly**: `player_td_features.py` doesn't
have a dedicated synthetic unit test file the way the other modules do
(time tradeoff to get the real model built and validated before Week 1).
It reuses the already-tested `rolling_features.py` trailing engine
directly, and was checked thoroughly against real data (candidate pool
size, correct team corrections, sane top-of-board predictions) rather than
synthetic fixtures. Worth adding a proper test file when there's time.

## Live odds: The Odds API (new this session)

Both tracks now fetch real sportsbook odds via `scripts/odds_api.py` and
compute genuine edge/EV — not just model probability. Degrades gracefully:
if `ODDS_API_KEY` isn't set or a call fails, both production scripts fall
back to model-only output rather than crashing.

**Correction to something stated earlier in this project**: current
(non-historical) player-prop odds — including anytime-TD — are included on
The Odds API's **free tier** (500 credits/month). Only *historical*
player-prop data requires a paid plan. Earlier notes here said props needed
a paid tier; that was wrong (or at least outdated) — checked against the
live docs this session.

- **Track B** (`generate_predictions.py`): one bulk call for h2h/spreads/
  totals across ALL games (3 credits/run — cost doesn't scale with game
  count). Best price selected across all returned bookmakers per side.
  Falls back to the schedule-riding lines (already-existing behavior) when
  no live odds are available for a game. Output now includes `market_source`
  ("live" or "schedule_fallback") and a real moneyline EV per $1 stake.
- **Track A** (`generate_player_predictions.py`): one event-odds call PER
  GAME for `player_anytime_td` (no bulk endpoint exists for additional
  markets) — costs len(games) credits per run. **The name-matching problem**:
  our internal `player_name` is abbreviated ("S.Barkley," from nflverse
  weekly data) but the API returns full names ("Saquon Barkley") — matched
  via the ID crosswalk's `merge_name` field (added to `load_id_crosswalk()`
  this session), normalized identically on both sides via `odds_api.normalize_name()`.
- **Credit budget**: at the current cadence (4 runs/week × ~16 games × 18
  weeks), both tracks combined use ~1,368 credits/season — comfortably
  inside the free tier's ~2,250/season budget (500/mo × ~4.5 months), with
  room for manual test runs. Re-check this math if cadence or game count
  assumptions change.
- **Tested against mocked responses, not a live key** (none was available
  in the build environment) — `test_odds_api.py` validates parsing logic
  (best-price selection, team-name mapping, name normalization, graceful
  no-key fallback) against realistic fixtures matching the documented
  schema, 15/15 passing. Also ran a full mocked integration test simulating
  a real Barkley anytime-TD price end-to-end through
  `generate_player_predictions.py` — model 70.96% vs market-implied 59.18%,
  edge +11.77%, EV +19.9%, correct at every step. **Worth a real smoke test
  once a key is live** — documented schemas can drift from reality, and
  this hasn't been checked against the actual API response yet.

**To activate**: add your key as a GitHub repo secret named `ODDS_API_KEY`
(Settings → Secrets and variables → Actions → New repository secret). The
workflow already passes it through to both prediction steps — no other
config needed. Trigger the workflow manually once and check the Action log
for `[odds api] ... credits remaining` lines to confirm it's actually
pulling live data rather than silently falling back.

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

    pip install --upgrade pip setuptools
    pip install --no-deps nfl_data_py
    pip install pandas numpy pyarrow scikit-learn scipy requests appdirs

nfl_data_py's own package metadata hard-pins `pandas<2.0,numpy<2.0`, which
has no prebuilt wheel for Python 3.12 and forces pip to build pandas from
source, which then fails with `ModuleNotFoundError: No module named
'pkg_resources'`. nfl_data_py's actual code only needs numpy/pandas/appdirs
at import time and works fine on modern versions (verified against a clean
venv matching a GitHub Actions runner) — hence `--no-deps` then installing
modern versions directly, rather than a plain `pip install -r
requirements.txt`. The GitHub Actions workflow already does this correctly;
this is only relevant for local setup.

## What to run

    python scripts/nfl_data.py 2024          # confirms network pull + caching
    python scripts/test_features.py          # offline, should be 10/10
    python scripts/test_rolling_features.py  # offline, should be 12/12
    python scripts/test_power_ratings.py     # offline, should be 5/5
    python scripts/test_odds_api.py          # offline (mocked), should be 15/15
    python scripts/team_features.py 2024     # builds team-week model table
    python scripts/team_td_model.py          # trains + validates Layer 2 (TD props)
    python scripts/game_features.py 2024     # builds team-game model table (Track B)
    python scripts/game_lines_model.py       # trains + validates margin/total/moneyline
    python scripts/generate_predictions.py         # production: Track B, writes data/nfl_lines.*
    python scripts/player_td_model.py              # trains + validates Track A (anytime TD)
    python scripts/generate_player_predictions.py  # production: Track A, writes data/player_td.*

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
