// test_dashboard_data_loading.js -- Functional test proving the window.X
// scoping bug fix actually works, not just that the file parses.
//
// Loads the REAL index.html from a real file:// URL with jsdom's actual
// resource loader (runScripts: 'dangerously' + resources: 'usable'), so
// the two data <script src> tags execute exactly as a real browser would --
// each a separate classic script, sharing one persistent global lexical
// scope for their top-level `const` declarations. This replicates the
// live-site failure precisely (data files return fine, the bug was in how
// the inline script reads them) and is why an earlier version of this test,
// which manually stitched scripts together with window.eval(), gave a
// false failure: indirect eval doesn't create that same persistent
// script-scope, so it wasn't actually testing the real mechanism.

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');

const MOCK_LINES_JS = `
const NFL_LINES_DATA = {
  "generated_at": "2026-09-09T15:06:55Z",
  "season": 2026,
  "week": 1,
  "model_notes": {"margin_residual_std": 13.16, "training_games": 2816, "caveat": "test caveat"},
  "games": [{
    "game_id": "2026_01_NE_SEA", "season": 2026, "week": 1,
    "home_team": "SEA", "away_team": "NE", "market_source": "live",
    "model": {"pred_home_margin": 5.96, "pred_total": 45.19, "home_win_prob": 0.6748, "home_fair_moneyline": -207.5},
    "market": {"spread_line": -3.5, "spread_price": -102, "total_line": 44.5, "home_moneyline": -170, "away_moneyline": 157, "home_implied_prob": 0.6296},
    "edges": {"spread_edge": 9.46, "total_edge": 0.69, "moneyline_edge": 0.0451, "home_moneyline_ev": 0.0717},
    "projector": {"proj_home_score": 25.6, "proj_away_score": 19.6, "cover_prob": 0.764, "margin_resid_std": 13.16,
      "distribution": [{"bin_start": 0, "prob": 0.09}], "waterfall": [{"label": "Power rating", "contribution": 0.4}],
      "situational_flags": []}
  }],
  "power_ratings": [{"team": "SEA", "net_rating": 2.29, "pace": 239.0}]
};
`;

const MOCK_PLAYER_TD_JS = `
const PLAYER_TD_DATA = {
  "generated_at": "2026-09-09T15:06:55Z",
  "model_notes": {"training_rows": 12345, "caveat": "test caveat"},
  "players": [{
    "player_name": "Test Player", "position": "RB", "team": "SEA", "matchup": "vs NE",
    "model": {"anytime_td_prob": 0.42},
    "usage": {"matchup_rating": 1.05},
    "market": {"anytime_td_price": -150, "edge": 0.03, "source": "live"},
    "qualification": {"qualifies": true, "tier": "A", "reason_codes": []}
  }]
};
`;

async function runTest(name, linesJs, playerJs, expectLinesLoaded, expectPropsLoaded) {
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'nfl_lines.js'), linesJs);
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'player_td.js'), playerJs);

  const dom = await JSDOM.fromFile(path.join(SITE_DIR, 'index.html'), {
    runScripts: 'dangerously',
    resources: 'usable',
    url: 'file://' + SITE_DIR + '/index.html',
  });

  // Give the real <script src> tags time to load from disk and execute.
  await new Promise(resolve => setTimeout(resolve, 300));

  const { window } = dom;
  const linesCaveat = window.document.getElementById('lines-caveat').textContent;
  const propsCaveat = window.document.getElementById('props-caveat').textContent;
  const linesLoaded = !linesCaveat.includes('No data found');
  const propsLoaded = !propsCaveat.includes('No data found');

  const linesOk = linesLoaded === expectLinesLoaded;
  const propsOk = propsLoaded === expectPropsLoaded;
  console.log(name);
  console.log(`  lines loaded=${linesLoaded} (expected ${expectLinesLoaded}) ${linesOk ? 'PASS' : 'FAIL'}`);
  console.log(`  props loaded=${propsLoaded} (expected ${expectPropsLoaded}) ${propsOk ? 'PASS' : 'FAIL'}`);
  let rowsOk = true;
  if (linesLoaded) {
    const gamesCount = window.document.querySelectorAll('#lines-body .board-row').length;
    rowsOk = gamesCount === 1;
    console.log(`  real rendered game rows: ${gamesCount} ${rowsOk ? 'PASS' : 'FAIL'}`);
    const weekLabel = window.document.getElementById('week-label').textContent;
    const weekOk = weekLabel === 'WEEK 1 · 2026';
    console.log(`  week label from real data: "${weekLabel}" ${weekOk ? 'PASS' : 'FAIL'}`);
    rowsOk = rowsOk && weekOk;
  }
  dom.window.close();
  return linesOk && propsOk && rowsOk;
}

(async () => {
  let allPass = true;

  allPass = (await runTest(
    'Real scenario: data files present (const-declared, loaded as real <script src> files)',
    MOCK_LINES_JS, MOCK_PLAYER_TD_JS, true, true
  )) && allPass;

  allPass = (await runTest(
    'Edge case: data files genuinely missing/empty -- must not throw, must show real fallback',
    '', '', false, false
  )) && allPass;

  console.log(allPass ? '\nALL PASS' : '\nSOME FAILED');
  process.exit(allPass ? 0 : 1);
})();
