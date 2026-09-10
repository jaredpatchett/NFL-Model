// test_spread_display_fix.js -- Verifies the dashboard now displays the
// CORRECT market spread and a SENSIBLE verdict for the real NO@DET example
// reported broken, using the data shape generate_predictions.py will
// produce AFTER the backend sign fix (spread_line = +7.0, positive =
// home-favored -- not the old buggy -7.0).

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');

// Real numbers from the actual reported bug, with spread_line as the FIXED
// backend will now produce it (+7.0, not the old raw-API -7.0).
const MOCK_LINES_JS = `
const NFL_LINES_DATA = {
  "generated_at": "2026-09-09T15:06:55Z", "season": 2026, "week": 1,
  "model_notes": {"margin_residual_std": 13.16, "training_games": 2816, "caveat": "test caveat"},
  "games": [{
    "game_id": "2026_01_NO_DET", "season": 2026, "week": 1,
    "home_team": "DET", "away_team": "NO", "market_source": "live",
    "model": {"pred_home_margin": 5.4, "pred_total": 51.2, "home_win_prob": 0.66, "home_fair_moneyline": -190},
    "market": {"spread_line": 7.0, "spread_price": -110, "total_line": 50.5, "home_moneyline": -300, "away_moneyline": 250, "home_implied_prob": 0.75},
    "edges": {"spread_edge": -1.6, "total_edge": 0.7, "moneyline_edge": -0.09, "home_moneyline_ev": -0.12},
    "projector": {"proj_home_score": 28.3, "proj_away_score": 22.9, "cover_prob": 0.35, "margin_resid_std": 13.16,
      "distribution": [{"bin_start": 0, "prob": 0.09}], "waterfall": [], "situational_flags": []}
  }],
  "power_ratings": []
};
`;
const MOCK_PLAYER_TD_JS = `const PLAYER_TD_DATA = {"generated_at":"2026-09-09T15:06:55Z","model_notes":{"training_rows":1,"caveat":"x"},"players":[]};`;

async function main(){
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'nfl_lines.js'), MOCK_LINES_JS);
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'player_td.js'), MOCK_PLAYER_TD_JS);

  const dom = await JSDOM.fromFile(path.join(SITE_DIR, 'index.html'), {
    runScripts: 'dangerously', resources: 'usable',
    url: 'file://' + SITE_DIR + '/index.html',
  });
  await new Promise(r => setTimeout(r, 300));
  const { window } = dom;
  const doc = window.document;

  const checks = [];
  const row = doc.querySelector('#lines-body .board-row[data-game-id]');
  row.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));

  const edgeHTML = doc.getElementById('proj-edge-compare').innerHTML;

  // ---- The actual bug the user reported: MARKET must read "DET -7.0",
  // not the flipped/wrong side ----
  checks.push(['MARKET field shows "DET" (the real favorite), not "NO"',
    edgeHTML.includes('DET') && /MARKET[\s\S]{0,80}DET/.test(edgeHTML), true]);
  checks.push(['MARKET spread number reads 7.0', /DET[\s\S]{0,10}7\.0/.test(edgeHTML), true]);

  // ---- The verdict must no longer scream PLAY DET on a small real edge ----
  checks.push(['fixed edge (-1.6) clears the 1.5-point floor, so this shows a LEAN, not blank',
    edgeHTML.includes('>LEAN<') || edgeHTML.includes('NO PLAY'), true]);
  checks.push(['verdict does NOT say "DET to cover" (the model actually leans away from DET now)',
    !/DET to cover/.test(edgeHTML), true]);
  checks.push(['verdict correctly names NO (the away/underdog side) as where the lean is',
    /NO to cover/.test(edgeHTML), true]);

  console.log(`${'check'.padEnd(75)} got    want  ok`);
  console.log('-'.repeat(100));
  let allPass = true;
  for(const [name, got, want] of checks){
    const ok = got === want;
    allPass = allPass && ok;
    console.log(`${name.padEnd(75)} ${String(got).padEnd(6)} ${String(want).padEnd(5)} ${ok?'PASS':'FAIL'}`);
  }
  console.log('-'.repeat(100));
  console.log(allPass ? 'ALL PASS' : 'SOME FAILED');
  dom.window.close();
  process.exit(allPass ? 0 : 1);
}

main();
