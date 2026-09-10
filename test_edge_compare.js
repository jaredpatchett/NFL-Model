// test_edge_compare.js -- Functional test for the redesigned Matchup
// Projector header + edge-compare section: real data in, real simulated
// click to open the modal, checking the actual rendered DOM text -- not
// just that the JS parses.

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');

// Real-shaped game: SEA home favored by 3.5 in the market, model likes SEA
// even more (home margin 5.96) -> positive spread_edge -> should read as a
// PLAY on SEA (home) covering, since |9.46| >= 3.
// Total: market 44.5, model 42.9 -> total_edge negative -> LEAN UNDER.
const MOCK_LINES_JS = `
const NFL_LINES_DATA = {
  "generated_at": "2026-09-09T15:06:55Z", "season": 2026, "week": 1,
  "model_notes": {"margin_residual_std": 13.16, "training_games": 2816, "caveat": "test caveat"},
  "games": [{
    "game_id": "2026_01_NE_SEA", "season": 2026, "week": 1,
    "home_team": "SEA", "away_team": "NE", "market_source": "live",
    "model": {"pred_home_margin": 5.96, "pred_total": 42.9, "home_win_prob": 0.6748, "home_fair_moneyline": -207.5},
    "market": {"spread_line": -3.5, "spread_price": -102, "total_line": 44.5, "home_moneyline": -170, "away_moneyline": 157, "home_implied_prob": 0.6296},
    "edges": {"spread_edge": 9.46, "total_edge": -1.6, "moneyline_edge": 0.0451, "home_moneyline_ev": 0.0717},
    "projector": {"proj_home_score": 24.1, "proj_away_score": 18.8, "cover_prob": 0.764, "margin_resid_std": 13.16,
      "distribution": [{"bin_start": 0, "prob": 0.09}], "waterfall": [{"label": "Power rating", "contribution": 0.4}],
      "situational_flags": []}
  }, {
    "game_id": "2026_01_TB_CIN", "season": 2026, "week": 1,
    "home_team": "CIN", "away_team": "TB", "market_source": "live",
    "model": {"pred_home_margin": 0.4, "pred_total": 50.6, "home_win_prob": 0.51, "home_fair_moneyline": -104},
    "market": {"spread_line": -0.5, "spread_price": -105, "total_line": 50.5, "home_moneyline": -108, "away_moneyline": -102, "home_implied_prob": 0.51},
    "edges": {"spread_edge": 0.9, "total_edge": 0.1, "moneyline_edge": 0.0, "home_moneyline_ev": 0.0},
    "projector": {"proj_home_score": 25.5, "proj_away_score": 25.1, "cover_prob": 0.51, "margin_resid_std": 13.16,
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

  // Click the first row (SEA vs NE -- clear PLAY case) to open the real modal.
  const firstRow = doc.querySelector('#lines-body .board-row[data-game-id]');
  firstRow.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));

  checks.push(['modal opened', doc.getElementById('projector-modal').classList.contains('open'), true]);
  checks.push(['header away score', doc.getElementById('proj-away-score').textContent, '18.8']);
  checks.push(['header home score', doc.getElementById('proj-home-score').textContent, '24.1']);
  checks.push(['matchup subtitle says PROJECTED', doc.getElementById('proj-matchup-sub').textContent, 'WEEK 1 · 2026 · PROJECTED']);

  const edgeHTML = doc.getElementById('proj-edge-compare').innerHTML;
  checks.push(['spread row shows market side (SEA -3.5)', edgeHTML.includes('SEA') && edgeHTML.includes('3.5'), true]);
  checks.push(['spread row shows model side (SEA -6.0)', edgeHTML.includes('6.0'), true]);
  checks.push(['spread verdict is PLAY (edge 9.46 >= 3)', edgeHTML.includes('>PLAY<'), true]);
  checks.push(['spread verdict names the right side (SEA)', /PLAY[\s\S]{0,30}SEA to cover/.test(edgeHTML), true]);
  checks.push(['total verdict is a LEAN, not a PLAY (never plays totals)', edgeHTML.includes('LEAN UNDER'), true]);
  checks.push(['total verdict explicitly flags low confidence', edgeHTML.toLowerCase().includes('low-confidence'), true]);
  checks.push(['no literal "PLAY" tag anywhere for the total row', !(/TOTAL[\s\S]{0,400}>PLAY</.test(edgeHTML)), true]);

  // Close and open the second, marginal game -- should show NO PLAY / NO LEAN.
  doc.getElementById('modal-close').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  const secondRow = doc.querySelectorAll('#lines-body .board-row[data-game-id]')[1];
  secondRow.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  const edgeHTML2 = doc.getElementById('proj-edge-compare').innerHTML;
  checks.push(['marginal game: spread verdict is NO PLAY (edge 0.9 < 1.5)', edgeHTML2.includes('NO PLAY'), true]);
  checks.push(['marginal game: total verdict is NO LEAN (edge 0.1 negligible)', edgeHTML2.includes('NO LEAN'), true]);

  console.log(`${'check'.padEnd(58)} ${'got'.padEnd(10)} want  ok`);
  console.log('-'.repeat(90));
  let allPass = true;
  for(const [name, got, want] of checks){
    const ok = got === want;
    allPass = allPass && ok;
    console.log(`${name.padEnd(58)} ${String(got).padEnd(10)} ${String(want).padEnd(6)} ${ok?'PASS':'FAIL'}`);
  }
  console.log('-'.repeat(90));
  console.log(allPass ? 'ALL PASS' : 'SOME FAILED');
  dom.window.close();
  process.exit(allPass ? 0 : 1);
}

main();
