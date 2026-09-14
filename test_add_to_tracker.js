// test_add_to_tracker.js -- Verifies the "Add to Tracker" button on Edge
// Board rows for all three markets: correct game/side/odds computed and
// pre-filled into the real Bet Tracker form, correct tab switch, doesn't
// also open the Matchup Projector modal, and doesn't silently add a bet
// (stake stays blank, real "+ Add Bet" click still required). Uses one
// game with a NEGATIVE spread/total edge specifically to exercise the
// away-team/Under branches, not just the straightforward positive case.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');
const MOCK_PLAYER_TD_JS = `const PLAYER_TD_DATA = null;`;

// Game A: positive spread edge (home DET is the play), positive total edge (Over)
// Game B: NEGATIVE spread edge (away NO is the play), NEGATIVE total edge (Under)
function game(id, home, away, spreadLine, spreadEdge, totalLine, totalEdge, spreadPrice, homeML) {
  return {
    game_id: id, season: 2026, week: 1, home_team: home, away_team: away, market_source: 'live',
    model: {pred_home_margin: spreadLine + spreadEdge, pred_total: totalLine + totalEdge, home_win_prob: 0.6,
            home_fair_moneyline: -150},
    market: {spread_line: spreadLine, spread_price: spreadPrice, total_line: totalLine, home_moneyline: homeML,
             away_moneyline: 120, home_implied_prob: 0.583},
    edges: {spread_edge: spreadEdge, total_edge: totalEdge, moneyline_edge: 0.02, home_moneyline_ev: 0.06},
    projector: {proj_home_score: 24.5, proj_away_score: 20.5, cover_prob: 0.6, margin_resid_std: 13.16,
                distribution: [{bin_start: 0, prob: 0.09}], waterfall: [], situational_flags: []},
  };
}

const MOCK_LINES_JS = `
const NFL_LINES_DATA = {
  "generated_at": "2026-09-12T12:00:00Z", "season": 2026, "week": 1,
  "model_notes": {"margin_residual_std": 13.16, "training_games": 2820, "caveat": "test"},
  "games": [
    ${JSON.stringify(game('gameA', 'DET', 'NO', 5.0, 3.5, 45.0, 1.8, -108, -190))},
    ${JSON.stringify(game('gameB', 'PHI', 'WAS', 5.0, -3.5, 45.0, -1.8, null, -140))}
  ],
  "power_ratings": []
};
`;

let server, baseURL;
function startServer() {
  return new Promise((resolve) => {
    server = http.createServer((req, res) => {
      const filePath = path.join(SITE_DIR, decodeURIComponent(req.url.split('?')[0]));
      fs.readFile(filePath, (err, data) => {
        if (err) { res.writeHead(404); res.end(); return; }
        const ext = path.extname(filePath);
        const type = ext === '.js' ? 'application/javascript' : 'text/html';
        res.writeHead(200, {'Content-Type': type});
        res.end(data);
      });
    });
    server.listen(0, () => { baseURL = `http://127.0.0.1:${server.address().port}/`; resolve(); });
  });
}

async function main() {
  await startServer();
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'nfl_lines.js'), MOCK_LINES_JS);
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'player_td.js'), MOCK_PLAYER_TD_JS);

  const dom = await JSDOM.fromURL(baseURL + 'index.html', {runScripts: 'dangerously', resources: 'usable'});
  await new Promise(r => setTimeout(r, 300));
  const doc = dom.window.document;
  const win = dom.window;

  const checks = [];

  function trackBtnFor(gameId) {
    return doc.querySelector(`[data-game-id="${gameId}"][data-action="add-to-tracker"]`);
  }

  // ---- SPREAD: Game A (home is the play, real spread_price present) ----
  trackBtnFor('gameA').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  checks.push(['switched to Tracker tab', doc.getElementById('view-tracker').classList.contains('active'), true]);
  checks.push(['game field: away @ home format', doc.getElementById('tf-game').value, 'NO @ DET']);
  checks.push(['market field: Spread', doc.getElementById('tf-market').value, 'Spread']);
  checks.push(['side field: home team, correct spread number (home favored by 5 -> DET -5.0)',
    doc.getElementById('tf-side').value, 'DET -5.0']);
  checks.push(['odds field: real spread_price used when available', doc.getElementById('tf-odds').value, '-108']);
  checks.push(['stake field left BLANK -- never silently defaulted', doc.getElementById('tf-stake').value, '']);
  checks.push(['modal did NOT open (stopPropagation worked)', doc.getElementById('projector-modal').classList.contains('open'), false]);
  checks.push(['no bet was actually added yet (0 bets in tracker)', doc.querySelectorAll('.board-row[data-bet-id]').length, 0]);

  // ---- SPREAD: Game B (away is the play, NO spread_price -> -110 default) ----
  doc.querySelector('[data-view="lines"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  trackBtnFor('gameB').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  checks.push(['negative edge: away team is correctly the play (WAS, not PHI)',
    doc.getElementById('tf-side').value, 'WAS +5.0']);
  checks.push(['no spread_price available -> falls back to standard -110 default', doc.getElementById('tf-odds').value, '-110']);

  // ---- TOTAL: Game A (positive edge -> Over) ----
  doc.querySelector('[data-view="lines"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  doc.querySelector('[data-market="total"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  trackBtnFor('gameA').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  checks.push(['total, positive edge -> Over', doc.getElementById('tf-side').value, 'Over 45.0']);
  checks.push(['total market label set correctly', doc.getElementById('tf-market').value, 'Total']);

  // ---- TOTAL: Game B (negative edge -> Under) ----
  doc.querySelector('[data-view="lines"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  doc.querySelector('[data-market="total"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  trackBtnFor('gameB').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  checks.push(['total, negative edge -> Under', doc.getElementById('tf-side').value, 'Under 45.0']);

  // ---- MONEYLINE: Game A ----
  doc.querySelector('[data-view="lines"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  doc.querySelector('[data-market="moneyline"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  trackBtnFor('gameA').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  checks.push(['moneyline side: home team ML', doc.getElementById('tf-side').value, 'DET ML']);
  checks.push(['moneyline odds: real home_moneyline used', doc.getElementById('tf-odds').value, '-190']);
  checks.push(['moneyline market label set correctly', doc.getElementById('tf-market').value, 'Moneyline']);

  // ---- Regression: clicking elsewhere on a row still opens the modal ----
  doc.querySelector('[data-view="lines"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  doc.querySelector('[data-market="spread"]').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  doc.querySelector('[data-game-id="gameA"] .matchup-main').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  checks.push(['clicking the row body (not the track button) still opens the modal',
    doc.getElementById('projector-modal').classList.contains('open'), true]);

  console.log(`${'check'.padEnd(80)} got            want           ok`);
  console.log('-'.repeat(120));
  let allPass = true;
  for (const [name, got, want] of checks) {
    const ok = got === want;
    allPass = allPass && ok;
    console.log(`${name.padEnd(80)} ${String(got).padEnd(14)} ${String(want).padEnd(14)} ${ok ? 'PASS' : 'FAIL'}`);
  }
  console.log('-'.repeat(120));
  console.log(allPass ? 'ALL PASS' : 'SOME FAILED');
  server.close();
  dom.window.close();
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); if (server) server.close(); process.exit(1); });
