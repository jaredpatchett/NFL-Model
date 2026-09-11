// test_hero_cards.js -- Verifies the TD Props hero cards render correctly:
// sorted by model probability descending, top 6 shown, correct percentage
// and progress bar, correct footer text for both a priced and an unpriced
// player, and team color applied to the card border.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');
const MOCK_LINES_JS = `const NFL_LINES_DATA = null;`;

function player(name, prob, team, matchup, position, market) {
  return {
    player_name: name, position, team, opponent: 'XXX', matchup,
    team_implied_total: 24.0,
    usage: {snap_share: 0.7, rz_target_share: 0.2, inside5_carry_share: 0.5, xtd_per_game: 1.0, matchup_rating: 1.0},
    model: {anytime_td_prob: prob, fair_moneyline: -150, tier: 'A'},
    market,
    qualification: {qualifies: true, tier: 'A', reason_codes: []},
  };
}

// 8 players (more than the top-6 cutoff) with DELIBERATELY unsorted input
// order, to confirm the card row re-sorts by probability rather than
// trusting input order. One has no market (null) to check the "no live
// price yet" footer path.
const MOCK_PLAYER_TD_JS = `
const PLAYER_TD_DATA = {
  "generated_at": "2026-09-11T16:29:17Z",
  "model_notes": {"training_rows": 1, "caveat": "test"},
  "players": [
    ${JSON.stringify(player('Low Guy', 0.10, 'ARI', 'ARI vs SEA', 'WR', {anytime_td_price: 400, implied_prob: 0.2, edge: -0.1, ev: -0.2, source: 'live_api'}))},
    ${JSON.stringify(player('Jahmyr Gibbs', 0.64, 'DET', 'DET vs NO', 'RB', {anytime_td_price: -285, implied_prob: 0.7403, edge: -0.1003, ev: -0.08, source: 'live_api'}))},
    ${JSON.stringify(player('Derrick Henry', 0.61, 'BAL', 'BAL vs IND', 'RB', {anytime_td_price: -180, implied_prob: 0.6429, edge: -0.0329, ev: -0.05, source: 'live_api'}))},
    ${JSON.stringify(player('No Price Guy', 0.59, 'LAC', 'LAC vs ARI', 'RB', null))},
    ${JSON.stringify(player('Mid Guy A', 0.50, 'SF', 'SF vs LA', 'RB', {anytime_td_price: 150, implied_prob: 0.4, edge: 0.1, ev: 0.2, source: 'live_api'}))},
    ${JSON.stringify(player('Mid Guy B', 0.45, 'CIN', 'CIN vs TB', 'RB', {anytime_td_price: 130, implied_prob: 0.43, edge: 0.02, ev: 0.05, source: 'live_api'}))},
    ${JSON.stringify(player('Mid Guy C', 0.40, 'IND', 'IND vs BAL', 'RB', {anytime_td_price: 120, implied_prob: 0.45, edge: -0.05, ev: -0.1, source: 'live_api'}))},
    ${JSON.stringify(player('Very Low Guy', 0.05, 'WAS', 'WAS vs PHI', 'WR', {anytime_td_price: 900, implied_prob: 0.1, edge: -0.05, ev: -0.1, source: 'live_api'}))}
  ]
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
  doc.querySelector('[data-view="props"]').dispatchEvent(new dom.window.MouseEvent('click', {bubbles: true}));

  const checks = [];
  const cards = [...doc.querySelectorAll('#td-hero-cards .hero-card')];

  checks.push(['exactly 6 cards shown (top 6 of 8 players)', cards.length, 6]);

  const names = cards.map(c => c.querySelector('.hero-card-name').textContent);
  checks.push(['sorted descending by probability (Gibbs 64% first)', names[0], 'Jahmyr Gibbs']);
  checks.push(['second highest is Henry (61%)', names[1], 'Derrick Henry']);
  checks.push(['lowest-probability players (10%, 5%) correctly excluded from top 6',
    names.includes('Low Guy') || names.includes('Very Low Guy'), false]);

  const gibbsCard = cards.find(c => c.querySelector('.hero-card-name').textContent === 'Jahmyr Gibbs');
  checks.push(['Gibbs shows 64% (rounded from 0.64)', gibbsCard.querySelector('.hero-card-pct').textContent, '64']);
  checks.push(['Gibbs progress bar width matches 64%', gibbsCard.querySelector('.hero-card-bar').getAttribute('style').includes('64%'), true]);
  checks.push(['Gibbs footer shows real market odds and edge',
    gibbsCard.querySelector('.hero-card-foot').textContent.includes('-285') &&
    gibbsCard.querySelector('.hero-card-foot').textContent.includes('-10.0%'), true]);
  checks.push(['Gibbs subtitle shows position and matchup', gibbsCard.querySelector('.hero-card-sub').textContent.includes('RB'), true]);

  const noPriceCard = cards.find(c => c.querySelector('.hero-card-name').textContent === 'No Price Guy');
  checks.push(['unpriced player still makes the top 6 (59% is high enough)', !!noPriceCard, true]);
  checks.push(['unpriced player shows the graceful "no live price yet" footer',
    noPriceCard.querySelector('.hero-card-foot').textContent.trim(), 'no live price yet']);

  // Team color actually applied (not the default placeholder line color)
  const borderColor = gibbsCard.getAttribute('style');
  checks.push(['card border uses a real team color (not the default gray)',
    borderColor.includes('border-top-color') && !borderColor.includes('#3E7BFA'), true]);

  console.log(`${'check'.padEnd(75)} got                want               ok`);
  console.log('-'.repeat(115));
  let allPass = true;
  for (const [name, got, want] of checks) {
    const ok = got === want;
    allPass = allPass && ok;
    console.log(`${name.padEnd(75)} ${String(got).padEnd(18)} ${String(want).padEnd(18)} ${ok ? 'PASS' : 'FAIL'}`);
  }
  console.log('-'.repeat(115));
  console.log(allPass ? 'ALL PASS' : 'SOME FAILED');
  server.close();
  dom.window.close();
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); if (server) server.close(); process.exit(1); });
