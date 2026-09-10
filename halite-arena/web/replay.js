// Canvas replay viewer — loads a real arena replay (plain JSON) from the API
// and animates map + ships + scoreboard turn by turn. No external deps.
const QS = new URLSearchParams(location.search);
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const CELL = 16; // canvas pixel per map cell (640 / 40)
const PCOLORS = ['#4db3ff', '#ff6b6b', '#7bf57b', '#ffd93b', '#c98bff', '#ff9de2', '#5ff0d8', '#f5a65b'];

let state = { replay: null, pmap: null, frames: [], players: [], turn: 0, maxTurn: 0, playing: false, timer: null };

async function api(p) { const r = await fetch(p); const d = await r.json(); if (!r.ok) throw new Error(d.error || r.status); return d; }

function playerName(pid) {
  const p = state.players.find(p => String(p.player_id) === String(pid));
  return p ? p.name : ('P' + pid);
}
function playerColor(pid) {
  const p = state.players.find(p => String(p.player_id) === String(pid));
  return PCOLORS[(p ? p.player_id : pid) % PCOLORS.length];
}

// ---- loading --------------------------------------------------------------
async function fillRunSelect() {
  const sel = document.getElementById('sel-run');
  const runs = await api('/api/runs');
  sel.innerHTML = '';
  for (const r of runs) {
    const who = r.agent || (r.participants || []).join(' vs ') || r.id;
    sel.add(new Option(`${r.id}  [${r.kind}] ${who}`, r.id));
  }
  if (QS.get('run')) { sel.value = QS.get('run'); }
  if (sel.value) loadRun(sel.value);
}
async function loadRun(runId) {
  const run = await api('/api/run/' + runId);
  const gpick = document.getElementById('game-picker');
  gpick.innerHTML = '<label>对局 <select id="sel-game" onchange="loadGame()"></select></label>';
  const sg = document.getElementById('sel-game');
  sg.innerHTML = '';
  for (const g of run.games || []) {
    const who = g.winner ? (g.winner + ' wins') : 'tie';
    const sc = (g.scores || []).join('-');
    sg.add(new Option(`${g._dir}  ${who}  (${sc})`, g._dir));
  }
  if (sg.options.length) loadGame();
}
async function loadGame() {
  const runId = document.getElementById('sel-run').value;
  const game = document.getElementById('sel-game').value;
  const replay = await api(`/api/replay/${runId}/${game}`);
  initReplay(replay);
}

// ---- render core ----------------------------------------------------------
function initReplay(replay) {
  state.replay = replay;
  state.players = replay.players || [];
  state.pmap = (replay.production_map || {}).grid || [];
  state.frames = replay.full_frames || [];
  state.maxTurn = state.frames.length - 1;
  state.turn = Math.min(QS.get('t') || 1, state.maxTurn);
  // canvas size from map
  const h = state.pmap.length || 32, w = (state.pmap[0] || []).length || 32;
  const side = Math.max(w, h);
  cv.width = side * CELL; cv.height = side * CELL;
  document.getElementById('lbl-total').textContent = state.maxTurn;
  setTurn(state.turn);
}

function frameOf(t) { return state.frames[Math.min(Math.max(t, 0), state.maxTurn)] || null; }

function render() {
  const f = frameOf(state.turn);
  if (!f) return;
  ctx.clearRect(0, 0, cv.width, cv.height);
  drawMap(f);
  drawShips(f);
  drawScoreboard(f);
  document.getElementById('lbl-turn').textContent = state.turn;
}
function drawMap(f) {
  const pm = state.pmap;
  for (let y = 0; y < pm.length; y++) {
    const row = pm[y] || [];
    for (let x = 0; x < row.length; x++) {
      const en = (row[x] && row[x].energy) || 0;
      // brightness in dark-teal, halite-rich = brighter warm
      const b = Math.min(1, en / 700);
      ctx.fillStyle = `rgb(${Math.round(20 + b * 150)},${Math.round(30 + b * 60)},${Math.round(24 + b * 30)})`;
      ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
    }
  }
  // structure cells (dropoffs / shipyards from frame cells list)
  for (const c of f.cells || []) {
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.fillRect(c.x * CELL, c.y * CELL, CELL, CELL);
  }
  // factories from players[].factory_location
  for (const p of state.players) {
    const fl = p.factory_location; if (!fl) continue;
    ctx.fillStyle = playerColor(p.player_id);
    ctx.fillRect(fl.x * CELL + 2, fl.y * CELL + 2, CELL - 4, CELL - 4);
  }
}
function drawShips(f) {
  const ent = f.entities || {};
  for (const pid in ent) {
    const col = playerColor(pid);
    for (const sid in (ent[pid] || {})) {
      const s = ent[pid][sid];
      const x = (s.x || 0) * CELL, y = (s.y || 0) * CELL;
      ctx.beginPath();
      ctx.arc(x + CELL / 2, y + CELL / 2, CELL * 0.33, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke();
      // cargo pips: draw energy as small inner arc
      const en = s.energy || 0;
      if (en > 0) {
        ctx.beginPath();
        ctx.arc(x + CELL / 2, y + CELL / 2, CELL * 0.2, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, en / 1000));
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      }
    }
  }
}
function drawScoreboard(f) {
  const el = document.getElementById('scoreboard');
  const e = f.energy || {}, dep = f.deposited || {};
  const lines = state.players.map(p => {
    const pid = p.player_id;
    const bank = e[pid] ?? 0, depd = dep[pid] ?? 0, score = bank + depd;
    return `<span style="color:${playerColor(pid)}">■ ${playerName(pid)}</span>
      银行 ${bank} + 结构 ${depd} = <b>${score}</b>`;
  });
  el.innerHTML = `<div class="row">${lines.join('')}</div>
    <div class="hint">回合 ${state.turn} · events ${(f.events||[]).length}</div>`;
}
function setTurn(t) {
  state.turn = Math.min(Math.max(t, 0), state.maxTurn);
  render();
}
function togglePlay() {
  state.playing = !state.playing;
  document.getElementById('b-play').textContent = state.playing ? '⏸' : '▶';
  if (state.playing) tick();
}
function tick() {
  if (!state.playing) return;
  if (state.turn >= state.maxTurn) { state.playing = false; document.getElementById('b-play').textContent = '▶'; return; }
  setTurn(state.turn + 1);
  const delay = Math.max(5, 200 - (document.getElementById('speed').value * 2));
  state.timer = setTimeout(tick, delay);
}

// events
document.getElementById('b-play').onclick = togglePlay;
document.getElementById('b-next').onclick = () => { state.playing = false; document.getElementById('b-play').textContent = '▶'; setTurn(state.turn + 1); };
document.getElementById('b-prev').onclick = () => { state.playing = false; document.getElementById('b-play').textContent = '▶'; setTurn(state.turn - 1); };
document.getElementById('b-verify').onclick = async () => {
  const runId = document.getElementById('sel-run').value;
  const game = document.getElementById('sel-game').value;
  const out = document.getElementById('verify-out');
  out.textContent = '⏳ verifying…';
  try {
    const d = await api(`/api/verify/${runId}/${game}`);
    if (!d.ok && !(d.findings || []).length) { out.textContent = '❌ ' + (d.error || 'verify 失败'); return; }
    const marks = { ok: '✓', warn: '⚠', fail: '❌' };
    const lines = [`[${game}] ${d.ok ? '✅ ok' : '❌ FAIL'}`];
    for (const f of d.findings || []) lines.push(`  ${marks[f.level] || '·'} ${f.check}: ${f.detail || ''}`);
    const inf = d.info || {};
    if (inf.exec_ms != null) lines.push(`  · exec_ms=${inf.exec_ms}  turns=${inf.turns}  预算=${inf.turn_budget_s}s`);
    if (inf.costs) for (const [n, c] of Object.entries(inf.costs))
      lines.push(`  · 成本 ${n}: bot_log=${c.bot_log_bytes}B/${c.turns_logged}T errorlog=${c.errorlog_bytes}B`);
    if (inf.timeline) lines.push('  · 写入顺序: ' + inf.timeline.map(([, n]) => n).join(' → '));
    out.textContent = lines.join('\n');
    out.style.color = d.ok ? '' : '#ff6b6b';
  } catch (e) { out.textContent = '❌ ' + e.message; }
};

fillRunSelect().catch(e => alert('加载失败: ' + e.message));
