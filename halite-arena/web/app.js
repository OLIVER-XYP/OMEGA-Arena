// Shared logic for main view (/) and agent view (/agent.html?name=...).
const QS = new URLSearchParams(location.search);
const AGENT = QS.get('name') || null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  const txt = await r.text();
  let d; try { d = JSON.parse(txt); } catch { d = { error: txt.slice(0, 200) }; }
  if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
  return d;
}
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const shortId = s => s ? String(s).split('/').pop() : '';

// ---- main view ------------------------------------------------------------
async function mainView() {
  const [agents, runs, targets] = await Promise.all([
    api('/api/agents'), api('/api/runs'), api('/api/targets'),
  ]);
  // agent selector
  const sa = document.getElementById('sel-agent');
  sa.innerHTML = '';
  for (const a of agents) sa.add(new Option(`${a.name}  (elo ${a.elo ?? '—'})`, a.name));
  // target multiselect
  const st = document.getElementById('sel-targets');
  st.innerHTML = '';
  for (const s of targets.scripts) st.add(new Option(s, s));
  for (const r of targets.rl.slice(0, 15)) st.add(new Option(r, r));
  // agents table
  const tb = document.querySelector('#tbl-agents tbody');
  tb.innerHTML = '';
  for (const a of agents) {
    const fb = a.feedback;
    const fbTxt = fb ? (fb.kind === 'practice'
      ? Object.entries(fb.summary || {}).map(([o, s]) => `${o}:${s.win_rate}`).join(' ') || fb.kind
      : `${fb.kind}`) : '—';
    const mat = a.materials ? ['rules', 'engine', 'cpp'].filter(k => a.materials[k]).join('/') : '—';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a href="/agent.html?name=${esc(a.name)}">${esc(a.name)}</a></td>
      <td>${a.elo != null ? a.elo.toFixed(0) : '—'}</td>
      <td>${esc(mat)}</td><td>${esc(fbTxt)}</td>`;
    tb.appendChild(tr);
  }
  // runs table
  const tr2 = document.querySelector('#tbl-runs tbody');
  tr2.innerHTML = '';
  for (const r of runs) {
    const who = r.agent || (r.participants || []).join(', ');
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a href="/replay?run=${esc(r.id)}">${esc(shortId(r.id))}</a></td>
      <td>${esc(r.kind)}</td><td>${esc(who)}</td>
      <td class="${r.failed ? 'bad' : 'ok'}">${r.failed}</td>
      <td>${esc(JSON.stringify(r.ratings_after ?? '')?.slice(0, 160))}</td>`;
    tr2.appendChild(tr);
  }
  bindPractice(agents.map(a => a.name));
}

function bindPractice(agentNames) {
  const btn = document.getElementById('btn-practice');
  const status = document.getElementById('run-status');
  btn.onclick = async () => {
    const agent = document.getElementById('sel-agent').value;
    const targets = [...document.getElementById('sel-targets').selectedOptions].map(o => o.value);
    const seeds = parseInt(document.getElementById('inp-seeds').value, 10) || 2;
    if (!agent) { status.textContent = '先建 agent（arena init）'; return; }
    if (!targets.length) { status.textContent = '至少选一个对手'; return; }
    status.textContent = '⏳ 提交中…';
    try {
      const d = await api('/api/practice', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent, targets, seeds }) });
      if (!d.ok) { status.textContent = '❌ ' + (d.error || ''); return; }
      pollRun(status, () => { status.textContent = '✅ 完成'; refreshRuns(); });
    } catch (e) { status.textContent = '❌ ' + e.message; }
  };
}

async function refreshRuns() {
  const runs = await api('/api/runs');
  const tr = document.querySelector('#tbl-runs tbody');
  tr.innerHTML = '';
  for (const r of runs) {
    const who = r.agent || (r.participants || []).join(', ');
    const tr2 = document.createElement('tr');
    tr2.innerHTML = `<td><a href="/replay?run=${esc(r.id)}">${esc(shortId(r.id))}</a></td>
      <td>${esc(r.kind)}</td><td>${esc(who)}</td>
      <td class="${r.failed ? 'bad' : 'ok'}">${r.failed}</td>
      <td>${esc(JSON.stringify(r.ratings_after ?? ''))?.slice(0, 160)}</td>`;
    tr.appendChild(tr2);
  }
}

function pollRun(statusEl, done) {
  const tick = async () => {
    try {
      const d = await api('/api/run/latest');
      if (d.busy) { statusEl.textContent = '⏳ 对局运行中…（' + shortId(d.run_id) + '）'; setTimeout(tick, 1500); return; }
      if (d.error) { statusEl.textContent = '❌ ' + d.error; }
      else { statusEl.textContent = '✅ 完成: ' + shortId(d.run_id); done && done(); }
    } catch (e) { statusEl.textContent = '❌ ' + e.message; }
  };
  tick();
}

// ---- agent view ------------------------------------------------------------
async function agentView() {
  if (!AGENT) { document.getElementById('agent-name').textContent = '(需 ?name=xxx)'; return; }
  document.getElementById('agent-name').textContent = AGENT;
  const d = await api('/api/agent/' + AGENT);
  const fb = d.feedback;
  document.getElementById('fb-json').textContent = fb ? JSON.stringify(fb, null, 1) : '（尚无反馈，先跑 practice）';
  // materials: browse the readonly file tree
  const mb = document.getElementById('mat-buttons');
  mb.innerHTML = '';
  const matFiles = await api('/api/materials/' + AGENT).catch(() => []);
  // curated: rules.md on top, then cpp/ source, then engine_src
  const groups = [
    ['rules.md', f => f.path === 'rules.md'],
    ['cpp/ starter kit', f => f.path.startsWith('cpp/') && /\.(cpp|hpp|h|md|json|txt)$/.test(f.path)],
    ['engine_src/ 引擎源码', f => f.path.startsWith('engine_src/') && /\.(cpp|hpp|h|json|md|sh)$/.test(f.path)],
  ];
  const sel = document.createElement('select');
  sel.id = 'mat-sel';
  sel.style.minWidth = '320px';
  let opts = [['rules.md', 'rules.md']];
  for (const [label, pred] of groups) {
    const head = document.createElement('optgroup');
    head.label = label;
    for (const f of matFiles.filter(pred).slice(0, 200)) {
      const o = document.createElement('option');
      o.value = '/materials/' + AGENT + '/' + f.path;
      o.textContent = f.path;
      head.appendChild(o);
    }
    sel.appendChild(head);
  }
  const btn = document.createElement('button');
  btn.className = 'secondary';
  btn.textContent = '打开';
  btn.onclick = () => { const v = sel.value; if (v) document.getElementById('mat-frame').src = v; };
  mb.appendChild(sel);
  mb.appendChild(btn);
  const openRules = document.createElement('button');
  openRules.className = 'secondary';
  openRules.textContent = '查看 rules.md';
  openRules.onclick = () => { document.getElementById('mat-frame').src = '/materials/' + AGENT + '/rules.md'; };
  mb.appendChild(openRules);
  if (sel.options.length > 1) { sel.selectedIndex = 1; btn.click(); }
  // practice form
  const st = document.getElementById('sel-opps');
  const tg = await api('/api/targets');
  st.innerHTML = '';
  for (const s of tg.scripts) st.add(new Option(s, s));
  for (const r of tg.rl.slice(0, 10)) st.add(new Option(r, r));
  // history
  const tb = document.querySelector('#tbl-hist tbody');
  tb.innerHTML = '';
  for (const h of d.history || []) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a href="/replay?run=${esc(shortId(h.run_id))}">${esc(shortId(h.run_id))}</a></td>
      <td>${esc(JSON.stringify(h.summary ?? {}))}</td>`;
    tb.appendChild(tr);
  }
  document.getElementById('btn-compile').onclick = async () => {
    const s = document.getElementById('action-status');
    s.textContent = '编译中…';
    const r = await api('/api/compile', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: AGENT }) });
    s.textContent = r.ok ? '✅ 编译成功' : '❌ ' + (r.kind || 'fail');
    if (r.log) s.textContent += ' ' + r.log.split('\n')[0];
  };
  document.getElementById('btn-practice').onclick = async () => {
    const targets = [...st.selectedOptions].map(o => o.value);
    if (!targets.length) { document.getElementById('action-status').textContent = '选对手'; return; }
    const s = document.getElementById('action-status');
    s.textContent = '⏳ 提交…';
    const d = await api('/api/practice', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: AGENT, targets }) });
    if (!d.ok) { s.textContent = '❌ ' + (d.error || ''); return; }
    pollRun(s, () => { s.textContent = '✅ 完成'; agentView(); });
  };
}

document.addEventListener('DOMContentLoaded', () => {
  const isMain = document.getElementById('tbl-agents');
  if (isMain) mainView().catch(e => console.error(e));
  if (document.getElementById('fb-json')) agentView().catch(e => console.error(e));
});
