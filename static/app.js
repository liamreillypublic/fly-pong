"use strict";

// ---------- constants (match the spec) ----------
const W = 800, H = 500;
const PADDLE_W = 12, PADDLE_H = 80, BALL_R = 8;
const PADDLE_SPEED = 7, BOT_SPEED = 4;
const LEFT_X = 20, RIGHT_X = W - 20 - PADDLE_W;   // paddle left edges
const SPEEDUP = 1.03, MAX_SPEED_FACTOR = 2;
const RECENT_BALLS = 20;
const SLIDER_DEFAULTS = { "ball-speed": 5, steps: 4, loom: 0.15, gain: 0.5, lrate: 0.005,
                          noise: 0.5, depression: 0.2, light: 0.05, contrast: 1, mb: 0.1 };

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const gameCanvas = $("game"), gctx = gameCanvas.getContext("2d");

// ---------- game state ----------
const game = {
  ball: { x: W / 2, y: H / 2, vx: 0, vy: 0 },
  left: { y: H / 2 }, right: { y: H / 2 },        // paddle centers
  score: { left: 0, right: 0 },
  paused: false, bot: false, baseSpeed: 5, speed: 5,
  flyMove: 0,                                       // latest brain command in [-1, 1]
  flyReturns: 0, flyMisses: 0,
  recent: [],                                       // 1 = returned, 0 = missed, last RECENT_BALLS
  pendingEvents: [],                                // "return" / "miss" since the last state message
  keys: new Set(),
};
let autoPaused = false;

function setFlyMove(m) { game.flyMove = Math.max(-1, Math.min(1, m)); }

function serve(direction) {
  const angle = (Math.random() * 60 - 30) * Math.PI / 180;   // ±30°
  game.speed = game.baseSpeed;
  game.ball.x = W / 2; game.ball.y = H / 2;
  game.ball.vx = Math.cos(angle) * game.speed * direction;
  game.ball.vy = Math.sin(angle) * game.speed;
}

function newGame() {
  game.score.left = 0; game.score.right = 0;
  game.flyReturns = 0; game.flyMisses = 0; game.recent = [];
  game.left.y = H / 2; game.right.y = H / 2;
  setFlyMove(0);
  serve(Math.random() < 0.5 ? -1 : 1);
  updateScores(); updateFlyStat();
  if (typeof onNewGame === "function") onNewGame();
}

function recordBall(returned) {
  game.recent.push(returned ? 1 : 0);
  if (game.recent.length > RECENT_BALLS) game.recent.shift();
  game.pendingEvents.push(returned ? "return" : "miss");
}

function clampPaddle(y) { return Math.max(PADDLE_H / 2, Math.min(H - PADDLE_H / 2, y)); }

function stepPhysics() {
  // left paddle: human or bot
  if (game.bot) {
    const target = game.ball.vx < 0 ? game.ball.y : H / 2;
    const d = target - game.left.y;
    game.left.y += Math.max(-BOT_SPEED, Math.min(BOT_SPEED, d));
  } else {
    let dir = 0;
    if (game.keys.has("KeyW") || game.keys.has("ArrowUp")) dir -= 1;
    if (game.keys.has("KeyS") || game.keys.has("ArrowDown")) dir += 1;
    game.left.y += dir * PADDLE_SPEED;
  }
  game.left.y = clampPaddle(game.left.y);
  // right paddle: the fly
  game.right.y = clampPaddle(game.right.y + game.flyMove * PADDLE_SPEED);

  // ball
  const b = game.ball;
  b.x += b.vx; b.y += b.vy;
  if (b.y - BALL_R < 0) { b.y = BALL_R; b.vy = Math.abs(b.vy); }
  if (b.y + BALL_R > H) { b.y = H - BALL_R; b.vy = -Math.abs(b.vy); }

  const hit = (paddleLeftX, paddleY, movingToward) => {
    if (!movingToward) return false;
    const withinX = b.x + BALL_R >= paddleLeftX && b.x - BALL_R <= paddleLeftX + PADDLE_W;
    const withinY = Math.abs(b.y - paddleY) <= PADDLE_H / 2 + BALL_R;
    return withinX && withinY;
  };
  const bounce = (paddleY, direction) => {
    const offset = (b.y - paddleY) / (PADDLE_H / 2);            // -1..1
    const angle = offset * 60 * Math.PI / 180;
    game.speed = Math.min(game.speed * SPEEDUP, game.baseSpeed * MAX_SPEED_FACTOR);
    b.vx = Math.cos(angle) * game.speed * direction;
    b.vy = Math.sin(angle) * game.speed;
  };
  if (hit(LEFT_X, game.left.y, b.vx < 0)) { b.x = LEFT_X + PADDLE_W + BALL_R; bounce(game.left.y, 1); }
  if (hit(RIGHT_X, game.right.y, b.vx > 0)) {
    b.x = RIGHT_X - BALL_R; bounce(game.right.y, -1);
    game.flyReturns += 1; recordBall(true); updateFlyStat();
  }

  if (b.x < -BALL_R) { game.score.right += 1; updateScores(); serve(-1); }
  if (b.x > W + BALL_R) {
    game.score.left += 1; game.flyMisses += 1; recordBall(false);
    updateScores(); updateFlyStat(); serve(1);
  }
}

function updateScores() {
  $("score-left").textContent = game.score.left;
  $("score-right").textContent = game.score.right;
}

function recentRate() {
  return game.recent.length ? game.recent.reduce((a, b) => a + b, 0) / game.recent.length : null;
}

function updateFlyStat() {
  const faced = game.flyReturns + game.flyMisses;
  const pct = faced ? ` (${Math.round(100 * game.flyReturns / faced)}%)` : "";
  const r = recentRate();
  const recent = r === null ? "–" : `${Math.round(100 * r)}%`;
  $("fly-stat").textContent = `Fly returned ${game.flyReturns} of ${faced}${pct} · last ${game.recent.length || RECENT_BALLS}: ${recent}`;
}

function renderGame() {
  gctx.fillStyle = "#05060a"; gctx.fillRect(0, 0, W, H);
  gctx.strokeStyle = "#262b38"; gctx.setLineDash([8, 10]);
  gctx.beginPath(); gctx.moveTo(W / 2, 0); gctx.lineTo(W / 2, H); gctx.stroke(); gctx.setLineDash([]);
  gctx.fillStyle = "#e6e8ef";
  gctx.fillRect(LEFT_X, game.left.y - PADDLE_H / 2, PADDLE_W, PADDLE_H);
  gctx.fillStyle = "#f3c454";
  gctx.fillRect(RIGHT_X, game.right.y - PADDLE_H / 2, PADDLE_W, PADDLE_H);
  // the fly's current command, drawn as a small arrow beside its paddle
  if (Math.abs(game.flyMove) > 0.05) {
    const dir = Math.sign(game.flyMove), len = 10 + 16 * Math.abs(game.flyMove);
    const x = RIGHT_X - 14, y0 = game.right.y, y1 = y0 + dir * len;
    gctx.strokeStyle = "#f3c454"; gctx.lineWidth = 2;
    gctx.beginPath(); gctx.moveTo(x, y0); gctx.lineTo(x, y1);
    gctx.moveTo(x - 4, y1 - dir * 5); gctx.lineTo(x, y1); gctx.lineTo(x + 4, y1 - dir * 5); gctx.stroke();
    gctx.lineWidth = 1;
  }
  gctx.fillStyle = "#ffffff";
  gctx.beginPath(); gctx.arc(game.ball.x, game.ball.y, BALL_R, 0, Math.PI * 2); gctx.fill();
  if (game.paused) {
    gctx.fillStyle = "rgba(0,0,0,0.5)"; gctx.fillRect(0, 0, W, H);
    gctx.fillStyle = "#e6e8ef"; gctx.font = "28px system-ui"; gctx.textAlign = "center";
    gctx.fillText("Paused", W / 2, H / 2);
  }
}

// state payload for the brain server
function fieldState() {
  return {
    ball: { x: game.ball.x, y: game.ball.y, vx: game.ball.vx, vy: game.ball.vy },
    paddle: { x: RIGHT_X, y: game.right.y },
    field: { w: W, h: H },
  };
}

// ---------- input and controls ----------
const GAME_KEYS = new Set(["ArrowUp", "ArrowDown", "Space", "KeyW", "KeyS", "KeyB", "KeyN"]);
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type === "range") {
    // a focused slider must not swallow game keys or move together with the paddle
    e.target.blur();
  }
  if (e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA" ||
      (e.target.tagName === "INPUT" && e.target.type === "text")) return;
  // Arrow keys and Space scroll the page by default; the page is much taller
  // than the window, so the game would scroll out of view while playing.
  if (GAME_KEYS.has(e.code)) e.preventDefault();
  if (e.code === "Space") { togglePause(); return; }
  if (e.code === "KeyB") { setBot(!game.bot); return; }
  if (e.code === "KeyN") { newGame(); return; }
  game.keys.add(e.code);
});
window.addEventListener("keyup", (e) => game.keys.delete(e.code));
function setBot(on) {
  game.bot = on; $("bot").checked = on;
  $("who-left").textContent = on ? "Bot" : "You";
}
$("bot").addEventListener("change", (e) => setBot(e.target.checked));
$("new-game").addEventListener("click", newGame);
$("pause").addEventListener("click", togglePause);
function togglePause() {
  game.paused = !game.paused;
  $("pause").textContent = game.paused ? "Resume (Space)" : "Pause (Space)";
  if (typeof onPauseChange === "function") onPauseChange();
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { if (!game.paused) { togglePause(); autoPaused = true; } }
  else if (autoPaused) { autoPaused = false; if (game.paused) togglePause(); }
});

const setSlider = (id, value) => {
  const el = $(id); el.value = value; el.dispatchEvent(new Event("input"));
};
const bindSlider = (id, onChange) => {
  const el = $(id), out = $(id + "-v");
  const apply = () => { out.textContent = el.value; onChange(parseFloat(el.value)); };
  el.addEventListener("input", apply); apply();
};
bindSlider("ball-speed", (v) => { game.baseSpeed = v; });
for (const id of ["steps", "loom", "gain", "lrate", "noise", "depression", "light", "contrast", "mb"]) bindSlider(id, () => {});

for (const btn of document.querySelectorAll(".preset")) {
  btn.addEventListener("click", () => {
    setSlider("ball-speed", btn.dataset.ball); setSlider("steps", btn.dataset.steps);
    for (const other of document.querySelectorAll(".preset")) other.classList.toggle("active", other === btn);
  });
}
$("reset-sliders").addEventListener("click", () => {
  for (const [id, v] of Object.entries(SLIDER_DEFAULTS)) setSlider(id, v);
  $("dt").value = "1"; $("shortcut").checked = true; $("readout-motor").checked = false; $("punish").checked = false;
  $("normalize").checked = true;
  for (const other of document.querySelectorAll(".preset")) other.classList.remove("active");
});
for (const el of document.querySelectorAll("input[type=range]")) el.addEventListener("input", () => {
  // any manual change clears the preset highlight
  for (const other of document.querySelectorAll(".preset")) other.classList.remove("active");
});

// ---------- main loop ----------
// Physics runs at a fixed 60 steps per second regardless of the display's
// refresh rate (120 Hz screens must not double the ball speed).
const PHYSICS_STEP_MS = 1000 / 60;
let physicsAccumulator = 0, lastPhysicsTime = null, physicsSteps = 0;
function frame(now) {
  if (lastPhysicsTime === null) lastPhysicsTime = now;
  physicsAccumulator += Math.min(now - lastPhysicsTime, 100);   // cap catch-up after a stall
  lastPhysicsTime = now;
  while (physicsAccumulator >= PHYSICS_STEP_MS) {
    physicsAccumulator -= PHYSICS_STEP_MS;
    if (!game.paused) { stepPhysics(); physicsSteps += 1; }
  }
  renderGame();
  if (typeof renderBrain === "function") renderBrain();
  if (typeof renderChart === "function") renderChart();
  if (typeof renderLearnChart === "function") renderLearnChart();
  requestAnimationFrame(frame);
}
// newGame() and the loop start at the very end of the file, after every `let` below is initialized.

// =====================================================================
// Brain link: one state in flight at a time; the reply drives the fly.
// =====================================================================
const brainCanvas = $("brain"), bctx = brainCanvas.getContext("2d");
const BW = brainCanvas.width, BH = brainCanvas.height;
const chartCanvas = $("chart"), cctx = chartCanvas.getContext("2d");
const CW = chartCanvas.width, CH = chartCanvas.height;
const learnCanvas = $("learn-chart"), lctx = learnCanvas.getContext("2d");
const LW = learnCanvas.width, LH = learnCanvas.height;
const CLASS_COLORS = ["#4a6fa5", "#7b6fd0", "#3aa6b9", "#e8a33d", "#d9764a", "#6bbf7a", "#e05d5d", "#8a8fa8", "#555a66"];
const FLASH_MS = 150;
const CHART_TICKS = 220;   // about 10 s at 22 ticks/s
const LEARN_EVENTS = 120;

let atlas = null;              // { n, x: Uint16Array, y: Uint16Array, cls: Uint8Array, base: canvas }
let flash = null;              // Float32Array brightness per neuron, 1 -> 0 over FLASH_MS
let active = [];               // indices with flash > 0
let legendCounts = [];         // <span> per class
let history = [];              // { l, r, m } per tick, newest last
let learnHistory = [];         // { event, rate, rpe } per reward/punishment, newest last
let modelInfo = null;
let lastFrameTime = performance.now();
// `pending` counts requests awaiting a reply (state, reset or forget). A new
// state is sent only when it is 0, so New game during a rally can never start
// a second request loop.
let ws = null, pending = 0, ticks = 0, tickTimer = performance.now(), ticksPerSec = 0;

function toast(message, ms = 3000) {
  const el = $("toast"); el.textContent = message; el.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { el.hidden = true; }, ms);
}

async function loadAtlas() {
  const buf = await (await fetch("/static/atlas.bin")).arrayBuffer();
  const view = new DataView(buf);
  const hlen = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  const n = header.neurons;
  let off = 4 + hlen;
  const x = new Uint16Array(buf, off, n); off += 2 * n;
  const y = new Uint16Array(buf, off, n); off += 2 * n;
  const cls = new Uint8Array(buf, off, n);
  const base = document.createElement("canvas"); base.width = BW; base.height = BH;
  const c = base.getContext("2d");
  c.fillStyle = "#05060a"; c.fillRect(0, 0, BW, BH);
  atlas = { n, x, y, cls, base, missing: header.missing, classes: header.classes };
  flash = new Float32Array(n);
  buildLegend(header.classes);
  // Paint the 166k base dots in chunks across animation frames so the game
  // never stalls; the brain visibly fills in over the first second.
  const CHUNK = 12000;
  let i = 0;
  const paint = () => {
    c.globalAlpha = 0.35;
    const end = Math.min(n, i + CHUNK);
    for (; i < end; i++) {
      if (x[i] === header.missing) continue;
      c.fillStyle = CLASS_COLORS[cls[i]] || "#555";
      c.fillRect(px(y[i]), py(x[i]), 1, 1);   // axis 0 vertical, axis 1 horizontal (same as renderBrain)
    }
    c.globalAlpha = 1;
    if (i < n) requestAnimationFrame(paint);
  };
  requestAnimationFrame(paint);
}
// atlas axis 0 (long axis, brain to nerve cord) runs vertically; axis 1 horizontally.
function px(v1) { return 20 + (v1 / 65534) * (BW - 40); }
function py(v0) { return 20 + (v0 / 65534) * (BH - 40); }

function buildLegend(classes) {
  const legend = $("legend"); legend.innerHTML = ""; legendCounts = [];
  classes.forEach((name, i) => {
    const row = document.createElement("div"); row.className = "row";
    const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = CLASS_COLORS[i] || "#555";
    const label = document.createElement("span"); label.textContent = name;
    const count = document.createElement("span"); count.className = "n"; count.textContent = "0";
    row.append(sw, label, count); legend.append(row); legendCounts.push(count);
  });
}

function renderBrain() {
  const now = performance.now(), dt = now - lastFrameTime; lastFrameTime = now;
  bctx.fillStyle = "#05060a"; bctx.fillRect(0, 0, BW, BH);
  if (!atlas) { bctx.fillStyle = "#8b90a0"; bctx.font = "16px system-ui"; bctx.fillText("loading atlas…", 20, 40); return; }
  bctx.drawImage(atlas.base, 0, 0);
  const keep = [];
  for (const i of active) {
    const b = flash[i];
    if (b <= 0) continue;
    if (atlas.x[i] !== atlas.missing) {
      bctx.fillStyle = `rgba(255, 244, 180, ${Math.min(1, b)})`;
      bctx.fillRect(px(atlas.y[i]) - 1, py(atlas.x[i]) - 1, 3, 3);
    }
    flash[i] = b - dt / FLASH_MS;
    if (flash[i] > 0) keep.push(i);
  }
  active = keep;
}

function renderChart() {
  cctx.fillStyle = "#05060a"; cctx.fillRect(0, 0, CW, CH);
  const mid = CH / 2, half = mid - 6, step = CW / CHART_TICKS;
  cctx.strokeStyle = "#262b38"; cctx.beginPath(); cctx.moveTo(0, mid); cctx.lineTo(CW, mid); cctx.stroke();
  if (!history.length) return;
  let peak = 3;
  for (const h of history) peak = Math.max(peak, h.l, h.r);
  const x0 = CW - history.length * step;
  for (let i = 0; i < history.length; i++) {
    const h = history[i], x = x0 + i * step;
    if (h.l) { cctx.fillStyle = "#f3c454"; cctx.fillRect(x, mid - (h.l / peak) * half, Math.max(1, step - 0.5), (h.l / peak) * half); }
    if (h.r) { cctx.fillStyle = "#6bb2f0"; cctx.fillRect(x, mid, Math.max(1, step - 0.5), (h.r / peak) * half); }
  }
  cctx.strokeStyle = "rgba(255,255,255,0.85)"; cctx.lineWidth = 1.5; cctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = x0 + i * step + step / 2, y = mid + history[i].m * half;
    if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y);
  }
  cctx.stroke(); cctx.lineWidth = 1;
  cctx.fillStyle = "#8b90a0"; cctx.font = "11px system-ui"; cctx.textAlign = "left";
  cctx.fillText(`peak ${peak} spikes/tick`, 4, 12);
}

function renderLearnChart() {
  lctx.fillStyle = "#05060a"; lctx.fillRect(0, 0, LW, LH);
  const mid = LH / 2, half = mid - 6, step = LW / LEARN_EVENTS;
  lctx.strokeStyle = "#262b38"; lctx.beginPath(); lctx.moveTo(0, mid); lctx.lineTo(LW, mid); lctx.stroke();
  if (!learnHistory.length) {
    lctx.fillStyle = "#8b90a0"; lctx.font = "11px system-ui"; lctx.textAlign = "left";
    lctx.fillText("waiting for the first return or miss", 4, 12);
    return;
  }
  const x0 = LW - learnHistory.length * step, bar = Math.max(1, step - 1);
  for (let i = 0; i < learnHistory.length; i++) {
    const e = learnHistory[i], x = x0 + i * step, h = half * Math.max(0.08, Math.min(1, Math.abs(e.rpe)));
    if (e.rpe >= 0) { lctx.fillStyle = "#6bbf7a"; lctx.fillRect(x, mid - h, bar, h); }
    else { lctx.fillStyle = "#e05d5d"; lctx.fillRect(x, mid, bar, h); }
  }
  lctx.strokeStyle = "rgba(255,255,255,0.9)"; lctx.lineWidth = 1.5; lctx.beginPath();
  for (let i = 0; i < learnHistory.length; i++) {
    const x = x0 + i * step + step / 2, y = LH - 4 - learnHistory[i].rate * (LH - 8);
    if (i === 0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
  }
  lctx.stroke(); lctx.lineWidth = 1;
  const last = learnHistory[learnHistory.length - 1];
  lctx.fillStyle = "#8b90a0"; lctx.font = "11px system-ui"; lctx.textAlign = "left";
  lctx.fillText(`return rate ${Math.round(last.rate * 100)}%`, 4, 12);
}

function params() {
  return {
    steps_per_tick: parseFloat($("steps").value),
    dt_ms: parseFloat($("dt").value),
    noise_mv: parseFloat($("noise").value),
    depression_u: parseFloat($("depression").value),
    loom_strength: parseFloat($("loom").value),
    loom_shortcut: $("shortcut").checked ? 1 : 0,
    light: parseFloat($("light").value),
    ball_contrast: parseFloat($("contrast").value),
    mb_strength: parseFloat($("mb").value),
    readout_motor: $("readout-motor").checked ? 1 : 0,
    readout_normalize: $("normalize").checked ? 1 : 0,
    motor_gain: parseFloat($("gain").value),
    learning_enabled: $("learning").checked ? 1 : 0,
    learning_rate: parseFloat($("lrate").value),
    punish_reflex: $("punish").checked ? 1 : 0,
  };
}

function sendState() {
  if (!ws || ws.readyState !== WebSocket.OPEN || pending > 0 || game.paused) return;
  pending += 1;
  ws.send(JSON.stringify({ type: "state", ...fieldState(), events: game.pendingEvents.splice(0), params: params() }));
}
function replied() { pending = Math.max(0, pending - 1); }

function fmtAge(s) {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  return m ? `${m}m ${String(sec).padStart(2, "0")}s` : `${sec}s`;
}

function onLearning(L) {
  $("dopamine").textContent = `PAM ${L.pam} · PPL1 ${L.ppl1} · expects ${Math.round(100 * L.expected)}% returns`;
  const saved = L.saved_ago_s == null ? "not saved yet" : `saved ${Math.round(L.saved_ago_s)} s ago`;
  $("memory").textContent =
    `mushroom body: ${L.mb_changed.toLocaleString()} synapses changed, ${(100 * L.mb_drift).toFixed(1)}% avg · ` +
    `reflex pathway: ${L.reflex_changed.toLocaleString()} changed, ${(100 * L.reflex_drift).toFixed(1)}% avg · ` +
    `rewards ${L.rewards} · punishments ${L.punishments} · age ${fmtAge(L.age_s)} · ${saved}${L.enabled ? "" : " · learning off"}`;
  if (L.event) {
    const badge = $("dopamine-badge");
    const sign = L.rpe >= 0 ? "+" : "";
    badge.textContent = `${L.event === "both" ? "reward + punishment" : L.event} (${sign}${L.rpe.toFixed(2)})`;
    badge.className = `badge ${L.event === "both" ? "reward" : L.event}`;
    clearTimeout(onLearning.timer);
    onLearning.timer = setTimeout(() => { badge.textContent = ""; badge.className = "badge"; }, 900);
    learnHistory.push({ event: L.event, rate: recentRate() ?? 0, rpe: L.rpe });
    if (learnHistory.length > LEARN_EVENTS) learnHistory.shift();
  }
}

function onCommand(cmd) {
  setFlyMove(cmd.move);
  if (flash) for (const i of cmd.spikes) { if (flash[i] <= 0) active.push(i); flash[i] = 1; }
  if (atlas && legendCounts.length) {
    const counts = new Array(legendCounts.length).fill(0);
    for (const i of cmd.spikes) counts[atlas.cls[i]] += 1;
    counts.forEach((v, c) => { legendCounts[c].textContent = v; });
  }
  const l = cmd.dn.left, r = cmd.dn.right, top = Math.max(1, l, r);
  $("dn-left").textContent = l; $("dn-right").textContent = r;
  $("bar-left").style.width = `${(l / top) * 100}%`; $("bar-right").style.width = `${(r / top) * 100}%`;
  history.push({ l, r, m: cmd.move });
  if (history.length > CHART_TICKS) history.shift();
  if (cmd.stats.learning) onLearning(cmd.stats.learning);
  ticks += 1;
  const now = performance.now();
  if (now - tickTimer >= 1000) { ticksPerSec = ticks; ticks = 0; tickTimer = now; }
  const s = cmd.stats, mon = s.monitors || {};
  $("stats").textContent = `move ${cmd.move.toFixed(2)} · ${ticksPerSec} ticks/s · ${s.brain_ms} brain ms in ${s.wall_ms} ms · ${s.spikes_per_step} spikes/step`;
  $("activity").textContent = `activity: motor neurons L/R ${cmd.mn.left}/${cmd.mn.right} · Kenyon cells ${mon.kc ?? 0} · MB output ${mon.mbon ?? 0} · LPLC2 ${mon.lplc2 ?? 0} · LC4 ${mon.lc4 ?? 0} · photoreceptors ${mon.photoreceptors ?? 0}`;
}

function statusText(msg) {
  let text = `${msg.neurons.toLocaleString()} neurons on ${msg.device}`;
  if (msg.model) {
    text += ` · ${msg.model.name} @ ${msg.model.dt_ms} ms · inhibition x${msg.model.inhibition_scale ?? 1}`;
    if (msg.model.steps_per_s) text += ` · idle ${msg.model.steps_per_s.toLocaleString()} steps/s`;
  }
  if (msg.learning) text += ` · ${msg.learning.plastic.toLocaleString()} plastic synapses · ${msg.learning.dopamine_cells} dopamine cells`;
  return text;
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { $("status").textContent = "connected"; pending = 1; ws.send(JSON.stringify({ type: "reset" })); };
  ws.onclose = () => { $("status").textContent = "disconnected, retrying…"; pending = 0; setFlyMove(0); setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "hello") {
      modelInfo = msg.model || null;
      $("status").textContent = statusText(msg);
    } else if (msg.type === "command") {
      replied(); onCommand(msg); sendState();
    } else if (msg.type === "reset_ok") {
      replied(); history = []; sendState();
    } else if (msg.type === "forget_ok") {
      replied(); learnHistory = []; toast("The fly forgot everything and is back to the original connectome."); sendState();
    } else if (msg.type === "error") {
      replied(); setFlyMove(0);
      if (msg.fatal) {
        // another tab took over the brain; stop driving until this page is reloaded
        $("status").textContent = "another tab is driving the fly · reload this page to take over";
        toast(msg.message, 8000);
        return;
      }
      toast(msg.message); sendState();
    }
  };
}

function onNewGame() {
  if (ws && ws.readyState === WebSocket.OPEN) { pending += 1; ws.send(JSON.stringify({ type: "reset" })); }
}
function onPauseChange() { if (!game.paused) sendState(); }

$("forget").addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!confirm("Erase everything the fly has learned and restore the original connectome?")) return;
  pending += 1;
  ws.send(JSON.stringify({ type: "forget" }));
});

loadAtlas().catch((e) => toast(`atlas failed: ${e.message}`));
connect();
newGame();
requestAnimationFrame(frame);
