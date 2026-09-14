"use strict";

// ---------- constants (match the spec) ----------
const W = 800, H = 500;
const PADDLE_W = 12, PADDLE_H = 80, BALL_R = 8;
const PADDLE_SPEED = 7, BOT_SPEED = 4;
const LEFT_X = 20, RIGHT_X = W - 20 - PADDLE_W;   // paddle left edges
const SPEEDUP = 1.03, MAX_SPEED_FACTOR = 2;

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
  keys: new Set(),
};

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
  game.left.y = H / 2; game.right.y = H / 2;
  setFlyMove(0);
  serve(Math.random() < 0.5 ? -1 : 1);
  updateScores();
  if (typeof onNewGame === "function") onNewGame();
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
  if (hit(RIGHT_X, game.right.y, b.vx > 0)) { b.x = RIGHT_X - BALL_R; bounce(game.right.y, -1); }

  if (b.x < -BALL_R) { game.score.right += 1; updateScores(); serve(-1); }
  if (b.x > W + BALL_R) { game.score.left += 1; updateScores(); serve(1); }
}

function updateScores() {
  $("score-left").textContent = game.score.left;
  $("score-right").textContent = game.score.right;
}

function renderGame() {
  gctx.fillStyle = "#05060a"; gctx.fillRect(0, 0, W, H);
  gctx.strokeStyle = "#262b38"; gctx.setLineDash([8, 10]);
  gctx.beginPath(); gctx.moveTo(W / 2, 0); gctx.lineTo(W / 2, H); gctx.stroke(); gctx.setLineDash([]);
  gctx.fillStyle = "#e6e8ef";
  gctx.fillRect(LEFT_X, game.left.y - PADDLE_H / 2, PADDLE_W, PADDLE_H);
  gctx.fillStyle = "#f3c454";
  gctx.fillRect(RIGHT_X, game.right.y - PADDLE_H / 2, PADDLE_W, PADDLE_H);
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
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); togglePause(); return; }
  if (e.code === "KeyB") { $("bot").checked = !$("bot").checked; game.bot = $("bot").checked; return; }
  game.keys.add(e.code);
});
window.addEventListener("keyup", (e) => game.keys.delete(e.code));
$("bot").addEventListener("change", (e) => { game.bot = e.target.checked; });
$("new-game").addEventListener("click", newGame);
$("pause").addEventListener("click", togglePause);
function togglePause() {
  game.paused = !game.paused;
  $("pause").textContent = game.paused ? "Resume" : "Pause";
  if (typeof onPauseChange === "function") onPauseChange();
}
const bindSlider = (id, onChange) => {
  const el = $(id), out = $(id + "-v");
  const apply = () => { out.textContent = el.value; onChange(parseFloat(el.value)); };
  el.addEventListener("input", apply); apply();
};
bindSlider("ball-speed", (v) => { game.baseSpeed = v; });
bindSlider("steps", () => {}); bindSlider("loom", () => {}); bindSlider("retina", () => {}); bindSlider("gain", () => {});

// ---------- main loop ----------
function frame() {
  if (!game.paused) stepPhysics();
  renderGame();
  if (typeof renderBrain === "function") renderBrain();
  requestAnimationFrame(frame);
}
// newGame() and the loop start at the very end of the file, after every `let` below is initialized.

// =====================================================================
// Brain link: one state in flight at a time; the reply drives the fly.
// =====================================================================
const brainCanvas = $("brain"), bctx = brainCanvas.getContext("2d");
const BW = brainCanvas.width, BH = brainCanvas.height;
const CLASS_COLORS = ["#4a6fa5", "#7b6fd0", "#3aa6b9", "#e8a33d", "#d9764a", "#6bbf7a", "#e05d5d", "#8a8fa8", "#555a66"];
const FLASH_MS = 150;

let atlas = null;              // { n, x: Uint16Array, y: Uint16Array, cls: Uint8Array, base: canvas }
let flash = null;              // Float32Array brightness per neuron, 1 -> 0 over FLASH_MS
let active = [];               // indices with flash > 0
let lastFrameTime = performance.now();
let ws = null, inflight = false, ticks = 0, tickTimer = performance.now(), ticksPerSec = 0;

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
  c.globalAlpha = 0.35;
  for (let i = 0; i < n; i++) {
    if (x[i] === header.missing) continue;
    c.fillStyle = CLASS_COLORS[cls[i]] || "#555";
    c.fillRect(px(y[i]), py(x[i]), 1, 1);   // axis 0 vertical, axis 1 horizontal (same as renderBrain)
  }
  c.globalAlpha = 1;
  atlas = { n, x, y, cls, base, missing: header.missing };
  flash = new Float32Array(n);
}
// atlas axis 0 (long axis, brain to nerve cord) runs vertically; axis 1 horizontally.
function px(v1) { return 20 + (v1 / 65534) * (BW - 40); }
function py(v0) { return 20 + (v0 / 65534) * (BH - 40); }

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

function params() {
  return {
    steps_per_tick: parseFloat($("steps").value),
    loom_strength: parseFloat($("loom").value),
    retina_strength: parseFloat($("retina").value),
    motor_gain: parseFloat($("gain").value),
  };
}

function sendState() {
  if (!ws || ws.readyState !== WebSocket.OPEN || inflight || game.paused) return;
  inflight = true;
  ws.send(JSON.stringify({ type: "state", ...fieldState(), params: params() }));
}

function onCommand(cmd) {
  setFlyMove(cmd.move);
  if (flash) for (const i of cmd.spikes) { if (flash[i] <= 0) active.push(i); flash[i] = 1; }
  const l = cmd.dn.left, r = cmd.dn.right, top = Math.max(1, l, r);
  $("dn-left").textContent = l; $("dn-right").textContent = r;
  $("bar-left").style.width = `${(l / top) * 100}%`; $("bar-right").style.width = `${(r / top) * 100}%`;
  ticks += 1;
  const now = performance.now();
  if (now - tickTimer >= 1000) { ticksPerSec = ticks; ticks = 0; tickTimer = now; }
  $("stats").textContent = `move ${cmd.move.toFixed(2)} · ${ticksPerSec} ticks/s · ${cmd.stats.total_spikes} spikes · ${cmd.stats.wall_ms} ms/tick`;
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { $("status").textContent = "connected"; ws.send(JSON.stringify({ type: "reset" })); };
  ws.onclose = () => { $("status").textContent = "disconnected, retrying…"; inflight = false; setFlyMove(0); setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "hello") {
      $("status").textContent = `${msg.neurons.toLocaleString()} neurons on ${msg.device}`;
    } else if (msg.type === "command") {
      inflight = false; onCommand(msg); sendState();
    } else if (msg.type === "reset_ok") {
      inflight = false; sendState();
    } else if (msg.type === "error") {
      inflight = false; toast(msg.message); setFlyMove(0); sendState();
    }
  };
}

function onNewGame() {
  if (ws && ws.readyState === WebSocket.OPEN) { inflight = true; ws.send(JSON.stringify({ type: "reset" })); }
}
function onPauseChange() { if (!game.paused) sendState(); }

loadAtlas().catch((e) => toast(`atlas failed: ${e.message}`));
connect();
newGame();
requestAnimationFrame(frame);
