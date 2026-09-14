# Fly Pong: design

Date: 2026-09-14
Status: approved in conversation, awaiting written-spec review

## Goal

A Pong game where the right paddle is controlled by a live simulation of the
complete adult male fruit fly central nervous system (MaleCNS v1.0, 166,700
neurons, 25.6 million directed connections). The brain steps continuously
during play with state carried between frames. No training: the paddle is
driven by the fly's real looming-escape pathway. The human plays the left
paddle with the keyboard and can hand it to a scripted bot.

Runs entirely on the user's Mac (Apple M3 Pro, 18 GB, PyTorch MPS).

## Non-goals

- No learning or trained readout in this version. (Possible later: a small
  readout trained on live activity. Listed here so nobody adds it by accident.)
- No claim of biological fidelity beyond what the simulator provides. The
  simulator is a leaky integrate-and-fire model with engineering assumptions
  documented in the upstream repo.
- No deployment, accounts, or hosting. Localhost only.
- No 3D rendering. The brain view is a 2D projection.

## Upstream dependency

`/Users/k/Developer/mps-malecns-model` (GitHub seohyunjun/mps-malecns-model),
cloned 2026-09-14. Provides:

- `malecns.model.ConnectomeLIF`: the neuron model. `forward(drive)` takes a
  per-neuron voltage increment and returns a boolean spike vector. Buffers
  `voltage`, `spikes` live on the chosen device. `reset()` zeroes state.
- `data/graph.npz` (built, 338 MB): `body_ids`, `source`, `target`, `weight`,
  `cell_types`, `neurotransmitters`, `metadata`. Neuron order in this file is
  the canonical index order for everything in Fly Pong.
- `data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather`: per-neuron
  `somaSide` (L/R/M/?), `superclass`, `somaLocation` ([x,y,z] in 8 nm voxels,
  present for 139,662 of 166,700 neurons), `assignedOlHex1`/`assignedOlHex2`
  (eye column coordinates, present for 23,720 optic-lobe neurons including
  L1, L2, L3, Mi1, Mi4, Mi9, Tm9; range 1-36 by 1-39).

Fly Pong imports `malecns` as a package; it never copies or modifies the
upstream code. Data files are read from the sibling checkout by default and
the location is overridable with `FLYPONG_DATA=/path/to/mps-malecns-model`.

## Measured facts the design rests on

All measured 2026-09-14 on the target Mac with the full graph on MPS:

| Fact | Value |
|---|---|
| Simulation speed, full brain | 95 steps/s (repo smoke test), ~70 steps/s with per-step readback |
| Stimulating LC4 left-eye cells (0.3/step) | fires DNp02 and DNp04 on the LEFT only |
| Stimulating LC4 right-eye cells | fires DNp02 and DNp04 on the RIGHT only |
| Latency, stimulus onset to first descending spike | 7-8 steps |
| Sustained LC4 input for 100 steps | steady 2-5 descending spikes per 10 steps, no runaway, ~15-20 total spikes/step |
| Switching input side | output side flips within 7 steps, old side goes silent |
| No input (background drive 0.02 only) | zero spikes |
| Lamina/medulla patch stimulus (radius 6 columns) | lights local L1/L2/Tm/T4 cells; never reaches LC4, LPLC2, or descending neurons within 60 steps |
| Whole-eye T4/T5 stimulus | reaches LPLC2 (94 cells) and 7 descending spikes |

Consequence: descending output is cleanly lateralized, so "ball above" vs
"ball below" is distinguishable from anatomy alone. A pure retina image does
not propagate to the escape pathway in this model, so the ball must also
drive the looming detectors directly.

## Section 1: how the fly sees and moves

### Frame of reference

The fly sits at the right paddle facing left, viewed from above. Its left eye
covers the upper half of the screen, its right eye the lower half. "Above the
paddle" means the ball is on the fly's left; "below" means on its right.

### Sensory encoding (game state to drive vector)

Inputs per frame: ball center (x, y) and velocity (vx, vy), fly paddle center
y, field width W and height H.

Derived quantities:

- `dy = (ball.y - paddle.y) / (H / 2)`, clipped to [-1, 1]. Negative means
  above the paddle (left eye).
- `eye = L if dy < 0 else R`.
- `proximity = 1 - clip((paddle.x - ball.x) / W, 0, 1)`. 1 when the ball is
  at the paddle, 0 at the far wall.
- `approaching = vx > 0` (ball moving toward the fly).

Drive vector (float32, one value per neuron, in graph order):

1. Background: 0.02 on every neuron (upstream default).
2. Looming: neurons of type LC4 and LPLC2 with `somaSide == eye` receive
   `loom_strength * proximity * |dy| * (1 if approaching else 0.25)`.
   Default `loom_strength = 0.3`. When the ball is level with the paddle the
   term vanishes, giving a natural dead zone.
3. Retina image: lamina neurons (types L1, L2, L3) with `somaSide == eye`
   whose eye column lies within `ball_radius_columns` (default 2, Manhattan
   distance on the hex axes) of the ball's column receive `retina_strength`
   (default 0.3). The ball's column is
   `hex1 = lerp(hex1_min, hex1_max, 1 - proximity)` and
   `hex2 = lerp(hex2_min, hex2_max, |dy|)`, where the min/max are computed
   per eye from the data at load time and rounded to the nearest integer.
   Which hex axis is anterior-posterior in the real eye is not documented
   in the annotation file; this mapping is a display convention. Its purpose
   is that the retina and local motion detectors fire where the ball is,
   visibly, in the brain view. It does not contribute to the motor output in
   this model (measured above).

The looming shortcut is the one place the design computes something the
fly's own optic lobe should compute. It is stated plainly in the UI.

### Brain stepping

- One `FlyBrain.tick(drive, k)` call steps the model `k` times with the same
  drive. Default `k = 4` (4 brain milliseconds per game frame). Adjustable
  1-16 from the page.
- State is never reset between ticks during a game. `reset` happens on
  "New game" only.
- During a tick the brain records which neurons fired at least once
  (`fired_any |= spikes` each step) and sums spikes of the left and right
  descending-neuron sets.
- At 95 steps/s, k=4 gives ~20 ticks/s and a reaction latency of about
  2 ticks. Game rendering is decoupled (below), so this is perceived as
  reaction time, not frame rate.

### Motor readout (spikes to paddle command)

- `DN_L` = neurons with `superclass == descending_neuron` and `somaSide == L`;
  `DN_R` likewise for R. Neurons with side M or ? are ignored.
- Per tick: `delta = spikes(DN_R) - spikes(DN_L)`.
- Leaky integrator across ticks: `a = decay * a + delta`, default
  `decay = 0.7`.
- Command: `move = clip(a * motor_gain, -1, 1)`, default `motor_gain = 0.5`.
  Negative moves the paddle up, positive down. Ball above the paddle drives
  the left eye, which fires DN_L, which makes `delta` negative, which moves
  the paddle up toward the ball. Ball below is the mirror image.
- The client multiplies `move` by its max paddle speed.

### Game rules

- Field 800 by 500 logical units, paddles 12 by 80, ball radius 8.
- Left paddle: human (arrow keys or W/S) or bot. Right paddle: fly.
- Bot: tracks ball y at a capped speed (default 4 units/frame) while the ball
  moves toward it, otherwise drifts to center. Toggle with a checkbox or the
  B key. Switching mid-rally is allowed.
- Ball speed default 5 units/frame, slider 2-10. Ball angle changes with
  where it hits the paddle. Speed increases 3% per paddle hit, capped at 2x
  the base speed, and resets on each serve.
- A point is scored when the ball passes a paddle. Score shown; "New game"
  resets score and brain. No win condition; play until bored.
- Pause with the space bar or a button. Paused means the client stops sending
  state; the brain idles.

## Section 2: what gets built

### Project layout

```
/Users/k/Developer/fly-pong/
  pyproject.toml            uv-managed; Python >= 3.12
  README.md                 setup, run, credits, honest caveats
  flypong/
    __init__.py
    __main__.py             `python -m flypong` starts the server
    config.py               data paths (FLYPONG_DATA override), defaults table
    atlas.py                build static/atlas.bin from annotations
    senses.py               SensoryMap: index sets + drive(state, params)
    motor.py                MotorReadout: integrate(dn_left, dn_right) -> move
    brain.py                FlyBrain: load graph, tick(drive, k) -> TickResult
    server.py               aiohttp app: static files + /ws protocol
  static/
    index.html, app.js, style.css, atlas.bin (generated, gitignored)
  tests/
    test_senses.py, test_motor.py, test_brain.py, test_server.py, test_atlas.py
  docs/superpowers/specs/2026-09-14-fly-pong-design.md
```

Dependencies: `torch>=2.14,<2.15`, `numpy>=2.2,<3`, `pyarrow>=19,<26`,
`aiohttp>=3.10`, `malecns-mps @ file:///Users/k/Developer/mps-malecns-model`,
and `pytest` for tests. Managed with `uv`; run with `uv run python -m flypong`.

### Components

**config.py.** Resolves `DATA_ROOT` (env `FLYPONG_DATA`, else the sibling
checkout) and exposes `GRAPH_PATH`, `ANNOTATIONS_PATH`. Holds the defaults
table (`loom_strength`, `retina_strength`, `ball_radius_columns`,
`steps_per_tick`, `motor_decay`, `motor_gain`, `background_drive`) with
min/max bounds used to validate client parameters.

**atlas.py.** Reads the annotation file, joins it to graph order by body ID,
projects `somaLocation` onto its two principal axes (PCA over the neurons that
have a location), rescales to 0-65535, and writes `static/atlas.bin`:
a header `{"neurons": N, "classes": [...]}` as a little-endian uint32 length
followed by UTF-8 JSON padded with spaces to a multiple of 4 bytes, then three
planar arrays: `x` as uint16[N], `y` as uint16[N], `class_code` as uint8[N];
neurons without a location get `x = y = 65535`. Both axes are scaled by the
same factor (the larger span) so the brain's aspect ratio is preserved. Class codes index a fixed list of superclass
groups: optic lobe, central brain, visual projection, descending, ascending,
sensory, motor, VNC, other. Invoked once by `python -m flypong.atlas`; the
server refuses to start without the file and says how to build it.

**senses.py.** `SensoryMap(body_ids, cell_types, annotations)` builds at
load time: boolean masks for LC4+LPLC2 by eye, lamina neuron indices with
their hex coordinates by eye, per-eye hex extents, DN_L and DN_R index
arrays. `drive(state, params) -> np.ndarray[float32]` implements Section 1.
Pure numpy, no torch, fully unit-testable with a synthetic annotation table.

**motor.py.** `MotorReadout(decay, gain)` with `update(dn_left, dn_right)
-> float` and `reset()`. Pure Python.

**brain.py.** `FlyBrain(graph_path, device)` wraps `ConnectomeLIF`.
`tick(drive: np.ndarray, k: int) -> TickResult(fired_indices: np.ndarray,
dn_left: int, dn_right: int, total_spikes: int, wall_ms: float)` given the
DN index tensors set once via `set_readout(dn_left_idx, dn_right_idx)`.
Keeps a persistent `fired_any` buffer on the device, ORs each step's spikes
into it, then one `nonzero` and one transfer per tick. `reset()` forwards to
the model. Device resolution: MPS if available, else CPU with a printed
warning. Testable on CPU with a tiny hand-built graph.

**server.py.** aiohttp application. `GET /` and `/static/*` serve the page.
`GET /ws` is the game socket. On connect the server sends
`{"type":"hello","neurons":N,"device":"mps","defaults":{...}}`. The server
holds one `FlyBrain` and, per connection, a `MotorReadout`. Concurrent
connections are allowed but share the brain; the second tab's ticks
interleave with the first. This is accepted for a local toy. Brain ticks run
in a worker thread via `loop.run_in_executor` so the event loop stays
responsive; a lock serializes ticks.

Protocol (JSON text frames):

- Client `{"type":"state","ball":{"x","y","vx","vy"},"paddle":{"y","x"},
  "field":{"w","h"},"params":{...}}` where `params.steps_per_tick` is k →
  server replies
  `{"type":"command","move":m,"dn":{"left":a,"right":b},"spikes":[i,...],
  "stats":{"total_spikes":t,"wall_ms":w,"brain_ms":k}}`.
- Client `{"type":"reset"}` → server resets brain and readout, replies
  `{"type":"reset_ok"}`.
- Unknown types or malformed state → `{"type":"error","message":...}` and
  the connection stays open. Parameters outside config bounds are clamped
  and the clamped values echoed in `stats.params`.
- The client sends at most one `state` at a time and sends the next only
  after the `command` arrives. This is the rate limiter.

**static/app.js.** One `requestAnimationFrame` loop at display rate runs
physics, input, bot, rendering. The fly paddle's velocity is
`latestMove * paddleSpeed`. A separate async loop does send-state /
await-command / repeat. The brain panel draws the atlas once to an offscreen
canvas (dim dots colored by class) and, on each command, flashes the
`spikes` indices bright with a 150 ms fade. HUD shows ticks/s, DN left and
right counts as two bars, total spikes, and the current `move`. Controls:
bot toggle, brain steps per frame, ball speed, looming strength, retina
strength, motor gain, pause, new game. A short caption under the brain panel
states what is real and what is the shortcut.

### Error handling

- Missing graph or annotations: server exits with the exact upstream
  commands to run (`malecns download`, `malecns prepare`).
- Missing atlas: exits naming `python -m flypong.atlas`.
- MPS unavailable: warning, CPU fallback, `hello.device` says `cpu`.
- Nonfinite voltages after a tick (can happen with extreme parameters):
  server resets the brain, replies `{"type":"error"}` with a message, and
  the client shows a toast and continues.
- Socket closed mid-tick: result discarded, brain state kept.
- Client reconnect: sends `reset`, starts fresh.

### Testing

Test-driven, pytest.

- `test_senses.py`: synthetic annotation table with a handful of LC4, LPLC2,
  L1, DN neurons on each side. Assert: ball above → left-eye looming cells
  driven and right-eye cells at background; ball below → mirrored; ball level
  → no looming drive; proximity scales the drive; retina drive lands on the
  expected columns and stays within each eye's extents at the field edges;
  drive vector length equals neuron count and every value is finite.
- `test_motor.py`: left spikes give negative move, right spikes positive,
  decay shrinks the command over silent ticks, clipping at ±1, reset zeroes.
- `test_brain.py`: 6-neuron chain graph on CPU. Assert a driven neuron
  spikes, its target spikes one step later, `fired_indices` is the union over
  k steps, DN counts match, `reset` clears state, tick with k=0 rejected.
- `test_atlas.py`: build from a synthetic table into a temp dir; parse
  header and records back; neurons without location are 65535.
- `test_server.py`: aiohttp test client with a `FakeBrain` returning fixed
  results. Assert hello on connect, state → command shape, parameter
  clamping, reset, malformed message → error without disconnect.
- Manual: `uv run python -m flypong`, open the page, play, toggle bot, watch
  the brain panel. Playwright screenshot for the README.

### Credits and licensing

MaleCNS data is CC BY 4.0, credited to FlyEM / HHMI Janelia, University of
Cambridge, MRC LMB, and Google Research. The simulator is
seohyunjun/mps-malecns-model, used as an unmodified dependency. Fly Pong's
own code is MIT.
