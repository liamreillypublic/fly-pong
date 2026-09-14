# Realism Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full transmitter table, the published Shiu et al. neuron model with an event-driven kernel and spontaneous activity, a real eye with a dark ball, reward-prediction-error dopamine, mushroom-body input, motor-neuron readout, page controls, and a measured verdict.

**Architecture:** A project-owned CSR graph (`flypong/graph.py`) feeds a new `ShiuLIF` model (`flypong/neurons.py`) that replaces upstream `ConnectomeLIF` inside `FlyBrain`. Senses gain photoreceptor light, dark-ball, and mushroom-body terms. Plasticity gains prediction error and a configurable punishment path. Server and page expose the new parameters and stats.

**Tech Stack:** unchanged.

**Spec:** `docs/superpowers/specs/2026-09-14-realism-batch-design.md`

## Global Constraints

- All earlier constraints hold. Upstream checkout is never modified.
- Units: model in mV and ms; senses dimensionless, scaled by `DRIVE_MV = 7.0` in `FlyBrain`.
- Shiu parameters exact: v_rest -52, v_reset -52, v_thresh -45, tau_m 20, tau_syn 5, t_ref 2.2, delay 1.8, PSP 0.275 mV per synapse.
- New params (name, default, bounds): `dt_ms` 1.0 [0.25, 1.0] (snapped to 1, 0.5, 0.25), `noise_mv` 0.5 [0, 3], `light` 1.0 [0, 3], `ball_contrast` 1.0 [0, 1], `loom_shortcut` 1 [0, 1], `mb_strength` 0.3 [0, 1], `readout_motor` 0 [0, 1], `punish_reflex` 0 [0, 1]. `retina_strength` is removed.
- Edge index means CSR position everywhere. `learned.npz` from before this batch is ignored (new graph hash).
- Every task ends with `uv run pytest -q` green and a commit.

## Tasks

### Task 1: Project graph with the full transmitter table (`flypong/graph.py`)

- `SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}`; `PSP_MV = 0.275`.
- `select_neurons(annotations_path) -> (body_ids, cell_types)` with upstream's rule (nonempty superclass, no "glia", sorted by body ID).
- `build(raw_dir, out_path) -> dict`: read annotations + transmitters, scan the raw edge feather in record batches, keep edges with both ends selected, sort by (source, target), build `indptr`, save `body_ids, indptr, source, target, count, weight, cell_types, neurotransmitters, metadata`. Metadata records signs, PSP, counts, source SHA-256s.
- `load(path) -> dict of arrays` and `main()` (`python -m flypong.graph`).
- `config.graph_path()`: project `data/graph.npz` if it exists, else upstream. `config.upstream_graph_path()` for the equality check.
- Tests (`tests/test_graph.py`): synthetic feather files (annotations, transmitters, edges) in `tmp_path`; assert ordering, glia exclusion, histamine sign -1, dopamine sign 0, CSR `indptr` correct, weights `0.275 * count * sign`.
- Real build: `uv run python -m flypong.graph`; assert `body_ids` equals upstream's (print both lengths and a boolean). Expect 166,700 neurons, 25,582,938 edges, histamine edges nonzero.

### Task 2: `ShiuLIF` model (`flypong/neurons.py`) and event-driven kernel

- Constructor `ShiuLIF(n, indptr, target, weight, *, device, dt_ms=1.0, noise_mv=0.0, seed=0)` with the Shiu constants as keyword defaults. `gain` calibrated numerically at init (single-neuron test spike).
- Methods: `forward(drive_mv_per_ms) -> spikes(bool)`, `reset()`, `set_dt(dt)` (rebuilds ring/refractory steps, recalibrates gain), `set_noise(sigma)`, properties `delayed_pre`, `last_positions`, `device`, `n`. Buffers `weight` (mutable), `source` (expanded int32 for plasticity), `target`, `indptr`.
- Tests (`tests/test_neurons.py`): a driven neuron spikes and then is refractory for `ref_steps`; a spike reaches its target after exactly `delay_steps`; PSP of one spike peaks at `w` mV within 2%; inhibitory weight lowers `v`; noise 0 is deterministic and noise > 0 produces spikes over 2,000 steps at rest; `last_positions` lists exactly the transmitted edges; event-driven result equals a dense reference on a random 50-neuron graph for 100 steps.
- Real timing probe: full graph, 1,000 steps at dt 1 with noise 0.5, report steps/s and spikes/step. Then re-run the lateralization probe (LC4 left vs right, dark-ball not yet) and record DNp02/DNp04 side counts. If lateralization is lost, stop and report.

### Task 3: `FlyBrain` on `ShiuLIF`; plasticity on CSR positions and RPE

- `FlyBrain.from_graph` loads the project graph into `ShiuLIF`; `from_arrays` builds CSR from edge lists for tests. `DRIVE_MV = 7.0`. `tick(drive, k, events, learning, rate, params)` where params carries `dt_ms`, `noise_mv` (applied via `set_dt`/`set_noise` when changed) and counts four readout sets (`dn_left/right`, `mn_left/right`) plus optional monitor sets (`kc`, `mbon`, `lplc2`).
- `Plasticity`: eligibility updated only on transmitted plastic edges (`plastic_rank[last_positions]`), decayed per tick by `decay_e ** k`; `expected`, `rpe`, magnitude-scaled bursts and `G`; `DIFFUSE_REWARD_GAIN`, `punish_reflex`. Stats gain `expected`, `rpe`. Save/load unchanged (new graph hash).
- Tests updated: `test_brain.py`, `test_plasticity.py` (RPE: first return from expected 0.5 gives magnitude 0.5; after many returns punishment magnitude approaches 1; `punish_reflex = 0` leaves reflex weights unchanged after a miss).

### Task 4: Senses: real eye, mushroom-body input, motor sets

- `SensoryMap` gains: photoreceptor column inheritance (`photoreceptors` per eye with hex coords), `light`/`ball_contrast` terms, `loom_shortcut` gate, `mb_vpn` per eye (visual projection neurons with edges onto Kenyon cells), `mn_left/right`, `kc`, `mbon`, `lplc2` index sets. `retina_strength` removed.
- Tests: synthetic graph where R cells target lamina cells with known hex; ball column darkens the right photoreceptors; shortcut off removes looming drive; mb term scales with proximity on the correct eye.
- Real probe (60 s equivalent, bot geometry): with the shortcut off, does the dark ball fire LPLC2 and DNp02/DNp04 on the correct side? Record and decide the `loom_shortcut` default.

### Task 5: Server, page, README, measurement

- Config params added; server passes params to `tick`; hello carries `model`; stats carry the new counts.
- Page: settings (brain step select, noise, light, contrast, shortcut, MB input, readout, punishment-on-reflex) with dropdowns; status shows steps/s and spikes/step; learning panel shows expected and last RPE; HUD shows MN counts and KC/MBON/LPLC2 spikes.
- Measurement session as in the spec; README gets a "Realism batch" section with all numbers and the verdicts on the eye, speed, and learning.

## Self-review

Spec sections 1 to 7 map to Tasks 1 to 5. Units and constants appear once in Global Constraints and are referenced by name. Edge-index semantics (CSR) is stated in Task 1 and used by Tasks 2 to 4. The measurement gate for the shortcut default is in Task 4 and repeated in the spec.
