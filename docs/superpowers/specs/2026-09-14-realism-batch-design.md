# Realism batch: design

Date: 2026-09-14
Status: approved in conversation
Extends: the Fly Pong design and the dopamine learning design.

## Goal

Seven improvements that narrow the gap between the simulation and a real
fly brain, each measured: (1) full transmitter table, (2) the published
Shiu et al. neuron model with spontaneous activity and an event-driven
kernel aimed at real-time brain speed, (3) a real eye with the ball as a
dark object, (4) dopamine as reward prediction error, (5) mushroom-body
input through its real visual projection neurons, (6) paddle readout from
the nerve cord's motor neurons, (7) page controls, README, measurement.

A physics body is out of scope (multi-day). Item 6 is the bounded form.

## Facts measured before design

- Histamine neurons: 7,891 (R1-R6 3,377; T1 1,777; R7/R8 about 1,800);
  89,723 synapses currently weight 0. Photoreceptors have no eye-column
  coordinates in the annotation file; lamina cells L1/L2/L3 do.
- Mean out-degree 153.5, median 112. 500 spiking neurons touch about
  77,000 synapses of 25.6 million.
- Visual projection neurons onto Kenyon cells: 252 cells, 1,616 synapses,
  332 Kenyon cells reached (top types aMe26, aMe12, LoVP38, MeVP45).
- Motor neurons: 355 left, 353 right.

## 1. Own graph with the full transmitter table

`flypong/graph.py` builds `data/graph.npz` from the upstream raw files
with the same neuron selection and ordering as upstream (annotated
superclass, glia excluded, sorted by body ID), so `body_ids` is identical
and `atlas.bin` and `dopamine.npz` stay valid. A startup check asserts
equality with the upstream graph's `body_ids` when that file exists.

Signs: acetylcholine +1, GABA -1, glutamate -1, histamine -1 (ionotropic
histamine-gated chloride channels). Dopamine, octopamine, serotonin,
unclear, unknown: 0 (neuromodulators, not fast transmitters).

Edges are stored in CSR order by source: `indptr` (int64, n+1), `target`
(int32), `count` (int32 raw synapse count), `weight` (float32,
`0.275 mV * count * sign(source)`), plus `source` (int32, expanded, for
code that needs it), `body_ids`, `cell_types`, `neurotransmitters`,
`metadata`. Every "edge index" in Fly Pong now means CSR position.

`config.graph_path()` returns `data/graph.npz` under the project when it
exists, else the upstream file (which the new model cannot use; startup
says so and tells the user to run `python -m flypong.graph`).

## 2. Neuron model: `flypong/neurons.py`, class `ShiuLIF`

Parameters from Shiu et al. 2024 (units mV, ms): `v_rest -52`,
`v_reset -52`, `v_thresh -45`, `tau_m 20`, `tau_syn 5`, `t_ref 2.2`,
`delay 1.8`, PSP amplitude per synapse 0.275 (already in `weight`). `dt`
configurable: 1.0 (default), 0.5, 0.25. Refractory and delay are rounded
to whole steps, minimum 1.

State per neuron: `v`, `g` (synaptic drive in mV), `ref` (steps left),
a ring buffer of the last `delay_steps` spike vectors, `spikes` (last
step).

Per step:
1. `pre = ring[t - delay_steps]`; event-driven: `active = nonzero(pre)`,
   expand their CSR ranges, `g.index_add_(0, target[pos], weight[pos])`.
   Arriving input to a refractory neuron is dropped, as in the original.
   `g` jumps by `w` and is integrated unscaled, exactly as in the original
   PyTorch implementation of the model (`v += dt/tau_m * (g - (v -
   v_rest))`): one synapse peaks the membrane at about `0.157 * w`, about
   0.043 mV. (A first draft calibrated the peak to `w` itself; that made
   the network supercritical: one LC4 stimulus ignited self-sustained
   activity of 3,000 spikes per step. Measured 2026-09-14 and corrected.)
2. `v = v_rest + (v - v_rest) * exp(-dt/tau_m) + g * dt / tau_m
   + drive * dt + noise * sqrt(dt) * randn`; `g *= exp(-dt/tau_syn)`.
3. Refractory neurons are held at `v_reset` and do not spike.
4. `spikes = v >= v_thresh`; spiking neurons reset and enter refractory.
5. Push `spikes` into the ring.

`drive` is in mV per ms. The senses produce dimensionless drive as before
(threshold-1 units); `FlyBrain` multiplies by `DRIVE_MV = 7.0` (the
7 mV gap from rest to threshold) so existing strengths keep their meaning.

`noise` (mV per sqrt(ms)) is a parameter `noise_mv`, default 0.5, bounds
0 to 3. Its calibration (spikes per step at rest) is printed by a probe
and shown on the page.

Exposed for plasticity: `weight` (mutable, CSR order), `target`,
`source`, `indptr`, `delayed_pre()` (the pre-spike vector used this
step), and `last_edge_positions()` (the `pos` tensor of the last step, so
eligibility can be updated only on synapses that actually transmitted).

Speed target: at `dt = 1` with about 500 to 2,000 spikes per step, at
least 500 steps per second on the M3 Pro. Measured and reported.

## 3. The real eye

Photoreceptors get eye columns by inheritance: each R cell takes the hex
column of the lamina cell (L1/L2/L3) it targets with the most synapses.
R cells with no such target get none and receive light but no ball.

New sensory terms (all dimensionless, multiplied by `DRIVE_MV`):

- `light` (default 1.0, bounds 0 to 3): constant drive on every
  photoreceptor with a column (R1-R6, R7, R8 types, i.e. all histamine
  neurons whose type starts with `R`).
- Ball as dark object: photoreceptors within `ball_radius_columns` of the
  ball's column on the eye facing the ball get `light * (1 -
  ball_contrast)`; `ball_contrast` default 1.0, bounds 0 to 1.
- The old direct lamina drive (`retina_strength`) is removed.
- `loom_shortcut` (default 1, bounds 0 to 1, boolean by `>= 0.5`): when
  on, the LC4/LPLC2 looming drive from the first design still applies.
  Measurement decides the shipped default: if the dark ball alone fires
  LPLC2 and the escape descending neurons within 60 ms on more than half
  of approaches, the default becomes 0.

## 4. Dopamine as reward prediction error

`Plasticity` keeps `expected`, an exponential moving average (alpha 0.1,
initial 0.5) of outcomes (1 return, 0 miss), updated after each event.
Reward magnitude on a return: `1 - expected`; punishment magnitude on a
miss: `expected`. Burst drive on PAM/PPL1 cells and the diffuse trace `G`
are scaled by that magnitude. The diffuse trace has separate gains:
`DIFFUSE_REWARD_GAIN = 0.3`, and `punish_reflex` (parameter, default 0,
bounds 0 to 1) multiplies the punishment side, so by default failure
never weakens the reflex pathway while success can strengthen it up to
the 4x cap. Stats gain `expected` and `rpe` (last event's signed
magnitude).

## 5. Mushroom-body input

At load, the sensory map finds the visual projection neurons that synapse
onto Kenyon cells (252 cells). `mb_strength` (default 0.3, bounds 0 to 1)
drives those on the eye facing the ball by `mb_strength * proximity`, the
same shape as the looming term. This is a real pathway into the fly's
learning center; whether its output reaches the paddle is measured (KC
and MBON spikes per tick are reported; mushroom-body drift should become
nonzero).

## 6. Motor-neuron readout

The brain counts four sets each tick: descending left/right and motor
(superclass `vnc_motor`) left/right. `readout_motor` (default 0, bounds 0
to 1) selects which pair feeds `MotorReadout`. `TickResult` gains
`mn_left`, `mn_right`. The page shows both pairs.

## 7. Page, README, measurement

New settings, each with a "What is this?" dropdown: brain step (1, 0.5,
0.25 ms), noise, light, ball contrast, looming shortcut, mushroom-body
input, paddle readout, punishment on reflex. Status line shows steps per
second and spikes per step. Learning panel shows expected return rate and
the last prediction error. Escape chart unchanged. Legend unchanged.

Measurement, same protocol as before (bot, ball 5, from Forget): 3 min
baseline learning off, 5 min learning on, 3 min frozen; plus a 60 s probe
of the real eye with the shortcut off. Numbers go in the README.

## Protocol

`hello.model = {"name": "shiu-lif", "dt_ms": ..., "steps_per_s": measured
at startup, "delay_steps", "ref_steps"}`. `command.stats` gains
`spikes_per_step`, `mn`, `learning.expected`, `learning.rpe`, `kc`,
`mbon`, `lplc2` (spike counts this tick).

## Error handling

- Missing project graph: startup builds it (about 60 s) after printing
  what it is doing.
- `body_ids` mismatch with upstream: refuse to start, print both counts.
- Nonfinite voltages: as before (reset + error).
- Learned memory keyed by the new graph hash; old files are ignored.
