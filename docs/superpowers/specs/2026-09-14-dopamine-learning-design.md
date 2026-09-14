# Dopamine learning and endless play: design

Date: 2026-09-14
Status: approved in conversation
Extends: `2026-09-14-fly-pong-design.md`

## Goal

Make the fly learn from reward and punishment through its own synapses, as
close to real fly biology as this simulator allows, with memory that
persists across restarts. Remove the win condition so play is endless.

The user accepts that this may not improve Pong performance. The page and
README must report what learning actually does, measured, not assumed.

## Biology this follows

In a real fly, reward is signaled by PAM dopamine neurons and punishment by
PPL1 dopamine neurons. Their axons reach the mushroom body, where synapses
from Kenyon cells onto mushroom-body output neurons change when the Kenyon
cell was recently active and dopamine arrives (a three-factor rule:
presynaptic activity, postsynaptic activity, dopamine). All of these cells
are in MaleCNS v1.0: 316 PAM cells (15 types), 16 PPL1 cells (8 types),
4,064 Kenyon cells, 97 MBONs, 2 APL cells.

Dopamine neurons have output sign 0 in the upstream graph (dopamine is not
in the simulator's sign table), so their spikes carry no fast current. This
design uses them purely as modulators, which is their real role.

## What is real and what is added

Real: the dopamine neurons, which neurons they innervate and how strongly
(taken from the raw synapse counts), the eligibility rule, the sign of
every synapse, and the pathway the paddle uses.

Added, and stated on the page: (1) a diffuse dopamine signal that also
reaches the visual-projection-to-descending pathway, because in this model
mushroom-body output cannot reach the escape neurons, so without it learning
could never change play; (2) valence by dopamine type (PAM potentiates,
PPL1 depresses) instead of the fly's compartment-specific depression, which
would need per-compartment MBON valences this design does not model.

## Learning mechanics

### Dopamine delivery

- Reward (fly returned the ball): stimulate all PAM cells with +0.3 per
  step for 8 brain steps (two ticks at the default 4 steps per tick).
- Punishment (ball passed the fly): same for all PPL1 cells.
- Dopamine trace per neuron `D[j]` (float32, length n): each step
  `D = D * exp(-1/30)` then `D[j] += sum over dopamine edges (i -> j) of
  spike[i] * count_ij / total_dopamine_count_j`, positive for PAM sources
  and negative for PPL1 sources. Time constant 30 brain ms.
- Diffuse trace `G` (scalar): `G = G * exp(-1/30)`, plus `+1` on reward
  and `-1` on punishment at the tick the event arrives.
- Dopamine edges come from the raw upstream synapse file, not the
  prepared graph (which zeroed their weights). A build step writes
  `data/dopamine.npz` with `src`, `dst` (graph indices) and `count`
  (int64), for every raw edge whose presynaptic body is a PAM or PPL1 cell
  and whose postsynaptic body is in the graph. `__main__` builds it
  automatically if it is missing (about 10 s).

### Plastic synapses

The plastic set `P` is the union of:

1. edges whose postsynaptic neuron receives at least one dopamine synapse
   (the mushroom body and whatever else the dopamine axons reach), and
2. edges whose presynaptic neuron has superclass `visual_projection` or
   whose postsynaptic neuron has superclass `descending_neuron` (the
   reflex pathway, 2.2 million edges).

Edges with weight 0 in the prepared graph are excluded (they cannot change
sign-preservingly). If `|P|` exceeds 8 million, group 1 is restricted to
postsynaptic neurons with at least 5 dopamine synapses; the count is printed
at startup and shown on the page.

### Eligibility and update

Per brain step, for every plastic edge `p = (i -> j)`:
`e_p = min(1, e_p * exp(-1/100) + [spike_i at t-1] * [spike_j at t])`.
Time constant 100 brain ms. Presynaptic spikes are those of the previous
step, matching the model's one-step transmission delay.

Per tick, after the `k` steps:
`dopamine_p = D[j] + diffuse_gain * G * [p in reflex group]`,
`w_p = w_p + rate * dopamine_p * e_p * |w0_p|`,
then `w_p` is clamped so that `sign(w_p) = sign(w0_p)` and
`0.1 * |w0_p| <= |w_p| <= 4 * |w0_p|`.

`w0` is the original prepared weight. `rate` is the learning-rate
parameter (default 0.02, bounds 0 to 0.2). `diffuse_gain` is fixed at 0.3.
When learning is disabled (`learning_enabled = 0`), traces still decay but
no weight update is applied.

Eligibility and dopamine traces are cleared on `reset` (New game).
Weights are not.

### Memory

- `data/learned.npz` holds the graph SHA-256, the plastic edge indices,
  the current plastic weights, lifetime reward and punishment counts, and
  accumulated brain age in seconds.
- The server saves it every 60 s when anything changed, and on shutdown.
- On start, if the file exists and its graph hash matches, weights are
  loaded; otherwise the fly starts from the original connectome.
- `{"type":"forget"}` restores `w0`, zeroes counters and age, deletes the
  file, and replies `{"type":"forget_ok"}`. The page asks for confirmation
  before sending it.

### Per-tick learning stats

`stats.learning` in every `command`:
`{"pam": spikes this tick, "ppl1": spikes this tick, "event": "reward" |
"punishment" | null, "mb_drift": mean |w - w0| / |w0| over group 1,
"reflex_drift": same over group 2, "rewards": lifetime, "punishments":
lifetime, "age_s": brain age, "plastic": |P|, "saved_ago_s": seconds
since last save or null, "enabled": bool}`.
Drift is recomputed every 10 ticks and cached between.

## Protocol changes

- Client `state` gains `"events": ["return" | "miss", ...]`, the events
  since the last state message, usually empty or one item. The server
  applies the first reward and the first punishment in the list (one burst
  each per tick).
- New params: `learning_enabled` (default 1, bounds 0 to 1, treated as
  boolean by `>= 0.5`), `learning_rate` (default 0.02, bounds 0 to 0.2).
- New message `forget` -> `forget_ok`.
- `hello` gains `"learning": {"plastic": |P|, "dopamine_cells": 332,
  "loaded_memory": bool}`.

## Game changes

- No win condition. `WIN_SCORE`, the banner, and Rematch are removed.
  New game zeroes the score and return statistics and resets membrane
  state; learned weights persist.
- Return rate is shown for the last 20 balls (rolling) as well as lifetime
  for the session.

## Page: learning panel

Placed under the escape-neuron chart:

- Dopamine line: `PAM 12 · PPL1 0`, and a badge that flashes green
  "reward" or red "punishment" for 600 ms on each event.
- Memory line: `mushroom body drift 3.1% · reflex pathway drift 0.8% ·
  rewards 41 · punishments 17 · age 12m 04s · saved 20 s ago`.
- Learning chart (600 by 100): green bars up for rewards, red bars down for
  punishments over the last 120 events, with a white line for the rolling
  20-ball return rate at each event.
- Controls: checkbox "Learning (dopamine)", slider "Learning rate"
  0 to 0.2 step 0.005 default 0.02, button "Forget everything" (confirm
  dialog), each with a "What is this?" dropdown.

## Files

- `flypong/plasticity.py`: dopamine edge build, `Plasticity` class,
  save/load/forget. Pure torch on the brain's device.
- `flypong/brain.py`: `attach_plasticity`, `tick(drive, k, reward=0,
  learning=True, rate=0.02)`, `TickResult.learning`.
- `flypong/config.py`: `DATA_DIR`, `DOPAMINE_PATH`, `LEARNED_PATH`, new
  params.
- `flypong/server.py`: events, forget, periodic save, shutdown save.
- `flypong/__main__.py`: build dopamine edges if missing, attach
  plasticity, load memory.
- `static/*`: endless play, learning panel.
- `tests/test_plasticity.py`, updates to `test_brain.py`,
  `test_server.py`, `test_config.py`.
- `.gitignore`: `data/`.

## Error handling

- Raw synapse file missing when building dopamine edges: exit with the
  upstream download command.
- `learned.npz` with a different graph hash: ignored with a printed
  warning, not deleted.
- Nonfinite weights after an update: the update is discarded, weights are
  restored from the last good copy (kept on device), and an error message
  is sent; learning stays enabled.
- Save failure: logged, retried next minute.

## Measurement

Before merging, run a bot session at ball speed 5 for 5 minutes with
learning on and record return rate per minute, then 2 minutes with learning
off from the learned weights, then Forget and 2 minutes more. Put the
numbers in the README whatever they show.
