# Fly Pong

Pong where the right paddle is a live simulation of the complete adult male
fruit fly central nervous system: 166,700 neurons and 25.6 million connections
from the MaleCNS v1.0 connectome, stepping every frame on an Apple Silicon GPU.

![Fly Pong](docs/screenshot.png)

## How it works

The fly sits at the right paddle facing left, seen from above. Its left eye
covers the top half of the screen, its right eye the bottom half.

- **Seeing.** Each frame the ball is painted onto the lamina (retina) columns
  of the matching eye, and the looming detectors on that side (LC4 and LPLC2)
  receive a signal that grows as the ball approaches.
- **Thinking.** The whole brain steps live, with state carried between
  frames and no resets during a game. A slider sets brain milliseconds per
  game frame (default 4). On an M3 Pro that is about 22 brain updates per
  second. Reaction latency is real: the escape pathway needs about 8 brain
  milliseconds from stimulus to descending output.
- **Moving.** No training. Left-side descending neurons firing move the
  paddle up, right-side down. The cells that fire are DNp02 and DNp04, the
  fly's real looming-escape neurons, which the connectome wires strictly
  to one side.

The one shortcut on the sensory side, stated on the page too: the link from
the retina image to the looming detectors is computed by this program,
because the simplified leaky integrate-and-fire model cannot carry a small
patch of retinal input through the fly's lobula. Everything downstream of
the looming detectors is the fly's own wiring.

## Learning

The fly learns from reward and punishment through its own synapses, and
remembers across restarts.

- **Dopamine is real cells.** Returning the ball stimulates the fly's 316
  PAM dopamine neurons; missing it stimulates its 16 PPL1 neurons. These
  are the cells a real fly uses for reward and punishment. Their spikes
  travel down their real axons, rebuilt from the raw synapse counts, to the
  8,599 neurons they actually innervate: 4,061 of the 4,064 Kenyon cells and
  all 97 output neurons of the mushroom body, the fly's learning center,
  plus 74 descending neurons.
- **Three-factor plasticity.** 4.36 million synapses (everything onto a
  dopamine-innervated neuron, plus the visual-projection-to-descending
  pathway that drives the paddle) keep an eligibility trace that rises when
  the sending cell fired just before the receiving cell and fades over 100
  brain milliseconds. Each tick, every eligible synapse that received
  dopamine changes: reward strengthens, punishment weakens, scaled by the
  learning rate. Synapses never change sign and stay between 0.1x and 4x
  their original strength.
- **Memory.** Learned weights are saved to `data/learned.npz` every minute
  and on shutdown, and reloaded at start. "Forget everything" restores the
  original connectome.

Two additions are ours, not the fly's, and the page says so: a weak diffuse
dopamine signal also reaches the paddle pathway (mushroom-body output cannot
reach the escape neurons in this model, so without it learning could never
change play), and reward potentiates while punishment depresses, where the
real mushroom body depresses in both cases with valence set by which output
neuron is affected.

One honest finding: in Pong the mushroom body never changes, because nothing
in the game drives its Kenyon cells. All of the learning happens on the
reflex pathway. The fly cannot learn a new direction of movement, which is
fixed by anatomy; it can only learn how fast and how hard to react.

### Measured (2026-09-14, bot opponent, ball speed 5, default settings)

One clean session, single tab, from a fresh Forget:

| Phase | Balls faced | Returned | Synapses changed |
|---|---|---|---|
| Original connectome, learning off, 3 min | 37 | 32 (86%) | 0 |
| Learning on, 5 min (37 rewards, 27 punishments) | 64 | 37 (58%) | 24,916, by 7.1% on average |
| Trained weights frozen, learning off, 3.5 min | 48 | 23 (48%) | 24,916 |

So with this rule the fly learns, and gets worse. The mechanism did exactly
what it was built to do: every miss weakened the reflex synapses that had
just fired, which made the next approach slower, which caused another miss.
A punishment spiral. Rewards could not outrun it because the fly's reaction
time, not its reflex strength, is what loses rallies at this ball speed.

The obvious fix, not yet applied: let the diffuse signal that reaches the
paddle pathway carry reward only, and keep punishment where the real PPL1
axons actually go, the mushroom body. That is arguably more faithful, since
the diffuse signal was our addition in the first place. It would let the
reflex strengthen with success (up to the 4x cap) and never be weakened by
failure. Whether that improves play is a measurement for another session.

Slow the ball down and the fly returns nearly everything; speed it up and
its reaction time loses regardless of learning.

## Setup

1. Clone and build the upstream simulator next to this folder:
   ```sh
   cd ~/Developer
   git clone https://github.com/seohyunjun/mps-malecns-model.git
   cd mps-malecns-model
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements-lock.txt
   .venv/bin/python -m malecns download    # 1.1 GB of official data
   .venv/bin/python -m malecns prepare     # builds data/graph.npz
   ```
2. Install and build the atlas:
   ```sh
   cd ~/Developer/fly-pong
   uv sync
   uv run python -m flypong.atlas
   ```
3. Run:
   ```sh
   uv run python -m flypong
   ```
   Open http://127.0.0.1:8765/. Set `FLYPONG_DATA` if the simulator lives
   somewhere else.

## Controls

W/S or arrows move your paddle. Space pauses. B hands your paddle to a bot.
N starts a new game. First to 7 points wins; the page tracks the fly's
return rate as you play, and pauses itself when the tab is hidden.

Sliders: ball speed, brain ms per frame, looming strength, retina strength,
motor gain. Every setting has a "What is this?" dropdown explaining what it
changes and which real neurons are involved. Easy, Normal and Hard presets
set ball speed and brain time together; "Reset sliders" restores defaults.

The brain panel shows a color legend of the nine neuron classes with live
spike counts per tick, and a ten-second strip chart of left versus right
escape-neuron spikes with the resulting paddle command.

## Tests

```sh
uv run pytest -q
```

## Credits

MaleCNS v1.0 data: FlyEM / HHMI Janelia, University of Cambridge, MRC
Laboratory of Molecular Biology, and Google Research, CC BY 4.0.
Simulator: [seohyunjun/mps-malecns-model](https://github.com/seohyunjun/mps-malecns-model),
used unmodified. Fly Pong code: MIT.
