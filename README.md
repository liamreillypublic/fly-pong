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

The one shortcut, stated on the page too: the link from the retina image to
the looming detectors is computed by this program, because the simplified
leaky integrate-and-fire model cannot carry a small patch of retinal input
through the fly's lobula. Everything downstream of the looming detectors is
the fly's own wiring.

How good is it? Against the built-in bot at the default ball speed, the fly
returned 5 of 8 balls in a 30 second test. Slow the ball down and it returns
nearly everything; speed it up and its reaction time loses.

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
Sliders: ball speed, brain ms per frame, looming strength, retina strength,
motor gain.

## Tests

```sh
uv run pytest -q
```

## Credits

MaleCNS v1.0 data: FlyEM / HHMI Janelia, University of Cambridge, MRC
Laboratory of Molecular Biology, and Google Research, CC BY 4.0.
Simulator: [seohyunjun/mps-malecns-model](https://github.com/seohyunjun/mps-malecns-model),
used unmodified. Fly Pong code: MIT.
