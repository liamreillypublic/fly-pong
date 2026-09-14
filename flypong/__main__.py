"""Start the Fly Pong server: python -m flypong [--host H] [--port P] [--device auto|mps|cpu]."""
from __future__ import annotations

import argparse

from aiohttp import web

from . import config
from .brain import FlyBrain
from .senses import SensoryMap
from .server import create_app

UPSTREAM_HELP = """Build the upstream data first:
  cd {root}
  .venv/bin/python -m malecns download
  .venv/bin/python -m malecns prepare"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fly Pong server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args(argv)

    graph, annotations = config.graph_path(), config.annotations_path()
    missing = [p for p in (graph, annotations) if not p.exists()]
    if missing:
        print("Missing data files:\n  " + "\n  ".join(str(p) for p in missing))
        print(UPSTREAM_HELP.format(root=config.data_root()))
        return 1
    if not config.ATLAS_PATH.exists():
        print(f"Missing {config.ATLAS_PATH}. Run: uv run python -m flypong.atlas")
        return 1

    print("Loading brain...", flush=True)
    brain = FlyBrain.from_graph(graph, args.device)
    if brain.device != "mps":
        print("WARNING: running on CPU; expect slow ticks", flush=True)
    senses = SensoryMap.from_files(graph, annotations)
    brain.set_readout(senses.dn_left, senses.dn_right)
    print(f"Brain ready: {brain.n:,} neurons on {brain.device}; "
          f"descending left {len(senses.dn_left)}, right {len(senses.dn_right)}", flush=True)
    print(f"Open http://{args.host}:{args.port}/", flush=True)
    web.run_app(create_app(brain, senses), host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
