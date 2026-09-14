"""Start the Fly Pong server: python -m flypong [--host H] [--port P] [--device auto|mps|cpu]."""
from __future__ import annotations

import argparse

import numpy as np
from aiohttp import web

from . import config
from .brain import FlyBrain
from .plasticity import Plasticity, build_dopamine_edges, dopamine_cells, load_dopamine_edges, select_plastic
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
    parser.add_argument("--no-learning", action="store_true", help="Run the fixed connectome without plasticity")
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

    if not args.no_learning:
        with np.load(graph, allow_pickle=False) as g:
            body_ids, source, target, weight, types = g["body_ids"], g["source"], g["target"], g["weight"], g["cell_types"]
        if not config.DOPAMINE_PATH.exists():
            raw = config.raw_edges_path()
            if not raw.exists():
                print(f"Missing {raw}; run the upstream download (see README).")
                return 1
            print("Building dopamine edges from the raw synapse file (about 10 s)...", flush=True)
            print(build_dopamine_edges(raw, body_ids, types, config.DOPAMINE_PATH), flush=True)
        dop_src, dop_dst, dop_cnt = load_dopamine_edges(config.DOPAMINE_PATH, brain.n)
        pam, ppl1 = dopamine_cells(types)
        dop_sign = np.where(np.isin(dop_src, pam), 1.0, -1.0).astype(np.float32)
        idx, innervated, reflex = select_plastic(source, target, weight, dop_dst, dop_cnt, senses.annotations.superclass)
        brain.attach_plasticity(Plasticity(brain.model, idx, innervated, reflex, dop_src, dop_dst, dop_cnt,
                                           dop_sign, pam, ppl1, brain.graph_sha))
        loaded = brain.load_memory(config.LEARNED_PATH)
        print(f"Plasticity: {len(idx):,} synapses ({int(innervated.sum()):,} dopamine-innervated, "
              f"{int(reflex.sum()):,} reflex pathway), {len(pam) + len(ppl1)} dopamine cells, "
              f"{len(dop_dst):,} dopamine synapses; memory {'loaded' if loaded else 'fresh'}", flush=True)

    print(f"Open http://{args.host}:{args.port}/", flush=True)
    web.run_app(create_app(brain, senses), host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
