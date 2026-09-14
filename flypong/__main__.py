"""Start the Fly Pong server: python -m flypong [--host H] [--port P] [--device auto|mps|cpu]."""
from __future__ import annotations

import argparse
import time

import numpy as np
from aiohttp import web

from . import config
from . import graph as graph_module
from .brain import FlyBrain
from .plasticity import Plasticity, build_dopamine_edges, dopamine_cells, load_dopamine_edges, select_plastic
from .senses import SensoryMap
from .server import create_app

UPSTREAM_HELP = """Build the upstream data first:
  cd {root}
  .venv/bin/python -m malecns download
  .venv/bin/python -m malecns prepare"""


def ensure_project_graph() -> int:
    """Build data/graph.npz (full transmitter table, CSR) if it is missing. Returns 0 or an exit code."""
    if config.PROJECT_GRAPH_PATH.exists():
        return 0
    paths = (config.annotations_path(), config.transmitters_path(), config.raw_edges_path())
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Missing raw data files:\n  " + "\n  ".join(str(p) for p in missing))
        print(UPSTREAM_HELP.format(root=config.data_root()))
        return 1
    print("Building the project graph with the full transmitter table (about 30 s)...", flush=True)
    meta = graph_module.build(*paths, config.PROJECT_GRAPH_PATH)
    print(f"Graph: {meta['neurons']:,} neurons, {meta['edges']:,} edges, "
          f"{meta['histamine_neurons']:,} histamine neurons now transmit", flush=True)
    upstream = config.upstream_graph_path()
    if upstream.exists():
        with np.load(upstream, allow_pickle=False) as u:
            if not np.array_equal(u["body_ids"], graph_module.load(config.PROJECT_GRAPH_PATH)["body_ids"]):
                print("Neuron order differs from the upstream graph; the atlas and dopamine files would be wrong.")
                return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fly Pong server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--no-learning", action="store_true", help="Run the fixed connectome without plasticity")
    parser.add_argument("--inhibition", type=float, default=config.INHIBITION_SCALE,
                        help="Scale factor on inhibitory synapses (default %(default)s; 1 = published weights)")
    args = parser.parse_args(argv)

    if not config.ATLAS_PATH.exists():
        print(f"Missing {config.ATLAS_PATH}. Run: uv run python -m flypong.atlas")
        return 1
    if not config.annotations_path().exists():
        print(f"Missing {config.annotations_path()}")
        print(UPSTREAM_HELP.format(root=config.data_root()))
        return 1
    code = ensure_project_graph()
    if code:
        return code
    graph_path = config.PROJECT_GRAPH_PATH
    defaults = config.defaults()

    print("Loading brain...", flush=True)
    brain = FlyBrain.from_graph(graph_path, args.device, dt_ms=defaults["dt_ms"], noise_mv=defaults["noise_mv"],
                                inhibition_scale=args.inhibition)
    if brain.device != "mps":
        print("WARNING: running on CPU; expect slow ticks", flush=True)
    senses = SensoryMap.from_files(graph_path, config.annotations_path())
    brain.set_readout(senses.dn_left, senses.dn_right, senses.mn_left, senses.mn_right)
    brain.set_monitors(**senses.monitors)
    eyes = senses.eyes
    print(f"Brain ready: {brain.n:,} neurons on {brain.device}; escape DNs left {len(senses.dn_left)}, "
          f"right {len(senses.dn_right)}; motor neurons {len(senses.mn_left)}/{len(senses.mn_right)}; "
          f"photoreceptors with columns {len(eyes['L'].photoreceptors)}/{len(eyes['R'].photoreceptors)}; "
          f"mushroom-body inputs {len(eyes['L'].mb_vpn)}/{len(eyes['R'].mb_vpn)}", flush=True)

    # speed probe: idle steps per second at the default settings
    idle = np.zeros(brain.n, np.float32)
    brain.tick(idle, 20)
    t0 = time.perf_counter()
    r = brain.tick(idle, 200)
    brain.steps_per_s = round(200 / (time.perf_counter() - t0))
    brain.reset()
    print(f"Model: {brain.model_info()['name']}, dt {brain.model.dt} ms, noise {brain.model.noise_mv} mV, "
          f"depression {brain.model.depression_u}, inhibition x{args.inhibition:g}; idle {brain.steps_per_s} steps/s, "
          f"{r.total_spikes / 200:.1f} spikes/step", flush=True)

    if not args.no_learning:
        g = graph_module.load(graph_path)
        if not config.DOPAMINE_PATH.exists():
            raw = config.raw_edges_path()
            if not raw.exists():
                print(f"Missing {raw}; run the upstream download (see README).")
                return 1
            print("Building dopamine edges from the raw synapse file (about 10 s)...", flush=True)
            print(build_dopamine_edges(raw, g["body_ids"], g["cell_types"], config.DOPAMINE_PATH), flush=True)
        dop_src, dop_dst, dop_cnt = load_dopamine_edges(config.DOPAMINE_PATH, brain.n)
        pam, ppl1 = dopamine_cells(g["cell_types"])
        dop_sign = np.where(np.isin(dop_src, pam), 1.0, -1.0).astype(np.float32)
        idx, innervated, reflex = select_plastic(g["source"], g["target"], g["weight"], dop_dst, dop_cnt,
                                                 senses.annotations.superclass)
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
