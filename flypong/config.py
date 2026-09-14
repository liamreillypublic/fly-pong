"""Paths and the tunable parameter table with bounds."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
STATIC_DIR = PROJECT_DIR / "static"
ATLAS_PATH = STATIC_DIR / "atlas.bin"
ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
RAW_EDGES_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
DATA_DIR = PROJECT_DIR / "data"
DOPAMINE_PATH = DATA_DIR / "dopamine.npz"
LEARNED_PATH = DATA_DIR / "learned.npz"
DIFFUSE_GAIN = 0.3
SAVE_INTERVAL_S = 60.0
# Inhibitory synapses are scaled up at load. Measured 2026-09-14: at 1x the
# mushroom body ignites (Kenyon cells at 21 Hz each) and the escape signal is
# 1.5:1; at 2x Kenyon cells fire at 0.7 Hz and the escape signal is 4:1.
INHIBITION_SCALE = 2.0


def data_root() -> Path:
    """Upstream simulator checkout holding data/graph.npz and data/raw/."""
    return Path(os.environ.get("FLYPONG_DATA", PROJECT_DIR.parent / "mps-malecns-model"))


PROJECT_GRAPH_PATH = DATA_DIR / "graph.npz"
TRANSMITTERS_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
DRIVE_MV = 7.0   # senses are dimensionless (threshold 1); the model's rest-to-threshold gap is 7 mV


def upstream_graph_path() -> Path:
    return data_root() / "data" / "graph.npz"


def graph_path() -> Path:
    """The project's own CSR graph when built, else the upstream one."""
    return PROJECT_GRAPH_PATH if PROJECT_GRAPH_PATH.exists() else upstream_graph_path()


def annotations_path() -> Path:
    return data_root() / "data" / "raw" / ANNOTATIONS_FILE


def transmitters_path() -> Path:
    return data_root() / "data" / "raw" / TRANSMITTERS_FILE


def raw_edges_path() -> Path:
    return data_root() / "data" / "raw" / RAW_EDGES_FILE


@dataclass(frozen=True)
class Param:
    default: float
    lo: float
    hi: float


PARAMS: dict[str, Param] = {
    "loom_strength": Param(0.15, 0.0, 1.0),
    "ball_radius_columns": Param(2, 0, 6),
    "steps_per_tick": Param(4, 1, 16),
    "motor_decay": Param(0.7, 0.0, 0.99),
    "motor_gain": Param(0.5, 0.0, 5.0),
    "background_drive": Param(0.02, 0.0, 0.1),
    "learning_enabled": Param(1, 0, 1),
    "learning_rate": Param(0.005, 0.0, 0.1),
    "dt_ms": Param(1.0, 0.25, 1.0),
    "noise_mv": Param(0.5, 0.0, 1.5),
    "light": Param(0.05, 0.0, 1.0),
    "ball_contrast": Param(1.0, 0.0, 1.0),
    "loom_shortcut": Param(1, 0, 1),
    "mb_strength": Param(0.1, 0.0, 1.0),
    "readout_motor": Param(0, 0, 1),
    "punish_reflex": Param(0, 0, 1),
    "depression_u": Param(0.2, 0.0, 0.5),
}
DT_CHOICES = (1.0, 0.5, 0.25)


def snap_dt(value: float) -> float:
    """Snap a requested time step to the nearest supported choice."""
    return min(DT_CHOICES, key=lambda c: abs(c - float(value)))


def defaults() -> dict[str, float]:
    return {name: p.default for name, p in PARAMS.items()}


def clamp_params(raw: object) -> dict[str, float]:
    """Full parameter dict: unknown keys dropped, missing or non-numeric values
    replaced by defaults, numeric values clamped to [lo, hi]."""
    out = defaults()
    if not isinstance(raw, dict):
        return out
    for name, p in PARAMS.items():
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value != value:  # NaN
            continue
        out[name] = min(max(float(value), p.lo), p.hi)
    return out
