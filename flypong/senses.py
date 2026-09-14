"""Game state to per-neuron drive vector. Pure numpy, no torch.

Frame of reference: the fly sits at the right paddle facing left, seen from
above. Its left eye covers the top half of the screen, its right eye the
bottom half.

Sensory terms (all dimensionless; the brain scales them to mV per ms):
- light: constant drive on every photoreceptor that has an eye column; the
  ball is a dark spot (reduced light) in the columns it covers, which is the
  stimulus a real fly's escape circuit responds to.
- looming shortcut (optional): direct drive on the LC4/LPLC2 looming
  detectors of the eye facing the ball, scaled by proximity and offset.
- mushroom-body input: drive on the visual projection neurons that synapse
  onto Kenyon cells, same shape as the looming term.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import graph as graph_module
from .annotations import Annotations, load

LOOM_TYPES = ("LC4", "LPLC2")
LAMINA_TYPES = ("L1", "L2", "L3")
PHOTORECEPTOR_PREFIXES = ("R1-R6", "R7", "R8")


@dataclass(frozen=True)
class GameState:
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    paddle_x: float
    paddle_y: float
    field_w: float
    field_h: float


@dataclass(frozen=True)
class Eye:
    loom: np.ndarray            # int64 indices of LC4 + LPLC2 cells on this side
    photoreceptors: np.ndarray  # int64 indices of R cells with an inherited eye column on this side
    hex1: np.ndarray            # int32, aligned with photoreceptors
    hex2: np.ndarray
    h1_min: int
    h1_max: int
    h2_min: int
    h2_max: int
    mb_vpn: np.ndarray          # int64 indices of visual projection neurons onto Kenyon cells, this side


def _starts(types: np.ndarray, prefixes) -> np.ndarray:
    out = np.zeros(len(types), dtype=bool)
    for p in prefixes:
        out |= np.char.startswith(types, p)
    return out


def inherit_columns(types, ann: Annotations, graph: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give each photoreceptor the eye column and the eye (soma side) of the
    columned cell it targets with the most synapses. Photoreceptor cell bodies
    lie in the retina outside the imaged volume, so their own side is unknown.
    Returns (hex1, hex2, side) arrays over all neurons: -1 / "?" where none."""
    n = len(types)
    hex1 = np.full(n, -1, np.int32)
    hex2 = np.full(n, -1, np.int32)
    side = np.asarray(ann.soma_side).copy()
    if graph is None:
        return hex1, hex2, side
    pr = _starts(types, PHOTORECEPTOR_PREFIXES)
    columned = (ann.hex1 >= 0) & (ann.hex2 >= 0) & ~pr
    src = np.asarray(graph["source"], np.int64)
    dst = np.asarray(graph["target"], np.int64)
    count = np.asarray(graph["count"], np.int64)
    keep = pr[src] & columned[dst]
    if not keep.any():
        return hex1, hex2, side
    s, d, c = src[keep], dst[keep], count[keep]
    order = np.lexsort((-c, s))                       # by source, strongest target first
    s, d = s[order], d[order]
    first = np.concatenate(([True], s[1:] != s[:-1]))
    hex1[s[first]] = ann.hex1[d[first]]
    hex2[s[first]] = ann.hex2[d[first]]
    side[s[first]] = ann.soma_side[d[first]]
    return hex1, hex2, side


class SensoryMap:
    def __init__(self, cell_types: np.ndarray, ann: Annotations, graph: dict | None = None):
        if len(cell_types) != ann.n:
            raise ValueError("cell_types and annotations length mismatch")
        self.n = len(cell_types)
        self.annotations = ann
        types = np.asarray(cell_types).astype(str)
        dn = ann.superclass == "descending_neuron"
        escape = dn & np.char.startswith(types, "DNp")
        self.dn_left = np.flatnonzero(escape & (ann.soma_side == "L")).astype(np.int64)
        self.dn_right = np.flatnonzero(escape & (ann.soma_side == "R")).astype(np.int64)
        motor = ann.superclass == "vnc_motor"
        self.mn_left = np.flatnonzero(motor & (ann.soma_side == "L")).astype(np.int64)
        self.mn_right = np.flatnonzero(motor & (ann.soma_side == "R")).astype(np.int64)
        kc = np.char.startswith(types, "KC")
        self.monitors = {
            "kc": np.flatnonzero(kc).astype(np.int64),
            "mbon": np.flatnonzero(np.char.startswith(types, "MBON")).astype(np.int64),
            "lplc2": np.flatnonzero(types == "LPLC2").astype(np.int64),
            "lc4": np.flatnonzero(types == "LC4").astype(np.int64),
            "photoreceptors": np.flatnonzero(_starts(types, PHOTORECEPTOR_PREFIXES)).astype(np.int64),
        }
        self.pr_hex1, self.pr_hex2, self.pr_side = inherit_columns(types, ann, graph)
        mb_vpn_all = np.zeros(self.n, dtype=bool)
        if graph is not None:
            src = np.asarray(graph["source"], np.int64)
            dst = np.asarray(graph["target"], np.int64)
            vp = ann.superclass == "visual_projection"
            mb_vpn_all[np.unique(src[vp[src] & kc[dst]])] = True
        self.eyes = {side: self._build_eye(types, ann, side, mb_vpn_all) for side in ("L", "R")}
        self.all_photoreceptors = np.concatenate([self.eyes["L"].photoreceptors, self.eyes["R"].photoreceptors])

    def _build_eye(self, types, ann, side, mb_vpn_all) -> Eye:
        on_side = ann.soma_side == side
        loom = np.flatnonzero(np.isin(types, LOOM_TYPES) & on_side).astype(np.int64)
        pr = np.flatnonzero((self.pr_side == side) & (self.pr_hex1 >= 0) & (self.pr_hex2 >= 0)).astype(np.int64)
        h1, h2 = self.pr_hex1[pr], self.pr_hex2[pr]
        extents = (int(h1.min()), int(h1.max()), int(h2.min()), int(h2.max())) if len(pr) else (0, 0, 0, 0)
        mb_vpn = np.flatnonzero(mb_vpn_all & on_side).astype(np.int64)
        return Eye(loom, pr, h1, h2, *extents, mb_vpn)

    @staticmethod
    def geometry(state: GameState) -> tuple[str, float, float, bool]:
        """(eye, |dy|, proximity, approaching). dy < 0 means the ball is above
        the paddle, which is the fly's left. proximity is 1 at the paddle and
        0 at the far wall."""
        half = max(state.field_h / 2.0, 1e-6)
        dy = float(np.clip((state.ball_y - state.paddle_y) / half, -1.0, 1.0))
        eye = "L" if dy < 0 else "R"
        width = max(state.field_w, 1e-6)
        proximity = 1.0 - float(np.clip((state.paddle_x - state.ball_x) / width, 0.0, 1.0))
        return eye, abs(dy), proximity, state.ball_vx > 0

    def drive(self, state: GameState, params: dict) -> np.ndarray:
        drive = np.full(self.n, float(params["background_drive"]), dtype=np.float32)
        eye_name, dy_abs, proximity, approaching = self.geometry(state)
        eye = self.eyes[eye_name]
        gate = 1.0 if approaching else 0.25
        shape = proximity * dy_abs * gate
        if float(params.get("loom_shortcut", 1)) >= 0.5:
            drive[eye.loom] += float(params["loom_strength"]) * shape
        drive[eye.mb_vpn] += float(params.get("mb_strength", 0.0)) * shape
        light = float(params.get("light", 0.0))
        if light > 0 and len(self.all_photoreceptors):
            drive[self.all_photoreceptors] += light
            if len(eye.photoreceptors):
                target1 = round(eye.h1_min + (eye.h1_max - eye.h1_min) * (1.0 - proximity))
                target2 = round(eye.h2_min + (eye.h2_max - eye.h2_min) * dy_abs)
                distance = np.abs(eye.hex1 - target1) + np.abs(eye.hex2 - target2)
                dark = eye.photoreceptors[distance <= int(params["ball_radius_columns"])]
                drive[dark] -= light * float(params.get("ball_contrast", 1.0))
        return drive

    @classmethod
    def from_files(cls, graph_path: Path, annotations_path: Path) -> "SensoryMap":
        g = graph_module.load(graph_path)
        return cls(g["cell_types"], load(annotations_path, g["body_ids"]), g)
