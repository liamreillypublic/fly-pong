"""Game state to per-neuron drive vector. Pure numpy, no torch.

Frame of reference: the fly sits at the right paddle facing left, seen from
above. Its left eye covers the top half of the screen, its right eye the
bottom half.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .annotations import Annotations, load

LOOM_TYPES = ("LC4", "LPLC2")
LAMINA_TYPES = ("L1", "L2", "L3")


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
    loom: np.ndarray    # int64 indices of LC4 + LPLC2 cells on this side
    lamina: np.ndarray  # int64 indices of L1/L2/L3 cells with hex coordinates on this side
    hex1: np.ndarray    # int32, aligned with lamina
    hex2: np.ndarray    # int32, aligned with lamina
    h1_min: int
    h1_max: int
    h2_min: int
    h2_max: int


class SensoryMap:
    def __init__(self, cell_types: np.ndarray, ann: Annotations):
        if len(cell_types) != ann.n:
            raise ValueError("cell_types and annotations length mismatch")
        self.n = len(cell_types)
        self.annotations = ann
        types = np.asarray(cell_types).astype(str)
        dn = ann.superclass == "descending_neuron"
        self.dn_left = np.flatnonzero(dn & (ann.soma_side == "L")).astype(np.int64)
        self.dn_right = np.flatnonzero(dn & (ann.soma_side == "R")).astype(np.int64)
        self.eyes = {side: self._build_eye(types, ann, side) for side in ("L", "R")}

    @staticmethod
    def _build_eye(types: np.ndarray, ann: Annotations, side: str) -> Eye:
        on_side = ann.soma_side == side
        loom = np.flatnonzero(np.isin(types, LOOM_TYPES) & on_side).astype(np.int64)
        lamina_mask = np.isin(types, LAMINA_TYPES) & on_side & (ann.hex1 >= 0) & (ann.hex2 >= 0)
        lamina = np.flatnonzero(lamina_mask).astype(np.int64)
        h1, h2 = ann.hex1[lamina], ann.hex2[lamina]
        if len(lamina):
            extents = (int(h1.min()), int(h1.max()), int(h2.min()), int(h2.max()))
        else:
            extents = (0, 0, 0, 0)
        return Eye(loom, lamina, h1, h2, *extents)

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
        drive[eye.loom] += float(params["loom_strength"]) * proximity * dy_abs * gate
        retina = float(params["retina_strength"])
        if len(eye.lamina) and retina > 0:
            target1 = round(eye.h1_min + (eye.h1_max - eye.h1_min) * (1.0 - proximity))
            target2 = round(eye.h2_min + (eye.h2_max - eye.h2_min) * dy_abs)
            distance = np.abs(eye.hex1 - target1) + np.abs(eye.hex2 - target2)
            drive[eye.lamina[distance <= int(params["ball_radius_columns"])]] += retina
        return drive

    @classmethod
    def from_files(cls, graph_path: Path, annotations_path: Path) -> "SensoryMap":
        with np.load(graph_path, allow_pickle=False) as g:
            body_ids, types = g["body_ids"], g["cell_types"]
        return cls(types, load(annotations_path, body_ids))
