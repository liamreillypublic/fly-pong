"""Live wrapper around the upstream ConnectomeLIF model.

State persists across ticks; reset() is the only way to clear it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from malecns.model import ConnectomeLIF, resolve_device


class BrainUnstable(RuntimeError):
    """Membrane voltages became nonfinite; the caller should reset the brain."""


@dataclass
class TickResult:
    fired_indices: np.ndarray   # int64 graph indices that fired at least once during the tick
    dn_left: int
    dn_right: int
    total_spikes: int
    wall_ms: float


class FlyBrain:
    def __init__(self, model: ConnectomeLIF, n: int, metadata: dict | None = None):
        self.model = model
        self.n = n
        self.metadata = metadata or {}
        self.device = model.device.type   # "mps" or "cpu", never "mps:0"
        dev = model.device
        self._fired_any = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dn_left = torch.zeros(0, dtype=torch.int64, device=dev)
        self._dn_right = torch.zeros(0, dtype=torch.int64, device=dev)

    @classmethod
    def from_graph(cls, graph_path: Path, device: str = "auto") -> "FlyBrain":
        resolved = resolve_device(device)
        with np.load(graph_path, allow_pickle=False) as g:
            n = len(g["body_ids"])
            metadata = json.loads(str(g["metadata"]))
            model = ConnectomeLIF(n, g["source"], g["target"], g["weight"], device=str(resolved))
        return cls(model, n, metadata)

    @classmethod
    def from_arrays(cls, n: int, source, target, weight, device: str = "cpu") -> "FlyBrain":
        return cls(ConnectomeLIF(n, source, target, weight, device=device), n)

    def set_readout(self, dn_left: np.ndarray, dn_right: np.ndarray) -> None:
        dev = self.model.device
        self._dn_left = torch.as_tensor(np.asarray(dn_left, dtype=np.int64), device=dev)
        self._dn_right = torch.as_tensor(np.asarray(dn_right, dtype=np.int64), device=dev)

    def reset(self) -> None:
        self.model.reset()
        self._fired_any.zero_()

    def tick(self, drive: np.ndarray, k: int) -> TickResult:
        if k < 1:
            raise ValueError("k must be at least 1")
        drive = np.asarray(drive, dtype=np.float32)
        if drive.shape != (self.n,):
            raise ValueError(f"drive must have shape ({self.n},), got {drive.shape}")
        if not np.isfinite(drive).all():
            raise ValueError("drive contains nonfinite values")
        dev = self.model.device
        start = time.perf_counter()
        drive_t = torch.as_tensor(drive, device=dev)
        self._fired_any.zero_()
        left = torch.zeros((), dtype=torch.int32, device=dev)
        right = torch.zeros((), dtype=torch.int32, device=dev)
        total = torch.zeros((), dtype=torch.int32, device=dev)
        with torch.inference_mode():
            for _ in range(k):
                fired = self.model(drive_t).bool()
                self._fired_any |= fired
                left += fired[self._dn_left].sum(dtype=torch.int32)
                right += fired[self._dn_right].sum(dtype=torch.int32)
                total += fired.sum(dtype=torch.int32)
        if not bool(torch.isfinite(self.model.voltage).all()):
            raise BrainUnstable("nonfinite membrane voltage; lower the stimulus strengths")
        fired_idx = torch.nonzero(self._fired_any, as_tuple=False).flatten().cpu().numpy().astype(np.int64)
        wall_ms = (time.perf_counter() - start) * 1000.0
        return TickResult(fired_idx, int(left.item()), int(right.item()), int(total.item()), wall_ms)
