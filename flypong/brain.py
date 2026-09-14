"""Live wrapper: the Shiu et al. neuron model on the project graph, with named
readout and monitor sets and hooks for plasticity.

State persists across ticks; reset() is the only way to clear it. The senses
produce dimensionless drive (threshold-1 units); this wrapper multiplies it by
config.DRIVE_MV to get mV per ms.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from malecns.data import sha256

from . import config
from . import graph as graph_module
from .neurons import ShiuLIF, csr_from_edges
from .plasticity import WeightsUnstable


class BrainUnstable(RuntimeError):
    """Membrane voltages became nonfinite; the caller should reset the brain."""


@dataclass
class TickResult:
    fired_indices: np.ndarray   # int64 graph indices that fired at least once during the tick
    dn_left: int                # escape descending neurons (DNp types), left
    dn_right: int
    total_spikes: int
    wall_ms: float
    learning: dict | None = None
    mn_left: int = 0            # nerve-cord motor neurons, left
    mn_right: int = 0
    monitors: dict = field(default_factory=dict)   # named spike counts this tick
    steps: int = 0


class FlyBrain:
    def __init__(self, model: ShiuLIF, n: int, metadata: dict | None = None, graph_sha: str = ""):
        self.model = model
        self.n = n
        self.metadata = metadata or {}
        self.graph_sha = graph_sha
        self.device = model.device.type   # "mps" or "cpu"
        self.plasticity = None
        dev = model.device
        self._fired_any = torch.zeros(n, dtype=torch.bool, device=dev)
        empty = torch.zeros(0, dtype=torch.int64, device=dev)
        self._sets: dict[str, torch.Tensor] = {"dn_left": empty, "dn_right": empty, "mn_left": empty, "mn_right": empty}
        self._monitors: dict[str, torch.Tensor] = {}

    @classmethod
    def from_graph(cls, graph_path: Path, device: str = "auto", dt_ms: float = 1.0, noise_mv: float = 0.0) -> "FlyBrain":
        if device not in ("auto", "cpu", "mps"):
            raise ValueError("device must be auto, cpu or mps")
        available = torch.backends.mps.is_available()
        if device == "mps" and not available:
            raise RuntimeError("MPS is unavailable; use --device cpu")
        dev = "mps" if (device != "cpu" and available) else "cpu"
        g = graph_module.load(graph_path)
        n = len(g["body_ids"])
        model = ShiuLIF(n, g["indptr"], g["target"], g["weight"], g["source"], device=dev, dt_ms=dt_ms, noise_mv=noise_mv)
        return cls(model, n, g["metadata"], graph_sha=sha256(graph_path))

    @classmethod
    def from_arrays(cls, n: int, source, target, weight, device: str = "cpu", **model_kw) -> "FlyBrain":
        indptr, tgt, w, src = csr_from_edges(n, source, target, weight)
        return cls(ShiuLIF(n, indptr, tgt, w, src, device=device, **model_kw), n)

    # ----- configuration -----
    def _idx(self, values) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values, dtype=np.int64), device=self.model.device)

    def set_readout(self, dn_left, dn_right, mn_left=(), mn_right=()) -> None:
        self._sets = {"dn_left": self._idx(dn_left), "dn_right": self._idx(dn_right),
                      "mn_left": self._idx(mn_left), "mn_right": self._idx(mn_right)}

    def set_monitors(self, **sets) -> None:
        self._monitors = {name: self._idx(values) for name, values in sets.items()}

    def configure(self, dt_ms: float | None = None, noise_mv: float | None = None) -> None:
        if dt_ms is not None and abs(dt_ms - self.model.dt) > 1e-9:
            self.model.set_dt(dt_ms)
        if noise_mv is not None and abs(noise_mv - self.model.noise_mv) > 1e-9:
            self.model.set_noise(noise_mv)

    def model_info(self) -> dict:
        m = self.model
        return {"name": "shiu-lif", "dt_ms": m.dt, "noise_mv": m.noise_mv, "delay_steps": m.delay_steps,
                "ref_steps": m.ref_steps, "psp_peak_factor": round(m.psp_peak_factor, 4)}

    # ----- learning -----
    def attach_plasticity(self, plasticity) -> None:
        self.plasticity = plasticity

    def learning_info(self):
        return None if self.plasticity is None else self.plasticity.info()

    def forget(self) -> None:
        if self.plasticity is not None:
            self.plasticity.forget()

    def save_memory(self, path) -> None:
        if self.plasticity is not None:
            self.plasticity.save(path)

    def load_memory(self, path) -> bool:
        return self.plasticity is not None and self.plasticity.load(path)

    # ----- simulation -----
    def reset(self) -> None:
        self.model.reset()
        self._fired_any.zero_()
        if self.plasticity is not None:
            self.plasticity.reset()

    def tick(self, drive: np.ndarray, k: int, events=(), learning: bool = True, rate: float = 0.02,
             punish_reflex: float = 0.0) -> TickResult:
        if k < 1:
            raise ValueError("k must be at least 1")
        drive = np.asarray(drive, dtype=np.float32)
        if drive.shape != (self.n,):
            raise ValueError(f"drive must have shape ({self.n},), got {drive.shape}")
        if not np.isfinite(drive).all():
            raise ValueError("drive contains nonfinite values")
        dev = self.model.device
        start = time.perf_counter()
        drive_t = torch.as_tensor(drive, device=dev) * config.DRIVE_MV
        self._fired_any.zero_()
        counts = {name: torch.zeros((), dtype=torch.int32, device=dev) for name in (*self._sets, *self._monitors)}
        total = torch.zeros((), dtype=torch.int32, device=dev)
        p = self.plasticity
        if p is not None:
            p.begin_tick(events)
        with torch.inference_mode():
            for _ in range(k):
                extra = None if p is None else p.extra_drive()
                fired = self.model(drive_t if extra is None else drive_t + extra * config.DRIVE_MV)
                if p is not None:
                    p.step(self.model.delayed_pre, fired, self.model.last_positions)
                self._fired_any |= fired
                for name, idx in self._sets.items():
                    counts[name] += fired[idx].sum(dtype=torch.int32)
                for name, idx in self._monitors.items():
                    counts[name] += fired[idx].sum(dtype=torch.int32)
                total += fired.sum(dtype=torch.int32)
        if not bool(torch.isfinite(self.model.v).all()):
            raise BrainUnstable("nonfinite membrane voltage; lower the stimulus strengths")
        fired_idx = torch.nonzero(self._fired_any, as_tuple=False).flatten().cpu().numpy().astype(np.int64)
        wall_ms = (time.perf_counter() - start) * 1000.0
        learning_stats = None
        if p is not None:
            try:
                learning_stats = p.end_tick(learning, rate, wall_ms / 1000.0, k, punish_reflex)
            except WeightsUnstable as e:
                raise BrainUnstable(str(e)) from None
        c = {name: int(v.item()) for name, v in counts.items()}
        return TickResult(fired_idx, c["dn_left"], c["dn_right"], int(total.item()), wall_ms, learning_stats,
                          c["mn_left"], c["mn_right"], {name: c[name] for name in self._monitors}, k)
