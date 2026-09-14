"""Leaky integrate-and-fire neurons with the Shiu et al. (2024) parameters,
exponential synapses, transmission delay, refractory period, optional noise,
and event-driven propagation over a CSR graph.

Equations follow the published model: an arriving spike adds `weight` (mV,
0.275 per synapse in the project graph) to the synaptic variable g, which
decays with tau_syn; the membrane integrates dv/dt = (g - (v - v_rest)) / tau_m.
One synapse therefore peaks the membrane at about 0.157 * weight, roughly
0.043 mV. Inputs arriving while a neuron is refractory are dropped, as in
the original. `drive` is an external current in mV per ms.

Units: mV and ms.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

V_REST, V_RESET, V_THRESH = -52.0, -52.0, -45.0
TAU_M, TAU_SYN, T_REF, DELAY = 20.0, 5.0, 2.2, 1.8


def csr_from_edges(n: int, source, target, weight):
    """Build (indptr, target, weight, source) sorted by source then target."""
    src, dst, w = np.asarray(source, np.int64), np.asarray(target, np.int64), np.asarray(weight, np.float32)
    if len(src) and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= n):
        raise ValueError("Edge index outside neuron range")
    order = np.lexsort((dst, src))
    src, dst, w = src[order], dst[order], w[order]
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst.astype(np.int32), w, src.astype(np.int32)


class ShiuLIF(nn.Module):
    def __init__(self, n: int, indptr, target, weight, source=None, *, device="cpu", dt_ms: float = 1.0,
                 noise_mv: float = 0.0, seed: int = 0, v_rest=V_REST, v_reset=V_RESET, v_thresh=V_THRESH,
                 tau_m=TAU_M, tau_syn=TAU_SYN, t_ref=T_REF, delay=DELAY):
        super().__init__()
        if n < 1:
            raise ValueError("n must be positive")
        indptr = np.asarray(indptr, np.int64)
        if indptr.shape != (n + 1,) or indptr[0] != 0 or np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must have n + 1 nondecreasing entries starting at 0")
        target, weight = np.asarray(target), np.asarray(weight, np.float32)
        if len(target) != indptr[-1] or len(weight) != len(target):
            raise ValueError("target and weight must have indptr[-1] entries")
        if len(target) and (target.min() < 0 or target.max() >= n):
            raise ValueError("Edge target outside neuron range")
        if not np.isfinite(weight).all():
            raise ValueError("Nonfinite weights")
        dev = torch.device(device)
        self.n = n
        self.v_rest, self.v_reset, self.v_thresh = float(v_rest), float(v_reset), float(v_thresh)
        self.tau_m, self.tau_syn, self.t_ref, self.delay = float(tau_m), float(tau_syn), float(t_ref), float(delay)
        self.register_buffer("indptr", torch.as_tensor(indptr, device=dev))
        self.register_buffer("target", torch.as_tensor(target.astype(np.int32), device=dev))
        self.register_buffer("weight", torch.as_tensor(weight, device=dev))
        self.source = None if source is None else np.asarray(source, np.int32)   # CPU, for plasticity
        self.register_buffer("v", torch.full((n,), self.v_rest, device=dev))
        self.register_buffer("g", torch.zeros(n, device=dev))
        self.register_buffer("ref", torch.zeros(n, dtype=torch.int32, device=dev))
        self.register_buffer("spikes", torch.zeros(n, dtype=torch.bool, device=dev))
        self.noise_mv = float(noise_mv)
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        self._pre = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arriving = torch.zeros(n, device=dev)
        self._last_pos = torch.zeros(0, dtype=torch.int64, device=dev)
        self.ring = None
        self.ptr = 0
        self.set_dt(dt_ms)

    # ----- configuration -----
    @property
    def device(self):
        return self.v.device

    def set_dt(self, dt_ms: float) -> None:
        if not (0 < dt_ms <= self.tau_syn):
            raise ValueError("dt must be positive and no larger than tau_syn")
        self.dt = float(dt_ms)
        self.decay_m = math.exp(-self.dt / self.tau_m)
        self.decay_s = math.exp(-self.dt / self.tau_syn)
        self.ref_steps = max(1, round(self.t_ref / self.dt))
        self.delay_steps = max(1, round(self.delay / self.dt))
        # Exact one-step integration of dv'/dt = (g - v')/tau_m with g decaying at
        # tau_syn: v' <- v' * a_m + g * k_syn, g <- g * a_s. Independent of dt.
        self.k_syn = (self.tau_syn / (self.tau_m - self.tau_syn)) * (self.decay_m - self.decay_s)
        self.psp_peak_factor = self._psp_peak_factor()
        self.ring = torch.zeros(self.delay_steps, self.n, dtype=torch.bool, device=self.device)
        self.ptr = 0

    def _psp_peak_factor(self) -> float:
        """Peak membrane deflection produced by a unit jump in g (about 0.157)."""
        v, g, peak = 0.0, 1.0, 0.0
        for _ in range(int(200 / self.dt)):
            v = v * self.decay_m + g * self.k_syn
            g *= self.decay_s
            peak = max(peak, v)
        return peak

    def set_noise(self, noise_mv: float) -> None:
        if noise_mv < 0 or not math.isfinite(noise_mv):
            raise ValueError("noise must be finite and nonnegative")
        self.noise_mv = float(noise_mv)

    # ----- state -----
    @torch.no_grad()
    def reset(self) -> None:
        self.v.fill_(self.v_rest)
        self.g.zero_()
        self.ref.zero_()
        self.spikes.zero_()
        self.ring.zero_()
        self.ptr = 0
        self._pre.zero_()
        self._last_pos = self._last_pos[:0]

    @property
    def delayed_pre(self) -> torch.Tensor:
        """Presynaptic spikes that transmitted during the last step."""
        return self._pre

    @property
    def last_positions(self) -> torch.Tensor:
        """CSR positions of the synapses that transmitted during the last step."""
        return self._last_pos

    # ----- dynamics -----
    @torch.no_grad()
    def forward(self, drive) -> torch.Tensor:
        drive = torch.as_tensor(drive, dtype=torch.float32, device=self.device)
        if drive.ndim > 1 or (drive.ndim == 1 and drive.shape != (self.n,)):
            raise ValueError("drive must be scalar or one value per neuron")
        pre = self.ring[self.ptr].clone()   # clone: the slot is overwritten at the end of this step
        self._pre = pre
        active = torch.nonzero(pre, as_tuple=False).flatten()
        if active.numel():
            starts = self.indptr[active]
            counts = self.indptr[active + 1] - starts
            total = int(counts.sum().item())
            if total:
                offsets = torch.cumsum(counts, 0) - counts
                pos = torch.repeat_interleave(starts - offsets, counts) + torch.arange(total, device=self.device)
                self._arriving.zero_()
                self._arriving.index_add_(0, self.target[pos].long(), self.weight[pos])
                self._arriving.masked_fill_(self.ref > 0, 0.0)   # refractory neurons drop arriving input
                self.g.add_(self._arriving)
                self._last_pos = pos
            else:
                self._last_pos = self._last_pos[:0]
        else:
            self._last_pos = self._last_pos[:0]
        # membrane update: exact step of dv/dt = (g - (v - v_rest)) / tau_m
        self.v.sub_(self.v_rest).mul_(self.decay_m).add_(self.v_rest)
        self.v.add_(self.g, alpha=self.k_syn)
        self.v.add_(drive * self.dt)
        if self.noise_mv > 0:
            self.v.add_(torch.randn(self.n, device=self.device), alpha=self.noise_mv * math.sqrt(self.dt))
        self.g.mul_(self.decay_s)
        refractory = self.ref > 0
        self.v.masked_fill_(refractory, self.v_reset)
        self.ref.sub_(refractory.to(torch.int32))
        fired = (self.v >= self.v_thresh) & ~refractory
        self.v.masked_fill_(fired, self.v_reset)
        self.ref.masked_fill_(fired, self.ref_steps)
        self.spikes.copy_(fired)
        self.ring[self.ptr] = fired
        self.ptr = (self.ptr + 1) % self.delay_steps
        return self.spikes
