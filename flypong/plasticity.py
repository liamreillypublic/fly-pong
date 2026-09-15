"""Reward-modulated three-factor plasticity driven by the fly's real dopamine neurons.

Reward stimulates the PAM cells, punishment the PPL1 cells, each scaled by
prediction error (how surprising the outcome was given the fly's recent
return rate). Their spikes reach the neurons they really innervate (dopamine
edges rebuilt from the raw synapse file) and set a per-neuron dopamine trace
D. Diffuse traces G+ (reward) and G- (punishment) also reach the reflex
pathway so learning can change play; the punishment side is gated by the
`punish_reflex` parameter, default 0.

Eligibility is causal (pre before post): every neuron keeps a presynaptic
trace x that rises when its spikes transmit and decays over TAU_PRE; when a
neuron fires, each plastic synapse onto it gains its presynaptic trace. Each
tick, eligible synapses that received dopamine change by
rate * dopamine * eligibility * |w0|, bounded and sign-preserving.

Edge indices are CSR positions of the project graph.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from . import config

TAU_DOPAMINE_MS = 30.0
# The paddle move that returns or loses a ball happens half a second to a second before the
# outcome. With a 100 ms window (the first three sessions) the credit was gone before the
# dopamine arrived. In the mushroom body the pairing window between Kenyon-cell activity and
# dopamine is seconds long, so the eligibility trace now lasts 1.5 s.
TAU_ELIGIBILITY_MS = 1500.0
TAU_PRE_MS = 20.0
BURST_STEPS_MS = 8.0
BURST_DRIVE = 0.3          # dimensionless, scaled by DRIVE_MV in the brain
MIN_SCALE, MAX_SCALE = 0.1, 4.0
MAX_PLASTIC = 8_000_000
DRIFT_EVERY = 10
EXPECTATION_ALPHA = 0.1


class WeightsUnstable(RuntimeError):
    """A weight update produced nonfinite values and was discarded."""


def dopamine_cells(cell_types) -> tuple[np.ndarray, np.ndarray]:
    types = np.asarray(cell_types).astype(str)
    pam = np.flatnonzero(np.char.startswith(types, "PAM")).astype(np.int64)
    ppl1 = np.flatnonzero(np.char.startswith(types, "PPL1")).astype(np.int64)
    return pam, ppl1


def build_dopamine_edges(raw_edges_path: Path, body_ids: np.ndarray, cell_types, out_path: Path) -> dict:
    """Scan the raw synapse file for edges from PAM/PPL1 cells onto graph neurons."""
    pam, ppl1 = dopamine_cells(cell_types)
    body_ids = np.asarray(body_ids, dtype=np.int64)
    dan_bodies = body_ids[np.concatenate([pam, ppl1])]
    srcs, dsts, cnts = [], [], []
    with pa.memory_map(str(raw_edges_path), "r") as stream:
        reader = ipc.open_file(stream)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            pre = batch.column("body_pre").to_numpy()
            post = batch.column("body_post").to_numpy()
            cnt = batch.column("weight").to_numpy()
            keep = np.isin(pre, dan_bodies)
            if not keep.any():
                continue
            pre, post, cnt = pre[keep], post[keep], cnt[keep]
            si = np.searchsorted(body_ids, pre)
            di = np.searchsorted(body_ids, post)
            ok = (si < len(body_ids)) & (di < len(body_ids))
            ok[ok] &= (body_ids[si[ok]] == pre[ok]) & (body_ids[di[ok]] == post[ok])
            srcs.append(si[ok]); dsts.append(di[ok]); cnts.append(cnt[ok])
    src = np.concatenate(srcs).astype(np.int32) if srcs else np.zeros(0, np.int32)
    dst = np.concatenate(dsts).astype(np.int32) if dsts else np.zeros(0, np.int32)
    cnt = np.concatenate(cnts).astype(np.int64) if cnts else np.zeros(0, np.int64)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, src=src, dst=dst, count=cnt, graph_neurons=len(body_ids))
    return {"edges": int(len(src)), "targets": int(len(np.unique(dst))), "cells": int(len(dan_bodies))}


def load_dopamine_edges(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        if int(f["graph_neurons"]) != n:
            raise ValueError("dopamine.npz was built for a different graph; delete it and restart")
        return f["src"].astype(np.int64), f["dst"].astype(np.int64), f["count"].astype(np.float64)


LOOM_TYPES = ("LC4", "LPLC2")


def select_plastic(source, target, weight, dop_dst, dop_count, superclass, cell_types, max_plastic=MAX_PLASTIC):
    """Plastic edge indices (sorted CSR positions) plus two aligned masks:
    dopamine-innervated target, and the reflex arc (synapses from the looming
    detectors LC4/LPLC2, or onto the DNp escape descending neurons)."""
    source, target, weight = np.asarray(source), np.asarray(target), np.asarray(weight)
    superclass = np.asarray(superclass).astype(str)
    types = np.asarray(cell_types).astype(str)
    n = len(superclass)
    synapses_per_target = np.bincount(np.asarray(dop_dst), weights=np.asarray(dop_count, dtype=np.float64), minlength=n)
    nonzero = weight != 0
    loom = np.isin(types, LOOM_TYPES)
    escape = (superclass == "descending_neuron") & np.char.startswith(types, "DNp")
    reflex_all = (loom[source] | escape[target]) & nonzero
    innervated_all = (synapses_per_target[target] >= 1) & nonzero
    if int((innervated_all | reflex_all).sum()) > max_plastic:
        innervated_all = (synapses_per_target[target] >= 5) & nonzero
    idx = np.flatnonzero(innervated_all | reflex_all).astype(np.int64)
    return idx, innervated_all[idx], reflex_all[idx]


class Plasticity:
    def __init__(self, model, idx, innervated, reflex, dop_src, dop_dst, dop_count, dop_sign,
                 pam_idx, ppl1_idx, graph_sha: str = ""):
        dev = model.device
        self.model = model
        self.n = model.n
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) and np.any(np.diff(idx) <= 0):
            raise ValueError("plastic indices must be strictly increasing")
        self.idx = torch.as_tensor(idx, device=dev)
        self.innervated = torch.as_tensor(np.asarray(innervated, dtype=bool), device=dev)
        self.reflex = torch.as_tensor(np.asarray(reflex, dtype=np.float32), device=dev)
        self.reflex_mask = self.reflex > 0
        source = np.asarray(model.source)
        self.src = torch.as_tensor(source[idx].astype(np.int64), device=dev)
        self.dst = model.target[self.idx].long()
        self.w0 = model.weight[self.idx].clone()
        self.w0_abs = self.w0.abs()
        self.sign = torch.sign(self.w0)
        self.lo = MIN_SCALE * self.w0_abs
        self.hi = MAX_SCALE * self.w0_abs
        self.elig = torch.zeros(len(idx), device=dev)
        self.x = torch.zeros(self.n, device=dev)          # presynaptic trace per neuron
        self.D = torch.zeros(self.n, device=dev)
        self.G_plus = 0.0
        self.G_minus = 0.0
        # incoming-edge index (CSC): CSR positions grouped by target neuron
        target = model.target.cpu().numpy().astype(np.int64)
        order = np.argsort(target, kind="stable")
        indptr_t = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(target, minlength=self.n), out=indptr_t[1:])
        self.indptr_t = torch.as_tensor(indptr_t, device=dev)
        self.csc_pos = torch.as_tensor(order.astype(np.int32), device=dev)
        # dopamine edges normalized per target
        dop_dst = np.asarray(dop_dst, dtype=np.int64)
        dop_count = np.asarray(dop_count, dtype=np.float64)
        total = np.bincount(dop_dst, weights=dop_count, minlength=self.n)
        strength = dop_count / np.maximum(total[dop_dst], 1.0) * np.asarray(dop_sign, dtype=np.float64)
        self.dop_src = torch.as_tensor(np.asarray(dop_src, dtype=np.int64), device=dev)
        self.dop_dst = torch.as_tensor(dop_dst, device=dev)
        self.dop_strength = torch.as_tensor(strength.astype(np.float32), device=dev)
        self.pam = torch.as_tensor(np.asarray(pam_idx, dtype=np.int64), device=dev)
        self.ppl1 = torch.as_tensor(np.asarray(ppl1_idx, dtype=np.int64), device=dev)
        self._burst_drive = torch.zeros(self.n, device=dev)
        self.burst_pam = self.burst_ppl1 = 0
        self.mag_pam = self.mag_ppl1 = 0.0
        self.graph_sha = graph_sha
        self.expected = 0.5
        self.rpe = 0.0
        self.rewards = self.punishments = 0
        self.age_s = 0.0
        self.dirty = False
        self.last_save_time: float | None = None
        self.event = None
        self._ticks = 0
        self._drift = (0.0, 0.0)
        self._changed = (0, 0)
        self._pam_spikes = torch.zeros((), dtype=torch.int32, device=dev)
        self._ppl1_spikes = torch.zeros((), dtype=torch.int32, device=dev)

    # ----- time constants follow the model's dt -----
    def _decays(self) -> tuple[float, float, float]:
        dt = self.model.dt
        return math.exp(-dt / TAU_DOPAMINE_MS), math.exp(-dt / TAU_PRE_MS), math.exp(-dt / TAU_ELIGIBILITY_MS)

    def _burst_steps(self) -> int:
        return max(1, round(BURST_STEPS_MS / self.model.dt))

    # ----- per tick -----
    def begin_tick(self, events=(), injection: bool = True) -> None:
        """injection=False leaves the dopamine cells to the fly's own senses (sugar, heat)
        and only keeps the bookkeeping; the diffuse traces still carry the prediction error."""
        events = set(events)
        self.event = None
        self.rpe = 0.0
        burst = self._burst_steps() if injection else 0
        if "return" in events:
            magnitude = 1.0 - self.expected
            self.expected += EXPECTATION_ALPHA * (1.0 - self.expected)
            self.burst_pam, self.mag_pam = burst, magnitude
            self.G_plus += magnitude
            self.rewards += 1
            self.rpe += magnitude
            self.event = "reward"
        if "miss" in events:
            magnitude = self.expected
            self.expected += EXPECTATION_ALPHA * (0.0 - self.expected)
            self.burst_ppl1, self.mag_ppl1 = burst, magnitude
            self.G_minus += magnitude
            self.punishments += 1
            self.rpe -= magnitude
            self.event = "punishment" if self.event is None else "both"
        self._pam_spikes.zero_()
        self._ppl1_spikes.zero_()

    def extra_drive(self):
        if self.burst_pam <= 0 and self.burst_ppl1 <= 0:
            return None
        self._burst_drive.zero_()
        if self.burst_pam > 0:
            self._burst_drive[self.pam] = BURST_DRIVE * self.mag_pam
            self.burst_pam -= 1
        if self.burst_ppl1 > 0:
            self._burst_drive[self.ppl1] = BURST_DRIVE * self.mag_ppl1
            self.burst_ppl1 -= 1
        return self._burst_drive

    def step(self, pre_transmitted, fired, positions=None) -> None:
        decay_d, decay_x, _ = self._decays()
        self.x.mul_(decay_x).add_(pre_transmitted.float())
        self.D.mul_(decay_d)
        self.D.index_add_(0, self.dop_dst, fired[self.dop_src].float() * self.dop_strength)
        self.D.clamp_(-1.0, 1.0)            # dopamine saturates: at most one full burst's worth per neuron
        self.G_plus = min(1.0, self.G_plus * decay_d)
        self.G_minus = min(1.0, self.G_minus * decay_d)
        post = torch.nonzero(fired, as_tuple=False).flatten()
        if post.numel():
            starts = self.indptr_t[post]
            counts = self.indptr_t[post + 1] - starts
            total = int(counts.sum().item())
            if total:
                offsets = torch.cumsum(counts, 0) - counts
                k = torch.repeat_interleave(starts - offsets, counts) + torch.arange(total, device=fired.device)
                pos = self.csc_pos[k].long()
                r = torch.searchsorted(self.idx, pos).clamp(max=self.idx.numel() - 1)
                ok = self.idx[r] == pos
                pr = r[ok]
                if pr.numel():
                    self.elig.index_add_(0, pr, self.x[self.src[pr]])
        self._pam_spikes += fired[self.pam].sum(dtype=torch.int32)
        self._ppl1_spikes += fired[self.ppl1].sum(dtype=torch.int32)

    def end_tick(self, learning: bool, rate: float, wall_s: float, k: int = 1, punish_reflex: float = 0.0) -> dict:
        self.age_s += wall_s
        self._ticks += 1
        self.elig.clamp_(max=1.0)
        if learning and rate > 0:
            diffuse = config.DIFFUSE_GAIN * (self.G_plus - float(punish_reflex) * self.G_minus)
            dopamine = self.D[self.dst] + diffuse * self.reflex
            w = self.model.weight[self.idx]
            proposed = w + rate * dopamine * self.elig * self.w0_abs
            magnitude = (proposed * self.sign).clamp(min=self.lo, max=self.hi)
            new_w = magnitude * self.sign
            if not bool(torch.isfinite(new_w).all()):
                raise WeightsUnstable("nonfinite weight update discarded")
            self.model.weight[self.idx] = new_w
            self.dirty = True
        _, _, decay_e = self._decays()
        self.elig.mul_(decay_e ** k)
        if self._ticks % DRIFT_EVERY == 1:
            self._recompute_drift()
        return self.stats(learning)

    def _recompute_drift(self) -> None:
        """Mean relative change over the synapses that actually changed, plus their counts."""
        rel = (self.model.weight[self.idx] - self.w0).abs() / self.w0_abs.clamp(min=1e-12)
        changed = rel > 1e-6
        mb_changed = changed & self.innervated
        rf_changed = changed & self.reflex_mask
        mb = rel[mb_changed]
        rf = rel[rf_changed]
        self._drift = (float(mb.mean()) if mb.numel() else 0.0, float(rf.mean()) if rf.numel() else 0.0)
        self._changed = (int(mb_changed.sum().item()), int(rf_changed.sum().item()))

    def stats(self, learning: bool) -> dict:
        return {
            "pam": int(self._pam_spikes.item()), "ppl1": int(self._ppl1_spikes.item()), "event": self.event,
            "expected": round(self.expected, 3), "rpe": round(self.rpe, 3),
            "mb_drift": self._drift[0], "reflex_drift": self._drift[1],
            "mb_changed": self._changed[0], "reflex_changed": self._changed[1],
            "rewards": self.rewards, "punishments": self.punishments, "age_s": round(self.age_s, 1),
            "plastic": int(self.idx.numel()),
            "saved_ago_s": None if self.last_save_time is None else round(time.time() - self.last_save_time, 1),
            "enabled": bool(learning),
        }

    def info(self) -> dict:
        return {"plastic": int(self.idx.numel()), "dopamine_cells": int(self.pam.numel() + self.ppl1.numel()),
                "dopamine_synapses": int(self.dop_dst.numel())}

    # ----- lifecycle -----
    def reset(self) -> None:
        self.elig.zero_()
        self.x.zero_()
        self.D.zero_()
        self.G_plus = self.G_minus = 0.0
        self.burst_pam = self.burst_ppl1 = 0

    def forget(self) -> None:
        self.model.weight[self.idx] = self.w0.clone()
        self.reset()
        self.rewards = self.punishments = 0
        self.age_s = 0.0
        self.expected = 0.5
        self._drift = (0.0, 0.0)
        self._changed = (0, 0)
        self.dirty = True

    def _idx_sha(self) -> str:
        return hashlib.sha256(self.idx.cpu().numpy().tobytes()).hexdigest()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.part")
        with open(tmp, "wb") as f:
            np.savez(f, graph_sha=self.graph_sha, idx_sha=self._idx_sha(),
                     weights=self.model.weight[self.idx].cpu().numpy().astype(np.float32),
                     rewards=self.rewards, punishments=self.punishments, age_s=self.age_s, expected=self.expected)
        tmp.replace(path)
        self.dirty = False
        self.last_save_time = time.time()

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as f:
            if str(f["graph_sha"]) != self.graph_sha or str(f["idx_sha"]) != self._idx_sha():
                return False
            weights = f["weights"]
            if weights.shape != (self.idx.numel(),):
                return False
            self.model.weight[self.idx] = torch.as_tensor(weights, device=self.model.device)
            self.rewards = int(f["rewards"])
            self.punishments = int(f["punishments"])
            self.age_s = float(f["age_s"])
            self.expected = float(f["expected"]) if "expected" in f.files else 0.5
        self.last_save_time = time.time()
        self.dirty = False
        return True
