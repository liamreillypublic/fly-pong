# Dopamine Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The fly learns from reward and punishment through its own synapses, driven by its real PAM and PPL1 dopamine neurons, with memory that survives restarts; play is endless.

**Architecture:** A `Plasticity` object attached to `FlyBrain` keeps eligibility traces on 4.36 M plastic synapses, a per-neuron dopamine trace fed by the real dopamine synapses (rebuilt from the raw synapse file), and a diffuse trace for the reflex pathway. Each tick it applies a bounded, sign-preserving three-factor update and reports drift stats. The server forwards game events (return, miss) as reward and punishment, saves learned weights periodically, and the page grows a learning panel and loses its win condition.

**Tech Stack:** unchanged (Python, torch MPS, numpy, pyarrow, aiohttp; plain JS).

**Spec:** `docs/superpowers/specs/2026-09-14-dopamine-learning-design.md`

## Global Constraints

- Everything in the first plan's Global Constraints still holds.
- Constants (exact): dopamine time constant 30 brain ms, eligibility 100 brain ms, burst 8 steps at +0.3 drive, weight magnitude bounds 0.1x to 4x of original, diffuse gain 0.3, plastic cap 8,000,000, save interval 60 s.
- New params: `learning_enabled` default 1 bounds [0, 1]; `learning_rate` default 0.02 bounds [0, 0.2].
- Reward = the client event `return`; punishment = `miss`. Both may arrive in one tick and both burst.
- Data files live in `/Users/k/Developer/fly-pong/data/` (gitignored): `dopamine.npz`, `learned.npz`.
- `data/learned.npz` is only honored when its stored graph SHA-256 and plastic-index hash match.

## Measured sizing (2026-09-14)

332 dopamine cells (316 PAM, 16 PPL1). Raw scan: 242,503 dopamine edges, 148,999 land on graph neurons, 8,599 distinct targets (4,061 Kenyon cells, 97 MBONs, 74 descending neurons). Plastic union at "any dopamine synapse": 4,356,786 edges. Raw scan time 4.4 s.

## File structure

| File | Change |
|---|---|
| `flypong/config.py` | `DATA_DIR`, `DOPAMINE_PATH`, `LEARNED_PATH`, `raw_edges_path()`, two new params, `DIFFUSE_GAIN` |
| `flypong/plasticity.py` | new: dopamine edge build, plastic selection, `Plasticity` |
| `flypong/brain.py` | attach plasticity; `tick(..., events, learning, rate)`; `TickResult.learning`; forget/save/load pass-through |
| `flypong/server.py` | events, `forget`, hello learning info, periodic + shutdown save |
| `flypong/__main__.py` | build dopamine edges if missing, attach, load memory |
| `static/index.html`, `static/style.css`, `static/app.js` | endless play, learning panel |
| `tests/test_plasticity.py` new; `test_config.py`, `test_brain.py`, `test_server.py` updated | |
| `.gitignore` | add `data/` |
| `README.md` | learning section with measured numbers |

---

### Task 1: Config and plasticity core

**Files:** modify `flypong/config.py`, `.gitignore`; create `flypong/plasticity.py`, `tests/test_plasticity.py`; modify `tests/test_config.py`.

**Interfaces produced:**
- `config.DATA_DIR`, `config.DOPAMINE_PATH`, `config.LEARNED_PATH`, `config.raw_edges_path() -> Path`, `config.DIFFUSE_GAIN = 0.3`, params `learning_enabled`, `learning_rate`.
- `plasticity.dopamine_cells(cell_types) -> (pam_idx, ppl1_idx)`
- `plasticity.build_dopamine_edges(raw_edges_path, body_ids, cell_types, out_path) -> dict`
- `plasticity.load_dopamine_edges(path, n) -> (src, dst, count)`
- `plasticity.select_plastic(source, target, weight, dop_dst, dop_count, superclass, max_plastic=8_000_000) -> (idx int64, innervated bool, reflex bool)`
- `plasticity.Plasticity(model, idx, innervated, reflex, dop_src, dop_dst, dop_count, dop_sign, pam_idx, ppl1_idx, graph_sha)` with `begin_tick(events)`, `extra_drive() -> Tensor | None`, `step(prev_fired, fired)`, `end_tick(learning, rate, wall_s) -> dict`, `reset()`, `forget()`, `save(path)`, `load(path) -> bool`, `info() -> dict`, attributes `dirty`, `last_save_time`.

- [ ] **Step 1: config additions**

Append to `flypong/config.py` after `ANNOTATIONS_FILE`:
```python
RAW_EDGES_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
DATA_DIR = PROJECT_DIR / "data"
DOPAMINE_PATH = DATA_DIR / "dopamine.npz"
LEARNED_PATH = DATA_DIR / "learned.npz"
DIFFUSE_GAIN = 0.3
SAVE_INTERVAL_S = 60.0


def raw_edges_path() -> Path:
    return data_root() / "data" / "raw" / RAW_EDGES_FILE
```
Add to `PARAMS`:
```python
    "learning_enabled": Param(1, 0, 1),
    "learning_rate": Param(0.02, 0.0, 0.2),
```
Add `data/` to `.gitignore`. Add to `tests/test_config.py`:
```python
def test_learning_params_exist():
    d = config.defaults()
    assert d["learning_enabled"] == 1 and d["learning_rate"] == 0.02
    assert config.clamp_params({"learning_rate": 9})["learning_rate"] == 0.2
    assert config.DIFFUSE_GAIN == 0.3
```

- [ ] **Step 2: failing plasticity tests**

`tests/test_plasticity.py`:
```python
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch
from malecns.model import ConnectomeLIF
from flypong import plasticity as P

# neurons: 0 input (visual_projection), 1 post (descending), 2 PAM, 3 PPL1
TYPES = np.array(["LC4", "DNp02", "PAM01", "PPL101"])
SUPER = np.array(["visual_projection", "descending_neuron", "cb_intrinsic", "cb_intrinsic"])
BODY = np.array([100, 200, 300, 400], np.int64)
SRC = np.array([0, 2, 3], np.int32)
DST = np.array([1, 1, 1], np.int32)
W = np.array([2.0, 0.0, 0.0], np.float32)        # dopamine edges carry no current
DOP_SRC = np.array([2, 3]); DOP_DST = np.array([1, 1]); DOP_CNT = np.array([5, 5])


def make():
    model = ConnectomeLIF(4, SRC, DST, W, device="cpu")
    idx, innervated, reflex = P.select_plastic(SRC, DST, W, DOP_DST, DOP_CNT, SUPER)
    pam, ppl1 = P.dopamine_cells(TYPES)
    p = P.Plasticity(model, idx, innervated, reflex, DOP_SRC, DOP_DST, DOP_CNT,
                     np.array([1.0, -1.0], np.float32), pam, ppl1, graph_sha="abc")
    return model, p


def run_tick(model, p, events=(), steps=12, learning=True, rate=0.1):
    p.begin_tick(events)
    prev = model.spikes.bool()
    base = torch.zeros(4); base[0] = 1.5          # neuron 0 fires every step
    for _ in range(steps):
        extra = p.extra_drive()
        fired = model(base if extra is None else base + extra).bool()
        p.step(prev, fired)
        prev = fired
    return p.end_tick(learning, rate, wall_s=0.05)


def test_dopamine_cells_by_prefix():
    pam, ppl1 = P.dopamine_cells(TYPES)
    assert pam.tolist() == [2] and ppl1.tolist() == [3]


def test_select_plastic_excludes_zero_weight_edges():
    idx, innervated, reflex = P.select_plastic(SRC, DST, W, DOP_DST, DOP_CNT, SUPER)
    assert idx.tolist() == [0]
    assert innervated.tolist() == [True] and reflex.tolist() == [True]


def test_eligibility_rises_on_pre_then_post_spikes():
    model, p = make()
    run_tick(model, p)
    assert float(p.elig[0]) > 0.5


def test_reward_potentiates_and_punishment_depresses():
    model, p = make()
    run_tick(model, p)                            # build eligibility
    stats = run_tick(model, p, events=("return",))
    assert stats["event"] == "reward" and stats["pam"] > 0 and stats["ppl1"] == 0
    assert float(model.weight[0]) > 2.0
    up = float(model.weight[0])
    stats = run_tick(model, p, events=("miss",))
    assert stats["event"] == "punishment" and stats["ppl1"] > 0
    assert float(model.weight[0]) < up
    assert stats["rewards"] == 1 and stats["punishments"] == 1


def test_weights_stay_bounded_and_signed():
    model, p = make()
    for _ in range(60):
        run_tick(model, p, events=("return",), rate=0.2)
    assert float(model.weight[0]) == pytest.approx(8.0)       # 4x cap of 2.0
    for _ in range(120):
        run_tick(model, p, events=("miss",), rate=0.2)
    assert float(model.weight[0]) == pytest.approx(0.2)       # 0.1x floor, sign kept


def test_learning_off_keeps_weights_but_reports():
    model, p = make()
    run_tick(model, p)
    stats = run_tick(model, p, events=("return",), learning=False)
    assert float(model.weight[0]) == 2.0 and stats["enabled"] is False and stats["pam"] > 0


def test_reset_clears_traces_and_forget_restores_weights():
    model, p = make()
    run_tick(model, p); run_tick(model, p, events=("return",))
    p.reset()
    assert float(p.elig.sum()) == 0 and float(p.D.abs().sum()) == 0 and p.G == 0
    p.forget()
    assert float(model.weight[0]) == 2.0 and p.rewards == 0 and p.age_s == 0


def test_save_and_load_round_trip(tmp_path):
    model, p = make()
    run_tick(model, p); run_tick(model, p, events=("return",))
    learned = float(model.weight[0])
    path = tmp_path / "learned.npz"
    p.save(path)
    model2, p2 = make()
    assert p2.load(path) is True
    assert float(model2.weight[0]) == pytest.approx(learned) and p2.rewards == 1
    p3 = make()[1]; p3.graph_sha = "other"
    assert p3.load(path) is False


def test_build_dopamine_edges_from_raw_file(tmp_path):
    raw = tmp_path / "raw.feather"
    table = pa.table({"body_pre": pa.array([300, 400, 100, 300], pa.int64()),
                      "body_post": pa.array([200, 200, 200, 999], pa.int64()),
                      "weight": pa.array([5, 7, 1, 3], pa.int64())})
    feather.write_feather(table, str(raw))
    out = tmp_path / "dopamine.npz"
    info = P.build_dopamine_edges(raw, BODY, TYPES, out)
    src, dst, cnt = P.load_dopamine_edges(out, 4)
    assert info["edges"] == 2 and info["cells"] == 2
    assert src.tolist() == [2, 3] and dst.tolist() == [1, 1] and cnt.tolist() == [5, 7]
```

Run: `uv run pytest tests/test_plasticity.py -q` → ModuleNotFoundError.

- [ ] **Step 3: write `flypong/plasticity.py`**

```python
"""Reward-modulated three-factor plasticity driven by the fly's real dopamine neurons.

Reward stimulates the PAM cells, punishment the PPL1 cells. Their spikes
reach the neurons they really innervate (dopamine edges rebuilt from the raw
synapse file) and set a per-neuron dopamine trace D. A diffuse trace G also
reaches the reflex pathway so learning can change play. Plastic synapses keep
an eligibility trace of pre-then-post coincidences; each tick
w += rate * dopamine * eligibility * |w0|, bounded and sign-preserving.
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
TAU_ELIGIBILITY_MS = 100.0
BURST_STEPS = 8
BURST_DRIVE = 0.3
MIN_SCALE, MAX_SCALE = 0.1, 4.0
MAX_PLASTIC = 8_000_000
DRIFT_EVERY = 10


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
    dan_bodies = np.asarray(body_ids)[np.concatenate([pam, ppl1])]
    body_ids = np.asarray(body_ids, dtype=np.int64)
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
            si = np.searchsorted(body_ids, pre); di = np.searchsorted(body_ids, post)
            ok = (si < len(body_ids)) & (di < len(body_ids))
            ok[ok] &= (body_ids[si[ok]] == pre[ok]) & (body_ids[di[ok]] == post[ok])
            srcs.append(si[ok]); dsts.append(di[ok]); cnts.append(cnt[ok])
    src = np.concatenate(srcs).astype(np.int32) if srcs else np.zeros(0, np.int32)
    dst = np.concatenate(dsts).astype(np.int32) if dsts else np.zeros(0, np.int32)
    cnt = np.concatenate(cnts).astype(np.int64) if cnts else np.zeros(0, np.int64)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, src=src, dst=dst, count=cnt, graph_neurons=len(body_ids))
    return {"edges": int(len(src)), "targets": int(len(np.unique(dst))), "cells": int(len(dan_bodies))}


def load_dopamine_edges(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        if int(f["graph_neurons"]) != n:
            raise ValueError("dopamine.npz was built for a different graph; delete it and restart")
        return f["src"].astype(np.int64), f["dst"].astype(np.int64), f["count"].astype(np.float64)


def select_plastic(source, target, weight, dop_dst, dop_count, superclass, max_plastic=MAX_PLASTIC):
    """Plastic edge indices plus two aligned masks: dopamine-innervated target, reflex pathway."""
    source, target, weight = np.asarray(source), np.asarray(target), np.asarray(weight)
    superclass = np.asarray(superclass).astype(str)
    n = len(superclass)
    synapses_per_target = np.bincount(np.asarray(dop_dst), weights=np.asarray(dop_count, dtype=np.float64), minlength=n)
    nonzero = weight != 0
    vp = superclass == "visual_projection"
    dn = superclass == "descending_neuron"
    reflex_all = (vp[source] | dn[target]) & nonzero
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
        self.n = model.voltage.numel()
        self.idx = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=dev)
        self.innervated = torch.as_tensor(np.asarray(innervated, dtype=bool), device=dev)
        self.reflex = torch.as_tensor(np.asarray(reflex, dtype=np.float32), device=dev)
        self.src = model.source[self.idx]
        self.dst = model.target[self.idx]
        self.w0 = model.weight[self.idx].clone()
        self.w0_abs = self.w0.abs()
        self.sign = torch.sign(self.w0)
        self.lo = MIN_SCALE * self.w0_abs
        self.hi = MAX_SCALE * self.w0_abs
        self.elig = torch.zeros(len(self.idx), device=dev)
        self.D = torch.zeros(self.n, device=dev)
        self.G = 0.0
        dop_dst = np.asarray(dop_dst, dtype=np.int64); dop_count = np.asarray(dop_count, dtype=np.float64)
        total = np.bincount(dop_dst, weights=dop_count, minlength=self.n)
        strength = dop_count / np.maximum(total[dop_dst], 1.0) * np.asarray(dop_sign, dtype=np.float64)
        self.dop_src = torch.as_tensor(np.asarray(dop_src, dtype=np.int64), device=dev)
        self.dop_dst = torch.as_tensor(dop_dst, device=dev)
        self.dop_strength = torch.as_tensor(strength.astype(np.float32), device=dev)
        self.pam = torch.as_tensor(np.asarray(pam_idx, dtype=np.int64), device=dev)
        self.ppl1 = torch.as_tensor(np.asarray(ppl1_idx, dtype=np.int64), device=dev)
        self._burst_drive = torch.zeros(self.n, device=dev)
        self.burst_pam = 0; self.burst_ppl1 = 0
        self.decay_d = math.exp(-1.0 / TAU_DOPAMINE_MS)
        self.decay_e = math.exp(-1.0 / TAU_ELIGIBILITY_MS)
        self.graph_sha = graph_sha
        self.rewards = 0; self.punishments = 0; self.age_s = 0.0
        self.dirty = False; self.last_save_time: float | None = None
        self.event = None
        self._ticks = 0
        self._drift = (0.0, 0.0)
        self._pam_spikes = torch.zeros((), dtype=torch.int32, device=dev)
        self._ppl1_spikes = torch.zeros((), dtype=torch.int32, device=dev)

    # ----- per tick -----
    def begin_tick(self, events=()) -> None:
        events = set(events)
        self.event = None
        if "return" in events:
            self.burst_pam = BURST_STEPS; self.G += 1.0; self.rewards += 1; self.event = "reward"
        if "miss" in events:
            self.burst_ppl1 = BURST_STEPS; self.G -= 1.0; self.punishments += 1
            self.event = "punishment" if self.event is None else "both"
        self._pam_spikes.zero_(); self._ppl1_spikes.zero_()

    def extra_drive(self):
        if self.burst_pam <= 0 and self.burst_ppl1 <= 0:
            return None
        self._burst_drive.zero_()
        if self.burst_pam > 0:
            self._burst_drive[self.pam] = BURST_DRIVE; self.burst_pam -= 1
        if self.burst_ppl1 > 0:
            self._burst_drive[self.ppl1] = BURST_DRIVE; self.burst_ppl1 -= 1
        return self._burst_drive

    def step(self, prev_fired, fired) -> None:
        self.D.mul_(self.decay_d)
        self.D.index_add_(0, self.dop_dst, fired[self.dop_src].float() * self.dop_strength)
        self.G *= self.decay_d
        coincidence = (prev_fired[self.src] & fired[self.dst]).float()
        self.elig.mul_(self.decay_e).add_(coincidence).clamp_(max=1.0)
        self._pam_spikes += fired[self.pam].sum(dtype=torch.int32)
        self._ppl1_spikes += fired[self.ppl1].sum(dtype=torch.int32)

    def end_tick(self, learning: bool, rate: float, wall_s: float) -> dict:
        self.age_s += wall_s
        self._ticks += 1
        if learning and rate > 0:
            dopamine = self.D[self.dst] + config.DIFFUSE_GAIN * self.G * self.reflex
            w = self.model.weight[self.idx]
            proposed = w + rate * dopamine * self.elig * self.w0_abs
            magnitude = (proposed * self.sign).clamp(min=self.lo, max=self.hi)
            new_w = magnitude * self.sign
            if not bool(torch.isfinite(new_w).all()):
                raise WeightsUnstable("nonfinite weight update discarded")
            self.model.weight[self.idx] = new_w
            self.dirty = True
        if self._ticks % DRIFT_EVERY == 1:
            rel = (self.model.weight[self.idx] - self.w0).abs() / self.w0_abs.clamp(min=1e-12)
            mb = rel[self.innervated]; rf = rel[self.reflex > 0]
            self._drift = (float(mb.mean()) if mb.numel() else 0.0, float(rf.mean()) if rf.numel() else 0.0)
        return self.stats(learning)

    def stats(self, learning: bool) -> dict:
        return {
            "pam": int(self._pam_spikes.item()), "ppl1": int(self._ppl1_spikes.item()), "event": self.event,
            "mb_drift": self._drift[0], "reflex_drift": self._drift[1],
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
        self.elig.zero_(); self.D.zero_(); self.G = 0.0
        self.burst_pam = self.burst_ppl1 = 0

    def forget(self) -> None:
        self.model.weight[self.idx] = self.w0.clone()
        self.reset()
        self.rewards = self.punishments = 0; self.age_s = 0.0
        self._drift = (0.0, 0.0); self.dirty = True

    def _idx_sha(self) -> str:
        return hashlib.sha256(self.idx.cpu().numpy().tobytes()).hexdigest()

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.part")
        with open(tmp, "wb") as f:
            np.savez(f, graph_sha=self.graph_sha, idx_sha=self._idx_sha(),
                     weights=self.model.weight[self.idx].cpu().numpy().astype(np.float32),
                     rewards=self.rewards, punishments=self.punishments, age_s=self.age_s)
        tmp.replace(path)
        self.dirty = False; self.last_save_time = time.time()

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
            self.rewards = int(f["rewards"]); self.punishments = int(f["punishments"]); self.age_s = float(f["age_s"])
        self.last_save_time = time.time(); self.dirty = False
        return True
```

Run: `uv run pytest tests/test_plasticity.py tests/test_config.py -q` → all pass. Commit: `feat: dopamine-driven three-factor plasticity core`.

---

### Task 2: Brain integration

**Files:** modify `flypong/brain.py`, `tests/test_brain.py`.

**Interfaces produced:** `FlyBrain.attach_plasticity(p)`, `FlyBrain.plasticity`, `FlyBrain.graph_sha: str`, `tick(drive, k, events=(), learning=True, rate=0.02) -> TickResult` where `TickResult.learning: dict | None`, `FlyBrain.forget()`, `save_memory(path)`, `load_memory(path) -> bool`, `learning_info() -> dict | None`.

- [ ] **Step 1: tests** (append to `tests/test_brain.py`)

```python
def test_tick_without_plasticity_reports_none(brain):
    assert brain.tick(drive_first(), k=1).learning is None
    assert brain.learning_info() is None


def test_tick_with_plasticity_learns():
    import numpy as np
    from flypong import plasticity as P
    types = np.array(["LC4", "DNp02", "PAM01", "PPL101"])
    superclass = np.array(["visual_projection", "descending_neuron", "cb_intrinsic", "cb_intrinsic"])
    src = np.array([0, 2, 3], np.int32); dst = np.array([1, 1, 1], np.int32); w = np.array([2.0, 0.0, 0.0], np.float32)
    b = FlyBrain.from_arrays(4, src, dst, w, device="cpu")
    idx, inn, ref = P.select_plastic(src, dst, w, np.array([1, 1]), np.array([5, 5]), superclass)
    pam, ppl1 = P.dopamine_cells(types)
    b.attach_plasticity(P.Plasticity(b.model, idx, inn, ref, np.array([2, 3]), np.array([1, 1]), np.array([5, 5]),
                                     np.array([1.0, -1.0], np.float32), pam, ppl1, graph_sha=b.graph_sha))
    d = np.zeros(4, np.float32); d[0] = 1.5
    b.tick(d, k=12)
    r = b.tick(d, k=12, events=("return",), rate=0.1)
    assert r.learning["event"] == "reward" and r.learning["pam"] > 0
    assert float(b.model.weight[0]) > 2.0
    b.forget()
    assert float(b.model.weight[0]) == 2.0
```

- [ ] **Step 2: implement** in `flypong/brain.py`

Add `graph_sha` (sha256 of the graph file, computed in `from_graph`; `""` for `from_arrays`), `self.plasticity = None`, and:
```python
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
```
`TickResult` gains `learning: dict | None = None`. `tick` signature becomes `tick(self, drive, k, events=(), learning=True, rate=0.02)`; `reset()` also calls `self.plasticity.reset()` when present. Inside the step loop:
```python
        p = self.plasticity
        if p is not None:
            p.begin_tick(events)
        prev = self.model.spikes.bool()
        with torch.inference_mode():
            for _ in range(k):
                extra = None if p is None else p.extra_drive()
                fired = self.model(drive_t if extra is None else drive_t + extra).bool()
                if p is not None:
                    p.step(prev, fired)
                prev = fired
                ... (existing accumulation)
        ...
        wall_ms = (time.perf_counter() - start) * 1000.0
        learning_stats = None
        if p is not None:
            try:
                learning_stats = p.end_tick(learning, rate, wall_ms / 1000.0)
            except WeightsUnstable as e:
                raise BrainUnstable(str(e)) from None
        return TickResult(fired_idx, ..., wall_ms, learning_stats)
```
`from_graph` computes `graph_sha` with `malecns.data.sha256(graph_path)`.

Run `uv run pytest -q` → all pass. Commit: `feat: FlyBrain learns through attached plasticity`.

---

### Task 3: Server, entry point, and memory

**Files:** modify `flypong/server.py`, `flypong/__main__.py`, `tests/test_server.py`.

- [ ] **Step 1: tests** (append to `tests/test_server.py`; extend `FakeBrain` with `events = []`, `forgets = 0`, `tick(self, drive, k, events=(), learning=True, rate=0.02)` recording `(events, learning, rate)`, `forget()` incrementing, `learning_info()` returning `{"plastic": 3}`, `save_memory(path)` no-op)

```python
async def test_events_reach_the_brain_and_learning_stats_are_echoed(aiohttp_client, tmp_path):
    _, ws, hello, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    assert hello["learning"] == {"plastic": 3}
    await ws.send_json(dict(STATE, events=["return"], params={"learning_rate": 0.05, "learning_enabled": 0}))
    cmd = await ws.receive_json()
    assert brain.ticks[-1][2] == ("return",) and brain.ticks[-1][3] is False and brain.ticks[-1][4] == 0.05
    assert cmd["stats"]["learning"] == {"fake": True}
    await ws.close()


async def test_forget(aiohttp_client, tmp_path):
    _, ws, _, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    await ws.send_json({"type": "forget"})
    assert (await ws.receive_json()) == {"type": "forget_ok"}
    assert brain.forgets == 1
    await ws.close()
```
(FakeBrain.tick returns `TickResult(..., learning={"fake": True})`; `ticks.append((drive, k, tuple(events), learning, rate))`.)

- [ ] **Step 2: implement**

`server.py`: in the `state` branch:
```python
                events = tuple(e for e in (data.get("events") or []) if e in ("return", "miss"))
                learning = params["learning_enabled"] >= 0.5
                async with lock:
                    result = await loop.run_in_executor(None, lambda: brain.tick(drive, k, events, learning, params["learning_rate"]))
```
and `"learning": result.learning` inside `stats`. New branch:
```python
            elif kind == "forget":
                async with lock:
                    brain.forget()
                    if config.LEARNED_PATH.exists():
                        config.LEARNED_PATH.unlink()
                await ws.send_json({"type": "forget_ok"})
```
`hello` gains `"learning": brain.learning_info()` (use `getattr(brain, "learning_info", lambda: None)()`).
Periodic save in `create_app`:
```python
    async def saver(app):
        async def loop_save():
            while True:
                await asyncio.sleep(config.SAVE_INTERVAL_S)
                p = getattr(app["brain"], "plasticity", None)
                if p is not None and p.dirty:
                    try:
                        async with app["lock"]:
                            app["brain"].save_memory(config.LEARNED_PATH)
                    except OSError as e:
                        log.warning("save failed: %s", e)
        task = asyncio.create_task(loop_save())
        yield
        task.cancel()
        p = getattr(app["brain"], "plasticity", None)
        if p is not None and p.dirty:
            app["brain"].save_memory(config.LEARNED_PATH)
    app.cleanup_ctx.append(saver)
```
`__main__.py` after loading the brain and senses:
```python
    from .plasticity import Plasticity, build_dopamine_edges, dopamine_cells, load_dopamine_edges, select_plastic
    with np.load(graph, allow_pickle=False) as g:
        source, target, weight, types = g["source"], g["target"], g["weight"], g["cell_types"]
    if not config.DOPAMINE_PATH.exists():
        raw = config.raw_edges_path()
        if not raw.exists():
            print(f"Missing {raw}; run the upstream download."); return 1
        print("Building dopamine edges from the raw synapse file (about 10 s)...", flush=True)
        print(build_dopamine_edges(raw, body_ids, types, config.DOPAMINE_PATH))
    dop_src, dop_dst, dop_cnt = load_dopamine_edges(config.DOPAMINE_PATH, brain.n)
    pam, ppl1 = dopamine_cells(types)
    dop_sign = np.where(np.isin(dop_src, pam), 1.0, -1.0).astype(np.float32)
    idx, innervated, reflex = select_plastic(source, target, weight, dop_dst, dop_cnt, annotations.superclass)
    brain.attach_plasticity(Plasticity(brain.model, idx, innervated, reflex, dop_src, dop_dst, dop_cnt, dop_sign, pam, ppl1, brain.graph_sha))
    loaded = brain.load_memory(config.LEARNED_PATH)
    print(f"Plasticity: {len(idx):,} synapses, {len(pam) + len(ppl1)} dopamine cells; memory {'loaded' if loaded else 'fresh'}", flush=True)
```
(`SensoryMap.from_files` needs to expose annotations: add `self.annotations = ann` in `SensoryMap.__init__` and read `senses.annotations.superclass`; `body_ids` read from the graph.)

Run `uv run pytest -q`, then start the server and confirm the log shows `Plasticity: 4,356,786 synapses, 332 dopamine cells; memory fresh`, and the earlier round-trip script still works with `wall_ms` reported (expect 50 to 70 ms). Commit: `feat: reward and punishment events, memory persistence, forget`.

---

### Task 4: Page: endless play and learning panel

**Files:** modify `static/index.html`, `static/style.css`, `static/app.js`.

- [ ] **Step 1: HTML**
  - Remove `#banner`, `#rematch`, and the "First to 7 wins" span; the statline becomes `Fly returned 0 of 0 · last 20: –`.
  - Add after the escape chart:
```html
    <div class="learn">
      <div class="learn-line"><span id="dopamine">PAM 0 · PPL1 0</span> <span id="dopamine-badge" class="badge"></span></div>
      <div class="learn-line muted" id="memory">mushroom body drift 0.0% · reflex drift 0.0% · rewards 0 · punishments 0 · age 0s</div>
      <canvas id="learn-chart" width="600" height="100"></canvas>
      <div class="controls">
        <label><input type="checkbox" id="learning" checked> Learning (dopamine)</label>
        <label>Learning rate <input type="range" id="lrate" min="0" max="0.2" step="0.005" value="0.02"> <span id="lrate-v">0.02</span></label>
        <button id="forget">Forget everything</button>
      </div>
      <details><summary>What is this?</summary><p>…three-factor rule, PAM/PPL1, mushroom body, the diffuse shortcut, memory file…</p></details>
    </div>
```
  Write the full explanatory paragraph (150 to 220 words) covering: which real cells fire on reward and punishment, where their axons go, the eligibility rule, the bounded update, the diffuse addition, what memory persists, and that direction of movement cannot change, only speed and strength of reaction.

- [ ] **Step 2: CSS** — `.learn { border-top: 1px solid var(--line); margin-top: 8px; padding-top: 8px; }`, `.badge` empty by default, `.badge.reward { color: #6bbf7a; }`, `.badge.punishment { color: #e05d5d; }`, `#learn-chart { height: 100px; }`.

- [ ] **Step 3: JS**
  - Delete `WIN_SCORE`, `endMatch`, `game.over`, the banner and rematch handlers; `frame` steps physics whenever not paused; `sendState` no longer checks `over`.
  - `pendingEvents = []`; push `"return"` on the fly's paddle hit and `"miss"` when the ball passes the right edge; `recent = []` of 1/0 capped at 20; `updateFlyStat` shows lifetime and last-20 rate.
  - `sendState` includes `events: pendingEvents.splice(0)`.
  - `params()` adds `learning_enabled: $("learning").checked ? 1 : 0, learning_rate: parseFloat($("lrate").value)`.
  - `onCommand`: if `cmd.stats.learning`, update `#dopamine`, `#memory` (format age as `Xm YYs`, drift as percent with one decimal, `saved N s ago` when not null), flash the badge for 600 ms on `event`, and push `{event, rate: rolling20}` into `learnHistory` (cap 120) when `event` is set.
  - `renderLearnChart()` called from `frame`: green bars up for reward, red bars down for punishment, white line for the rolling rate (0 at the bottom, 1 at the top).
  - `#forget`: `if (confirm("Erase everything the fly has learned and restore the original connectome?"))` send `{"type":"forget"}`; on `forget_ok` toast "The fly forgot everything", clear `learnHistory`.
  - `hello.learning` appends ` · ${plastic.toLocaleString()} plastic synapses` to the status.

- [ ] **Step 4: verify in the browser** (Playwright): no console errors; bot on; after 60 s: `#memory` shows nonzero drift and rewards+punishments > 0; toggling learning off freezes drift; forget resets counters and drift to 0; stats line shows `ms/tick` still under 80; no banner exists; score keeps climbing past 7.

Commit: `feat: learning panel and endless play`.

---

### Task 5: Measurement and README

- [ ] **Step 1:** From a fresh `Forget`, run the bot at ball speed 5 with learning on for 5 minutes via a Playwright evaluate that samples the lifetime return rate each minute and the drift values. Then turn learning off for 2 minutes and sample. Then Forget and run 2 minutes. Record all numbers.
- [ ] **Step 2:** README: replace the return-rate paragraph with a "Learning" section: the biology, the two additions, the memory file, and a small table of the measured return rates. State plainly whether learning changed play.
- [ ] **Step 3:** `uv run pytest -q`, commit `docs: learning results`, merge to main.

## Self-review

Spec coverage: dopamine delivery (Task 1 `Plasticity.step`, `extra_drive`), plastic set (Task 1 `select_plastic`), eligibility/update/bounds (Task 1 `end_tick`), memory (Task 1 save/load, Task 3 periodic and shutdown save, `forget`), stats (Task 1 `stats`), protocol (Task 3), endless play and panel (Task 4), measurement (Task 5), error handling (WeightsUnstable → BrainUnstable path in Task 2, hash mismatch in `load`, save failure logged in Task 3). Type consistency: `tick(drive, k, events, learning, rate)` matches server lambda, FakeBrain, and tests; `TickResult.learning` used by server; `stats` keys match the page's reads.
