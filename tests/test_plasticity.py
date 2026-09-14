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
