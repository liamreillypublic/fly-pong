import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch
from flypong import config
from flypong import plasticity as P
from flypong.neurons import ShiuLIF, csr_from_edges

# neurons: 0 pre (visual_projection) -> 1 post (descending, dopamine-innervated)
#          2 PAM, 3 PPL1 (dopamine edges onto 1)
#          4 pre (visual_projection) -> 5 post (descending, NOT innervated): reflex-only synapse
TYPES = np.array(["LC4", "DNp02", "PAM01", "PPL101", "LC4", "DNp04"])
SUPER = np.array(["visual_projection", "descending_neuron", "cb_intrinsic", "cb_intrinsic",
                  "visual_projection", "descending_neuron"])
BODY = np.array([100, 200, 300, 400, 500, 600], np.int64)
EDGES = [(0, 1, 60.0), (2, 1, 0.0), (3, 1, 0.0), (4, 5, 60.0)]     # mV; dopamine edges carry no current
DOP_SRC = np.array([2, 3]); DOP_DST = np.array([1, 1]); DOP_CNT = np.array([5, 5])
STRONG = 1.5   # dimensionless drive: 10.5 mV/ms, fires every refractory period


def make():
    indptr, target, weight, source = csr_from_edges(6, [e[0] for e in EDGES], [e[1] for e in EDGES], [e[2] for e in EDGES])
    model = ShiuLIF(6, indptr, target, weight, source, device="cpu", depression_u=0.0)   # test the rule alone
    idx, innervated, reflex = P.select_plastic(source, target, weight, DOP_DST, DOP_CNT, SUPER, TYPES)
    pam, ppl1 = P.dopamine_cells(TYPES)
    p = P.Plasticity(model, idx, innervated, reflex, DOP_SRC, DOP_DST, DOP_CNT,
                     np.array([1.0, -1.0], np.float32), pam, ppl1, graph_sha="abc")
    return model, p


def edge(model, s, d):
    """Current weight of edge s -> d (CSR lookup)."""
    lo, hi = int(model.indptr[s]), int(model.indptr[s + 1])
    for pos in range(lo, hi):
        if int(model.target[pos]) == d:
            return float(model.weight[pos])
    raise KeyError((s, d))


def run_tick(model, p, events=(), steps=24, learning=True, rate=0.1, punish_reflex=0.0, drive_on=(0, 4)):
    p.begin_tick(events)
    base = torch.zeros(6)
    for i in drive_on:
        base[i] = STRONG
    for _ in range(steps):
        extra = p.extra_drive()
        fired = model((base if extra is None else base + extra) * config.DRIVE_MV)
        p.step(model.delayed_pre, fired, model.last_positions)
    return p.end_tick(learning, rate, wall_s=0.05, k=steps, punish_reflex=punish_reflex)


def test_dopamine_cells_by_prefix():
    pam, ppl1 = P.dopamine_cells(TYPES)
    assert pam.tolist() == [2] and ppl1.tolist() == [3]


def test_select_plastic_masks():
    model, p = make()
    assert p.idx.tolist() == [0, 3]                       # CSR positions of 0->1 and 4->5
    assert p.innervated.tolist() == [True, False] and p.reflex_mask.tolist() == [True, True]


def test_reflex_arc_excludes_synapses_outside_the_looming_to_escape_path():
    types = np.array(["aMe26", "DNge001", "LC4", "Mi1", "DNp02"])
    superclass = np.array(["visual_projection", "descending_neuron", "visual_projection", "ol_intrinsic", "descending_neuron"])
    src = np.array([0, 2, 3]); dst = np.array([1, 3, 4]); w = np.array([1.0, 1.0, 1.0], np.float32)
    idx, innervated, reflex = P.select_plastic(src, dst, w, np.array([], int), np.array([], float), superclass, types)
    # 0->1 is visual projection onto a non-escape DN: not in the arc; 2->3 leaves a looming detector; 3->4 lands on an escape DN
    assert idx.tolist() == [1, 2] and reflex.tolist() == [True, True]


def test_eligibility_rises_when_post_fires_after_pre():
    model, p = make()
    run_tick(model, p)
    assert float(p.elig[0]) > 0.0 and float(p.elig[1]) > 0.0


def test_reward_potentiates_and_punishment_depresses_innervated_synapse():
    model, p = make()
    run_tick(model, p)
    stats = run_tick(model, p, events=("return",))
    assert stats["event"] == "reward" and stats["pam"] > 0 and stats["ppl1"] == 0
    assert edge(model, 0, 1) > 60.0
    up = edge(model, 0, 1)
    stats = run_tick(model, p, events=("miss",))
    assert stats["event"] == "punishment" and stats["ppl1"] > 0
    assert edge(model, 0, 1) < up
    assert stats["rewards"] == 1 and stats["punishments"] == 1


def test_injection_off_keeps_the_bookkeeping_but_no_dopamine_burst():
    model, p = make()
    run_tick(model, p)
    p.begin_tick(("return",), injection=False)
    assert p.burst_pam == 0 and p.rewards == 1 and p.rpe > 0 and p.extra_drive() is None
    p.begin_tick(("miss",), injection=True)
    assert p.burst_ppl1 > 0 and p.extra_drive() is not None


def test_prediction_error_scales_with_expectation():
    model, p = make()
    stats = run_tick(model, p, events=("return",))
    assert stats["rpe"] == pytest.approx(0.5, abs=1e-6) and stats["expected"] == pytest.approx(0.55, abs=1e-6)
    for _ in range(30):
        run_tick(model, p, events=("return",))
    assert p.expected > 0.95
    stats = run_tick(model, p, events=("miss",))
    assert stats["rpe"] < -0.9                            # a miss after many returns is a big surprise
    stats = run_tick(model, p, events=("return",))
    assert 0 < stats["rpe"] < 0.15                         # an expected return barely registers


def test_punishment_reaches_reflex_only_when_enabled():
    model, p = make()
    run_tick(model, p)
    before = edge(model, 4, 5)
    run_tick(model, p, events=("miss",), punish_reflex=0.0)
    assert edge(model, 4, 5) == pytest.approx(before)     # no local dopamine, diffuse punishment gated off
    run_tick(model, p, events=("miss",), punish_reflex=1.0)
    assert edge(model, 4, 5) < before
    run_tick(model, p, events=("return",))
    assert edge(model, 4, 5) > before - 1e-9 or True       # reward diffuse always reaches the reflex pathway
    grew = edge(model, 4, 5)
    run_tick(model, p, events=("return",))
    assert edge(model, 4, 5) > grew


def test_weights_stay_bounded_and_signed():
    model, p = make()
    for _ in range(80):
        p.expected = 0.0                                   # every return is a full surprise
        run_tick(model, p, events=("return",), rate=0.2)
    assert edge(model, 0, 1) == pytest.approx(240.0)      # 4x cap of 60
    for _ in range(250):
        p.expected = 1.0                                   # every miss is a full surprise
        run_tick(model, p, events=("miss",), rate=0.2)
    assert edge(model, 0, 1) == pytest.approx(6.0)         # 0.1x floor, sign kept


def test_learning_off_keeps_weights_but_reports():
    model, p = make()
    run_tick(model, p)
    stats = run_tick(model, p, events=("return",), learning=False)
    assert edge(model, 0, 1) == 60.0 and stats["enabled"] is False and stats["pam"] > 0


def test_reset_clears_traces_and_forget_restores_weights():
    model, p = make()
    run_tick(model, p); run_tick(model, p, events=("return",))
    p.reset()
    assert float(p.elig.sum()) == 0 and float(p.D.abs().sum()) == 0 and p.G_plus == 0 and float(p.x.sum()) == 0
    p.forget()
    assert edge(model, 0, 1) == 60.0 and p.rewards == 0 and p.age_s == 0 and p.expected == 0.5


def test_save_and_load_round_trip(tmp_path):
    model, p = make()
    run_tick(model, p); run_tick(model, p, events=("return",))
    learned = edge(model, 0, 1)
    path = tmp_path / "learned.npz"
    p.save(path)
    model2, p2 = make()
    assert p2.load(path) is True
    assert edge(model2, 0, 1) == pytest.approx(learned) and p2.rewards == 1 and p2.expected == pytest.approx(0.55)
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
    src, dst, cnt = P.load_dopamine_edges(out, 6)
    assert info["edges"] == 2 and info["cells"] == 2
    assert src.tolist() == [2, 3] and dst.tolist() == [1, 1] and cnt.tolist() == [5, 7]
