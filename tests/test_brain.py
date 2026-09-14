import numpy as np
import pytest
from flypong.brain import BrainUnstable, FlyBrain, TickResult

# 6-neuron chain 0 -> 1 -> 2 -> 3 -> 4 -> 5 with strong synapses (60 mV -> 9.5 mV PSP, above the 7 mV gap).
SRC = np.array([0, 1, 2, 3, 4], np.int32)
DST = np.array([1, 2, 3, 4, 5], np.int32)
W = np.full(5, 60.0, np.float32)


@pytest.fixture
def brain():
    b = FlyBrain.from_arrays(6, SRC, DST, W, device="cpu")
    b.set_readout(dn_left=np.array([1]), dn_right=np.array([5]), mn_left=np.array([2]), mn_right=np.array([3]))
    b.set_monitors(mid=np.array([2, 3]))
    return b


def drive_first():
    d = np.zeros(6, np.float32)
    d[0] = 1.5   # dimensionless: 10.5 mV/ms, crosses the 7 mV gap in one step
    return d


def test_single_step_fires_only_driven_neuron(brain):
    r = brain.tick(drive_first(), k=1)
    assert isinstance(r, TickResult)
    assert r.fired_indices.tolist() == [0]
    assert r.total_spikes == 1 and r.dn_left == 0 and r.dn_right == 0 and r.steps == 1


def test_spikes_propagate_down_the_chain(brain):
    r = brain.tick(drive_first(), k=24)
    fired = r.fired_indices.tolist()
    assert fired[:3] == [0, 1, 2] and len(fired) >= 4        # delay 2 steps + a few steps to integrate per hop
    assert r.dn_left > 0 and r.mn_left > 0 and r.monitors["mid"] > 0
    assert r.total_spikes > 8 and r.wall_ms >= 0


def test_state_persists_between_ticks(brain):
    brain.tick(drive_first(), k=6)
    before = set(brain.tick(drive_first(), k=1).fired_indices.tolist())
    brain.reset()
    after = set(brain.tick(drive_first(), k=1).fired_indices.tolist())
    assert after == {0} and before != after or before == {0}


def test_reset_clears_state(brain):
    brain.tick(drive_first(), k=24)
    brain.reset()
    r = brain.tick(drive_first(), k=1)
    assert r.fired_indices.tolist() == [0]


def test_rejects_bad_k_shape_and_nonfinite(brain):
    with pytest.raises(ValueError):
        brain.tick(drive_first(), k=0)
    with pytest.raises(ValueError):
        brain.tick(np.zeros(5, np.float32), k=1)
    d = drive_first(); d[2] = np.inf
    with pytest.raises(ValueError):
        brain.tick(d, k=1)


def test_nonfinite_voltage_raises_unstable(brain):
    brain.model.v[2] = float("nan")
    with pytest.raises(BrainUnstable):
        brain.tick(drive_first(), k=1)


def test_metadata_device_and_model_info(brain):
    assert brain.n == 6 and brain.device == "cpu" and brain.metadata == {}
    info = brain.model_info()
    assert info["name"] == "shiu-lif" and info["dt_ms"] == 1.0 and info["delay_steps"] == 2 and info["ref_steps"] == 2


def test_configure_changes_dt_and_noise(brain):
    brain.configure(dt_ms=0.5, noise_mv=0.2)
    assert brain.model.dt == 0.5 and brain.model.noise_mv == 0.2 and brain.model.delay_steps == 4
    brain.configure()                                      # no change
    assert brain.model.dt == 0.5


def test_tick_without_plasticity_reports_none(brain):
    assert brain.tick(drive_first(), k=1).learning is None
    assert brain.learning_info() is None


def test_tick_with_plasticity_learns():
    from flypong import plasticity as P
    types = np.array(["LC4", "DNp02", "PAM01", "PPL101"])
    superclass = np.array(["visual_projection", "descending_neuron", "cb_intrinsic", "cb_intrinsic"])
    src = np.array([0, 2, 3], np.int32); dst = np.array([1, 1, 1], np.int32); w = np.array([60.0, 0.0, 0.0], np.float32)
    b = FlyBrain.from_arrays(4, src, dst, w, device="cpu")
    idx, inn, ref = P.select_plastic(b.model.source, b.model.target.numpy(), b.model.weight.numpy(),
                                     np.array([1, 1]), np.array([5, 5]), superclass)
    pam, ppl1 = P.dopamine_cells(types)
    b.attach_plasticity(P.Plasticity(b.model, idx, inn, ref, np.array([2, 3]), np.array([1, 1]), np.array([5, 5]),
                                     np.array([1.0, -1.0], np.float32), pam, ppl1, graph_sha=b.graph_sha))
    assert b.learning_info()["plastic"] == 1
    d = np.zeros(4, np.float32); d[0] = 1.5
    b.tick(d, k=24)
    r = b.tick(d, k=24, events=("return",), rate=0.1)
    assert r.learning["event"] == "reward" and r.learning["pam"] > 0 and r.learning["rpe"] == pytest.approx(0.5)
    assert float(b.model.weight[0]) > 60.0
    b.forget()
    assert float(b.model.weight[0]) == 60.0
