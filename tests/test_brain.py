import numpy as np
import pytest
from flypong.brain import BrainUnstable, FlyBrain, TickResult

# 6-neuron chain 0 -> 1 -> 2 -> 3 -> 4 -> 5, strong excitatory weights.
SRC = np.array([0, 1, 2, 3, 4], np.int32)
DST = np.array([1, 2, 3, 4, 5], np.int32)
W = np.full(5, 2.0, np.float32)


@pytest.fixture
def brain():
    b = FlyBrain.from_arrays(6, SRC, DST, W, device="cpu")
    b.set_readout(dn_left=np.array([1]), dn_right=np.array([5]))
    return b


def drive_first():
    d = np.zeros(6, np.float32)
    d[0] = 1.5   # above threshold 1.0 every step
    return d


def test_single_step_fires_only_driven_neuron(brain):
    r = brain.tick(drive_first(), k=1)
    assert isinstance(r, TickResult)
    assert r.fired_indices.tolist() == [0]
    assert r.total_spikes == 1 and r.dn_left == 0 and r.dn_right == 0


def test_spike_propagates_one_hop_per_step_and_union_is_reported(brain):
    r = brain.tick(drive_first(), k=3)
    assert r.fired_indices.tolist() == [0, 1, 2]
    assert r.total_spikes == 6           # 1 + 2 + 3
    assert r.dn_left == 2                # neuron 1 fires on steps 2 and 3
    assert r.dn_right == 0
    assert r.wall_ms >= 0


def test_state_persists_between_ticks(brain):
    brain.tick(drive_first(), k=3)
    r = brain.tick(drive_first(), k=1)   # neuron 3 fires now, plus 0,1,2 keep firing
    assert r.fired_indices.tolist() == [0, 1, 2, 3]


def test_reset_clears_state(brain):
    brain.tick(drive_first(), k=3)
    brain.reset()
    r = brain.tick(drive_first(), k=1)
    assert r.fired_indices.tolist() == [0]


def test_rejects_bad_k_and_shape(brain):
    with pytest.raises(ValueError):
        brain.tick(drive_first(), k=0)
    with pytest.raises(ValueError):
        brain.tick(np.zeros(5, np.float32), k=1)


def test_nonfinite_drive_is_rejected(brain):
    d = drive_first()
    d[2] = np.inf
    with pytest.raises(ValueError):
        brain.tick(d, k=1)


def test_nonfinite_voltage_raises_unstable(brain):
    # Poison the membrane state directly: a NaN voltage never crosses threshold,
    # so it is never reset and must be reported as instability.
    brain.model.voltage[2] = float("nan")
    with pytest.raises(BrainUnstable):
        brain.tick(drive_first(), k=1)


def test_metadata_and_device(brain):
    assert brain.n == 6 and brain.device == "cpu" and brain.metadata == {}
