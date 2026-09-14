import numpy as np
import pytest
import torch
from flypong.neurons import ShiuLIF, csr_from_edges


def make(n, edges, **kw):
    src = [e[0] for e in edges]; dst = [e[1] for e in edges]; w = [e[2] for e in edges]
    indptr, target, weight, source = csr_from_edges(n, src, dst, w)
    kw.setdefault("depression_u", 0.0)     # most tests check the published dynamics without depression
    return ShiuLIF(n, indptr, target, weight, source, device="cpu", **kw)


def run(model, drives):
    out = []
    for d in drives:
        out.append(model(torch.as_tensor(d, dtype=torch.float32)).clone())
    return torch.stack(out)


def test_csr_from_edges_sorts_by_source_then_target():
    indptr, target, weight, source = csr_from_edges(3, [2, 0, 0], [1, 2, 1], [1.0, 2.0, 3.0])
    assert indptr.tolist() == [0, 2, 2, 3]
    assert target.tolist() == [1, 2, 1] and weight.tolist() == [3.0, 2.0, 1.0] and source.tolist() == [0, 0, 2]


def test_driven_neuron_spikes_then_is_refractory():
    m = make(1, [])
    spikes = run(m, [[10.0]] * 6)          # 10 mV/ms: crosses threshold every step it is allowed to
    assert spikes[:, 0].tolist() == [True, False, False, True, False, False]   # ref_steps = round(2.2) = 2


def test_spike_arrives_after_the_transmission_delay():
    m = make(2, [(0, 1, 20.0)])
    v1 = []
    for step in range(6):
        m(torch.tensor([10.0 if step == 0 else 0.0, 0.0]))
        v1.append(float(m.v[1]))
    assert v1[0] == pytest.approx(-52.0) and v1[1] == pytest.approx(-52.0)   # nothing for delay_steps = 2
    assert v1[2] > -52.0                                                       # jump on arrival
    assert m.delay_steps == 2 and m.ref_steps == 2


def test_psp_peak_follows_the_original_model():
    # g jumps by w and the membrane integrates g / tau_m: the peak is the
    # double-exponential factor for tau_syn 5 and tau_m 20, about 0.157 * w.
    ts, tm = 5.0, 20.0
    factor = (ts / (tm - ts)) * ((ts / tm) ** (ts / (tm - ts)) - (ts / tm) ** (tm / (tm - ts)))
    assert factor == pytest.approx(0.1575, abs=0.001)
    for w in (3.0, -3.0):
        m = make(2, [(0, 1, w)])
        assert m.psp_peak_factor == pytest.approx(factor, rel=0.05)
        m(torch.tensor([10.0, 0.0]))
        extreme = -52.0
        for _ in range(120):
            m(torch.zeros(2))
            v = float(m.v[1])
            extreme = max(extreme, v) if w > 0 else min(extreme, v)
        assert extreme - (-52.0) == pytest.approx(factor * w, rel=0.06)


def test_short_term_depression_weakens_repeated_spikes_and_recovers():
    def second_psp(u, gap_steps):
        m = make(2, [(0, 1, 30.0)], depression_u=u, tau_rec=300.0)
        peaks = []
        for spike_at in (0, gap_steps):
            pass
        # spike 0 at step 0, spike 1 at step gap_steps; measure the arriving g jump each time
        jumps = []
        for step in range(gap_steps + 3):
            drive = torch.tensor([10.0 if step in (0, gap_steps) else 0.0, 0.0])
            before = float(m.g[1])
            m(drive)
            if step in (2, gap_steps + 2):
                # g_after = (g_before + arriving) * decay_s, so arriving = g_after / decay_s - g_before
                jumps.append(float(m.g[1]) / m.decay_s - before)
        return jumps
    no_std = second_psp(0.0, 6)
    assert no_std[0] == pytest.approx(30.0, rel=1e-4) and no_std[1] == pytest.approx(30.0, rel=1e-4)
    std = second_psp(0.2, 6)
    assert std[0] == pytest.approx(30.0, rel=1e-4)
    assert 30.0 * 0.8 <= std[1] < 30.0 * 0.83            # 20% used, a little recovered over 6 ms
    recovered = second_psp(0.2, 1500)
    assert recovered[1] == pytest.approx(30.0, rel=0.01)   # fully recovered after 1.5 s


def test_refractory_neuron_drops_arriving_input():
    m = make(2, [(0, 1, 50.0)])
    m(torch.tensor([10.0, 10.0]))      # both spike at step 0; neuron 1 is refractory for 2 steps
    m(torch.zeros(2))                  # step 1
    m(torch.zeros(2))                  # step 2: spike from 0 arrives while 1 is still refractory -> dropped
    assert float(m.g[1]) == 0.0


def test_noise_zero_is_deterministic_and_noise_makes_spontaneous_spikes():
    a = make(20, [], noise_mv=0.0); b = make(20, [], noise_mv=0.0)
    ra = run(a, [[0.0] * 20] * 50); rb = run(b, [[0.0] * 20] * 50)
    assert torch.equal(ra, rb) and not ra.any()
    c = make(200, [], noise_mv=3.0, seed=1)
    rc = run(c, [[0.0] * 200] * 500)
    assert 0 < int(rc.sum()) < rc.numel()


def test_last_positions_lists_transmitted_synapses():
    m = make(4, [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 3, 1.0)])
    m(torch.tensor([10.0, 0.0, 0.0, 0.0]))          # step 0: neuron 0 spikes
    assert m.last_positions.numel() == 0
    m(torch.zeros(4))                                # step 1: still in flight
    assert m.last_positions.numel() == 0
    m(torch.zeros(4))                                # step 2: arrives (delay_steps = 2)
    assert m.last_positions.tolist() == [0, 1, 2]
    assert m.delayed_pre.tolist() == [True, False, False, False]


def test_event_driven_matches_dense_reference():
    rng = np.random.default_rng(3)
    n, e = 50, 400
    src = rng.integers(0, n, e); dst = rng.integers(0, n, e); w = rng.normal(0, 4.0, e).astype(np.float32)
    m = make(n, list(zip(src.tolist(), dst.tolist(), w.tolist())))
    W = np.zeros((n, n), np.float32)
    for s, d, ww in zip(src, dst, w):
        W[s, d] += ww
    drives = rng.uniform(0, 1.2, (100, n)).astype(np.float32)
    # dense reference with the same update order
    v = np.full(n, -52.0); g = np.zeros(n); ref = np.zeros(n, int); ring = [np.zeros(n, bool) for _ in range(m.delay_steps)]
    ptr = 0; ref_spikes = []
    for t in range(100):
        pre = ring[ptr]
        arriving = pre.astype(np.float32) @ W
        arriving[ref > 0] = 0.0                      # refractory neurons drop arriving input
        g = g + arriving
        v = -52.0 + (v + 52.0) * m.decay_m + g * m.k_syn + drives[t] * m.dt
        g = g * m.decay_s
        refr = ref > 0
        v[refr] = -52.0; ref[refr] -= 1
        fired = (v >= -45.0) & ~refr
        v[fired] = -52.0; ref[fired] = m.ref_steps
        ring[ptr] = fired; ptr = (ptr + 1) % m.delay_steps
        ref_spikes.append(fired.copy())
    got = run(m, drives).numpy()
    assert np.array_equal(got, np.stack(ref_spikes))
    assert got.sum() > 20


def test_set_dt_rebuilds_steps_and_reset_clears():
    m = make(2, [(0, 1, 5.0)])
    m.set_dt(0.5)
    assert m.ref_steps == 4 and m.delay_steps == 4 and m.ring.shape == (4, 2)
    m(torch.tensor([10.0, 0.0])); m.reset()
    assert float(m.v.max()) == -52.0 and not m.ring.any() and m.ptr == 0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ShiuLIF(2, [0, 1], [5], [1.0])
    m = make(1, [])
    with pytest.raises(ValueError):
        m(torch.zeros(3))
    with pytest.raises(ValueError):
        m.set_dt(0)
