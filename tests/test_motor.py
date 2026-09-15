import pytest
from flypong.motor import MotorReadout


def test_left_spikes_move_up_negative():
    r = MotorReadout(decay=0.7, gain=0.5)
    assert r.update(dn_left=2, dn_right=0) < 0


def test_right_spikes_move_down_positive():
    r = MotorReadout(decay=0.7, gain=0.5)
    assert r.update(dn_left=0, dn_right=2) > 0


def test_balanced_spikes_give_zero():
    r = MotorReadout()
    assert r.update(3, 3) == 0.0


def test_decay_shrinks_command_over_silent_ticks():
    r = MotorReadout(decay=0.5, gain=0.1)
    first = r.update(0, 4)      # accumulator 4 -> move 0.4
    second = r.update(0, 0)     # accumulator 2 -> move 0.2
    third = r.update(0, 0)      # accumulator 1 -> move 0.1
    assert first > second > third > 0
    assert abs(second - 0.2) < 1e-9


def test_command_is_clipped():
    r = MotorReadout(decay=0.7, gain=0.5)
    assert r.update(0, 1000) == 1.0
    r.reset()
    assert r.update(1000, 0) == -1.0


def test_normalized_readout_is_invariant_to_global_excitability():
    quiet, loud = MotorReadout(decay=0.5, gain=0.5, normalize=True), MotorReadout(decay=0.5, gain=0.5, normalize=True)
    for _ in range(300):                       # let the running totals settle at 3 and 30 spikes per tick
        quiet.update(1, 2); loud.update(10, 20)
    assert quiet.update(1, 2) == pytest.approx(loud.update(10, 20), rel=0.02)
    raw_quiet, raw_loud = MotorReadout(decay=0.5, gain=0.05), MotorReadout(decay=0.5, gain=0.05)
    for _ in range(300):
        raw_quiet.update(1, 2); raw_loud.update(10, 20)
    assert raw_loud.update(10, 20) > 5 * raw_quiet.update(1, 2)   # the raw readout scales with activity


def test_reset_zeroes_accumulator():
    r = MotorReadout()
    r.update(0, 5)
    r.reset()
    assert r.accumulator == 0.0
    assert r.update(0, 0) == 0.0
