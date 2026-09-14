import numpy as np
import pytest
from flypong import config
from flypong.annotations import Annotations
from flypong.senses import GameState, SensoryMap

# index: 0 LC4 L, 1 LC4 R, 2 LPLC2 L, 3 LPLC2 R,
#        4-7 L1 on L eye at hex (1,1) (1,5) (5,1) (5,5), 8 L1 on R eye at (3,3),
#        9 DN L, 10 DN R, 11 Mi1 L (no hex)
TYPES = np.array(["LC4", "LC4", "LPLC2", "LPLC2", "L1", "L1", "L1", "L1", "L1", "DNp02", "DNp02", "Mi1"])
SIDE = np.array(["L", "R", "L", "R", "L", "L", "L", "L", "R", "L", "R", "L"])
SUPER = np.array(["visual_projection"] * 4 + ["ol_intrinsic"] * 5 + ["descending_neuron"] * 2 + ["ol_intrinsic"])
HEX1 = np.array([-1, -1, -1, -1, 1, 1, 5, 5, 3, -1, -1, -1], np.int32)
HEX2 = np.array([-1, -1, -1, -1, 1, 5, 1, 5, 3, -1, -1, -1], np.int32)
ANN = Annotations(SIDE, SUPER, HEX1, HEX2, np.full((12, 3), np.nan, np.float32))
BG = 0.02


@pytest.fixture
def senses():
    return SensoryMap(TYPES, ANN)


def state(ball_y, ball_x=788.0, vx=5.0):
    return GameState(ball_x=ball_x, ball_y=ball_y, ball_vx=vx, ball_vy=0.0,
                     paddle_x=788.0, paddle_y=250.0, field_w=800.0, field_h=500.0)


def test_readout_index_sets(senses):
    assert senses.n == 12
    assert senses.dn_left.tolist() == [9]
    assert senses.dn_right.tolist() == [10]
    assert senses.eyes["L"].loom.tolist() == [0, 2]
    assert senses.eyes["R"].loom.tolist() == [1, 3]
    assert senses.eyes["L"].lamina.tolist() == [4, 5, 6, 7]
    assert (senses.eyes["L"].h1_min, senses.eyes["L"].h1_max, senses.eyes["L"].h2_min, senses.eyes["L"].h2_max) == (1, 5, 1, 5)


def test_drive_has_one_finite_value_per_neuron(senses):
    d = senses.drive(state(100.0), config.defaults())
    assert d.shape == (12,) and d.dtype == np.float32 and np.isfinite(d).all()


def test_ball_above_drives_left_eye_looming_only(senses):
    d = senses.drive(state(0.0), config.defaults())          # dy = -1, at paddle, approaching
    assert d[0] == pytest.approx(BG + 0.3) and d[2] == pytest.approx(BG + 0.3)
    assert d[1] == pytest.approx(BG) and d[3] == pytest.approx(BG)


def test_ball_below_drives_right_eye_looming_only(senses):
    d = senses.drive(state(500.0), config.defaults())        # dy = +1
    assert d[1] == pytest.approx(BG + 0.3) and d[3] == pytest.approx(BG + 0.3)
    assert d[0] == pytest.approx(BG) and d[2] == pytest.approx(BG)


def test_ball_level_gives_no_looming(senses):
    d = senses.drive(state(250.0), config.defaults())
    assert d[[0, 1, 2, 3]].tolist() == pytest.approx([BG] * 4)


def test_looming_scales_with_proximity_and_offset(senses):
    half = senses.drive(state(125.0, ball_x=388.0), config.defaults())   # dy=-0.5, proximity=0.5
    assert half[0] == pytest.approx(BG + 0.3 * 0.5 * 0.5)
    far = senses.drive(state(0.0, ball_x=-12.0), config.defaults())      # proximity 0
    assert far[0] == pytest.approx(BG)


def test_receding_ball_is_gated_to_quarter(senses):
    d = senses.drive(state(0.0, vx=-5.0), config.defaults())
    assert d[0] == pytest.approx(BG + 0.3 * 0.25)


def test_retina_lands_on_ball_column_and_respects_radius(senses):
    p = dict(config.defaults(), ball_radius_columns=0)
    d = senses.drive(state(0.0), p)     # eye L, proximity 1 -> hex1 = 1, dy_abs 1 -> hex2 = 5 -> neuron 5
    assert d[5] == pytest.approx(BG + 0.3)
    assert d[[4, 6, 7, 8]].tolist() == pytest.approx([BG] * 4)
    p["ball_radius_columns"] = 4
    d = senses.drive(state(0.0), p)     # (1,1) and (5,5) are distance 4 from (1,5)
    assert d[[4, 5, 7]].tolist() == pytest.approx([BG + 0.3] * 3)
    assert d[6] == pytest.approx(BG) and d[8] == pytest.approx(BG)


def test_retina_far_ball_maps_to_far_column(senses):
    p = dict(config.defaults(), ball_radius_columns=0)
    d = senses.drive(state(0.0, ball_x=-12.0), p)  # proximity 0 -> hex1 = 5; dy_abs 1 -> hex2 = 5 -> neuron 7
    assert d[7] == pytest.approx(BG + 0.3)


def test_retina_strength_zero_skips_retina(senses):
    d = senses.drive(state(0.0), dict(config.defaults(), retina_strength=0.0))
    assert d[[4, 5, 6, 7, 8]].tolist() == pytest.approx([BG] * 5)


def test_geometry_clips_out_of_field_values(senses):
    eye, dy_abs, prox, approaching = SensoryMap.geometry(state(-999.0, ball_x=5000.0, vx=-1.0))
    assert eye == "L" and dy_abs == 1.0 and prox == 1.0 and approaching is False


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        SensoryMap(TYPES[:5], ANN)
