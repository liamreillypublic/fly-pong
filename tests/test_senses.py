import numpy as np
import pytest
from flypong import config
from flypong.annotations import Annotations
from flypong.senses import GameState, SensoryMap

# index: 0 LC4 L, 1 LC4 R, 2 LPLC2 L, 3 LPLC2 R,
#        4-7 L1 on L eye at hex (1,1) (1,5) (5,1) (5,5), 8 L1 on R eye at (3,3),
#        9 DNp02 L, 10 DNp02 R, 11 Mi1 L,
#        12 R1-R6 L (targets L1 #5 most -> column (1,5)), 13 R1-R6 L (-> #7, (5,5)), 14 R1-R6 R (-> #8, (3,3)),
#        15 R7 L (no lamina target, no column),
#        16 aMe26 L (visual projection onto KC 18), 17 aMe26 R (onto KC 19), 18 KCab-s L, 19 KCg-m R,
#        20 DNge001 L (descending, not an escape DN), 21 motor L, 22 motor R, 23 MBON01 L
TYPES = np.array(["LC4", "LC4", "LPLC2", "LPLC2", "L1", "L1", "L1", "L1", "L1", "DNp02", "DNp02", "Mi1",
                  "R1-R6", "R1-R6", "R1-R6", "R7y", "aMe26", "aMe26", "KCab-s", "KCg-m", "DNge001", "MNleg", "MNleg", "MBON01"])
# photoreceptors 12-14 have unknown soma side ("?"): their eye is inherited from the lamina cell they target
SIDE = np.array(["L", "R", "L", "R", "L", "L", "L", "L", "R", "L", "R", "L",
                 "?", "?", "?", "L", "L", "R", "L", "R", "L", "L", "R", "L"])
SUPER = np.array(["visual_projection"] * 4 + ["ol_intrinsic"] * 5 + ["descending_neuron"] * 2 + ["ol_intrinsic"]
                 + ["ol_sensory"] * 4 + ["visual_projection"] * 2 + ["cb_intrinsic"] * 2
                 + ["descending_neuron", "vnc_motor", "vnc_motor", "cb_intrinsic"])
HEX1 = np.full(24, -1, np.int32); HEX2 = np.full(24, -1, np.int32)
HEX1[4:9] = [1, 1, 5, 5, 3]; HEX2[4:9] = [1, 5, 1, 5, 3]
ANN = Annotations(SIDE, SUPER, HEX1, HEX2, np.full((24, 3), np.nan, np.float32))
GRAPH = {"source": np.array([12, 12, 13, 14, 16, 17, 15]), "target": np.array([5, 4, 7, 8, 18, 19, 11]),
         "count": np.array([10, 2, 3, 4, 2, 2, 1])}
BG = 0.02


@pytest.fixture
def senses():
    return SensoryMap(TYPES, ANN, GRAPH)


def state(ball_y, ball_x=788.0, vx=5.0):
    return GameState(ball_x=ball_x, ball_y=ball_y, ball_vx=vx, ball_vy=0.0,
                     paddle_x=788.0, paddle_y=250.0, field_w=800.0, field_h=500.0)


def P(**over):
    return dict(config.defaults(), **over)


def test_readout_and_monitor_sets(senses):
    assert senses.n == 24
    assert senses.dn_left.tolist() == [9] and senses.dn_right.tolist() == [10]      # DNp only, not DNge001
    assert senses.mn_left.tolist() == [21] and senses.mn_right.tolist() == [22]
    assert senses.monitors["kc"].tolist() == [18, 19] and senses.monitors["mbon"].tolist() == [23]
    assert senses.monitors["lplc2"].tolist() == [2, 3] and senses.monitors["photoreceptors"].tolist() == [12, 13, 14, 15]
    assert senses.eyes["L"].loom.tolist() == [0, 2] and senses.eyes["R"].loom.tolist() == [1, 3]


def test_photoreceptors_inherit_columns_from_their_main_lamina_target(senses):
    L, R = senses.eyes["L"], senses.eyes["R"]
    assert L.photoreceptors.tolist() == [12, 13] and L.hex1.tolist() == [1, 5] and L.hex2.tolist() == [5, 5]
    assert R.photoreceptors.tolist() == [14] and R.hex1.tolist() == [3] and R.hex2.tolist() == [3]
    assert (L.h1_min, L.h1_max, L.h2_min, L.h2_max) == (1, 5, 5, 5)
    assert 15 not in senses.all_photoreceptors.tolist()
    assert senses.pr_side[12] == "L" and senses.pr_side[14] == "R" and senses.pr_side[15] == "L"


def test_mushroom_body_inputs_by_eye(senses):
    assert senses.eyes["L"].mb_vpn.tolist() == [16] and senses.eyes["R"].mb_vpn.tolist() == [17]


def test_eye_columns_and_describe_report_what_the_eyes_get(senses):
    cols = senses.eye_columns()
    assert cols["L"]["columns"] == [[1, 5], [5, 5]] and cols["L"]["h1"] == [1, 5] and cols["L"]["h2"] == [5, 5]
    assert cols["R"]["columns"] == [[3, 3]] and cols["R"]["photoreceptors"] == 1 and cols["L"]["loom_cells"] == 2
    s = senses.describe(state(0.0), P(light=1.0, ball_contrast=0.5, ball_radius_columns=0, loom_strength=0.3, mb_strength=0.1))
    assert s["eye"] == "L" and s["dy"] == -1.0 and s["proximity"] == 1.0 and s["approaching"] is True
    assert s["ball_column"] == [1, 5] and s["dark_photoreceptors"] == 1 and s["contrast"] == 0.5
    assert s["loom"] == {"L": 0.3, "R": 0.0} and s["mb"] == pytest.approx(0.1) and s["light"] == 1.0
    s = senses.describe(state(500.0, vx=-5.0), P(light=0.0, loom_shortcut=0))     # below, moving away, dark, no shortcut
    assert s["eye"] == "R" and s["dy"] == 1.0 and s["approaching"] is False and s["ball_column"] is None
    assert s["loom"] == {"L": 0.0, "R": 0.0} and s["dark_photoreceptors"] == 0


def test_drive_has_one_finite_value_per_neuron(senses):
    d = senses.drive(state(100.0), P())
    assert d.shape == (24,) and d.dtype == np.float32 and np.isfinite(d).all()


def test_light_falls_on_every_columned_photoreceptor_and_the_ball_is_dark(senses):
    d = senses.drive(state(0.0), P(light=1.0, ball_contrast=1.0, ball_radius_columns=0))   # ball above, at paddle
    assert d[12] == pytest.approx(BG)            # column (1,5): the ball's column, dark
    assert d[13] == pytest.approx(BG + 1.0)      # column (5,5): lit
    assert d[14] == pytest.approx(BG + 1.0)      # other eye, lit
    assert d[15] == pytest.approx(BG)            # no column: no light
    d = senses.drive(state(0.0), P(light=1.0, ball_contrast=0.5, ball_radius_columns=0))
    assert d[12] == pytest.approx(BG + 0.5)
    d = senses.drive(state(0.0), P(light=0.0))
    assert d[[12, 13, 14]].tolist() == pytest.approx([BG] * 3)


def test_dark_spot_respects_radius_and_far_column(senses):
    d = senses.drive(state(0.0), P(light=1.0, ball_radius_columns=4))   # (5,5) is distance 4 from (1,5)
    assert d[12] == pytest.approx(BG) and d[13] == pytest.approx(BG)
    d = senses.drive(state(0.0, ball_x=-12.0), P(light=1.0, ball_radius_columns=0))   # far: hex1 = 5 -> (5,5)
    assert d[13] == pytest.approx(BG + 1.0 * 0.0 + 0.0) or d[13] == pytest.approx(BG)
    assert d[12] == pytest.approx(BG + 1.0)


def test_looming_shortcut_can_be_switched_off(senses):
    on = senses.drive(state(0.0), P(loom_shortcut=1, loom_strength=0.3, light=0.0, mb_strength=0.0))
    off = senses.drive(state(0.0), P(loom_shortcut=0, loom_strength=0.3, light=0.0, mb_strength=0.0))
    assert on[0] == pytest.approx(BG + 0.3) and on[2] == pytest.approx(BG + 0.3)
    assert off[0] == pytest.approx(BG) and off[2] == pytest.approx(BG)
    assert on[1] == pytest.approx(BG) and on[3] == pytest.approx(BG)


def test_looming_scales_with_proximity_offset_and_approach(senses):
    half = senses.drive(state(125.0, ball_x=388.0), P(light=0.0, loom_strength=0.3))     # dy=-0.5, proximity=0.5
    assert half[0] == pytest.approx(BG + 0.3 * 0.5 * 0.5)
    level = senses.drive(state(250.0), P(light=0.0, loom_strength=0.3))
    assert level[[0, 1, 2, 3]].tolist() == pytest.approx([BG] * 4)
    receding = senses.drive(state(0.0, vx=-5.0), P(light=0.0, loom_strength=0.3))
    assert receding[0] == pytest.approx(BG + 0.3 * 0.25)


def test_mushroom_body_drive_follows_the_looming_shape(senses):
    d = senses.drive(state(0.0), P(light=0.0, mb_strength=0.3))
    assert d[16] == pytest.approx(BG + 0.3) and d[17] == pytest.approx(BG)
    d = senses.drive(state(500.0), P(light=0.0, mb_strength=0.3))
    assert d[17] == pytest.approx(BG + 0.3) and d[16] == pytest.approx(BG)
    d = senses.drive(state(0.0), P(light=0.0, mb_strength=0.0))
    assert d[16] == pytest.approx(BG)


def test_geometry_clips_out_of_field_values(senses):
    eye, dy_abs, prox, approaching = SensoryMap.geometry(state(-999.0, ball_x=5000.0, vx=-1.0))
    assert eye == "L" and dy_abs == 1.0 and prox == 1.0 and approaching is False


def test_without_graph_there_are_no_columns_and_no_mb_inputs():
    s = SensoryMap(TYPES, ANN)
    assert len(s.all_photoreceptors) == 0 and len(s.eyes["L"].mb_vpn) == 0
    d = s.drive(state(0.0), P(light=1.0))
    assert d[12] == pytest.approx(BG)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        SensoryMap(TYPES[:5], ANN)
