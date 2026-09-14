import numpy as np
from flypong import atlas
from flypong.annotations import Annotations


def make_ann():
    xyz = np.array([[0, 0, 0], [100, 0, 0], [50, 10, 0], [np.nan, np.nan, np.nan]], np.float32)
    return Annotations(np.array(["L", "R", "L", "?"]),
                       np.array(["ol_intrinsic", "descending_neuron", "made_up", ""]),
                       np.full(4, -1, np.int32), np.full(4, -1, np.int32), xyz)


def test_class_codes_group_superclasses():
    assert atlas.CLASS_NAMES[atlas.class_code("ol_intrinsic")] == "optic lobe"
    assert atlas.CLASS_NAMES[atlas.class_code("descending_neuron")] == "descending"
    assert atlas.CLASS_NAMES[atlas.class_code("vnc_motor")] == "motor"
    assert atlas.CLASS_NAMES[atlas.class_code("nonsense")] == "other"
    assert atlas.CLASS_NAMES[atlas.class_code("")] == "other"


def test_project_spans_long_axis_and_marks_missing():
    xy = atlas.project(make_ann().soma_xyz)
    assert xy.dtype == np.uint16 and xy.shape == (4, 2)
    assert sorted(xy[:3, 0].tolist())[0] == 0 and sorted(xy[:3, 0].tolist())[-1] == 65534
    assert xy[3].tolist() == [atlas.MISSING, atlas.MISSING]
    assert xy[:3, 1].max() < 65534 // 2          # short axis uses the same scale, so it stays small


def test_project_with_fewer_than_two_points_is_all_missing():
    xyz = np.full((3, 3), np.nan, np.float32)
    xyz[0] = [1, 2, 3]
    assert (atlas.project(xyz) == atlas.MISSING).all()


def test_build_and_read_round_trip(tmp_path):
    path = tmp_path / "atlas.bin"
    header = atlas.build(make_ann(), path)
    assert header["neurons"] == 4 and header["classes"] == atlas.CLASS_NAMES
    read_header, xy, codes = atlas.read_atlas(path)
    assert read_header == header
    assert xy.tolist() == atlas.project(make_ann().soma_xyz).tolist()
    assert codes.tolist() == [atlas.class_code("ol_intrinsic"), atlas.class_code("descending_neuron"),
                              atlas.class_code("made_up"), atlas.class_code("")]


def test_header_is_padded_to_four_bytes(tmp_path):
    path = tmp_path / "atlas.bin"
    atlas.build(make_ann(), path)
    data = path.read_bytes()
    hlen = int.from_bytes(data[:4], "little")
    assert hlen % 4 == 0
