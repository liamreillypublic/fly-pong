import numpy as np
from flypong.annotations import Annotations, from_rows


ROWS = [
    {"bodyId": 20, "somaSide": "R", "superclass": "descending_neuron", "somaLocation": [1, 2, 3],
     "assignedOlHex1": None, "assignedOlHex2": None},
    {"bodyId": 10, "somaSide": "L", "superclass": "ol_intrinsic", "somaLocation": [4.0, 5.0, 6.0],
     "assignedOlHex1": 7.0, "assignedOlHex2": 9},
    {"bodyId": 40, "somaSide": None, "superclass": None, "somaLocation": None,
     "assignedOlHex1": None, "assignedOlHex2": None},
]


def test_rows_are_aligned_to_body_id_order():
    ann = from_rows(ROWS, np.array([10, 20, 30, 40], dtype=np.int64))
    assert ann.n == 4
    assert list(ann.soma_side) == ["L", "R", "?", "?"]
    assert list(ann.superclass) == ["ol_intrinsic", "descending_neuron", "", ""]


def test_hex_coordinates_default_to_minus_one():
    ann = from_rows(ROWS, np.array([10, 20, 30, 40], dtype=np.int64))
    assert ann.hex1.dtype == np.int32
    assert list(ann.hex1) == [7, -1, -1, -1]
    assert list(ann.hex2) == [9, -1, -1, -1]


def test_soma_xyz_is_float32_with_nan_for_missing():
    ann = from_rows(ROWS, np.array([10, 20, 30, 40], dtype=np.int64))
    assert ann.soma_xyz.shape == (4, 3)
    assert ann.soma_xyz.dtype == np.float32
    assert ann.soma_xyz[0].tolist() == [4.0, 5.0, 6.0]
    assert ann.soma_xyz[1].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(ann.soma_xyz[2]).all()
    assert np.isnan(ann.soma_xyz[3]).all()


def test_annotations_is_a_dataclass_with_n():
    ann = Annotations(np.array(["L"]), np.array([""]), np.array([-1], np.int32),
                      np.array([-1], np.int32), np.full((1, 3), np.nan, np.float32))
    assert ann.n == 1
