import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
from flypong import graph


@pytest.fixture
def raw(tmp_path):
    ann = pa.table({"bodyId": pa.array([30, 10, 20, 40], pa.int64()),
                    "superclass": pa.array(["glia", "cb_intrinsic", "ol_intrinsic", None]),
                    "type": pa.array(["G", "A", "R1-R6", None])})
    nt = pa.table({"body": pa.array([10, 20, 30], pa.int64()),
                   "consensus_nt": pa.array(["histamine", "dopamine", "gaba"])})
    edges = pa.table({"body_pre": pa.array([10, 20, 10, 20, 40], pa.int64()),
                      "body_post": pa.array([20, 10, 30, 20, 10], pa.int64()),
                      "weight": pa.array([4, 2, 5, 1, 9], pa.int64())})
    paths = {}
    for name, table in [("ann", ann), ("nt", nt), ("edges", edges)]:
        paths[name] = tmp_path / f"{name}.feather"
        feather.write_feather(table, str(paths[name]))
    return paths


def test_select_neurons_matches_upstream_rule(raw):
    ids, types = graph.select_neurons(raw["ann"])
    assert ids.tolist() == [10, 20] and types.tolist() == ["A", "R1-R6"]


def test_signs_include_histamine_and_exclude_modulators(raw):
    ids, _ = graph.select_neurons(raw["ann"])
    signs, labels = graph.neuron_signs(raw["nt"], ids)
    assert signs.tolist() == [-1.0, 0.0] and labels.tolist() == ["histamine", "dopamine"]


def test_build_writes_csr_with_absolute_weights(raw, tmp_path):
    out = tmp_path / "graph.npz"
    meta = graph.build(raw["ann"], raw["nt"], raw["edges"], out)
    g = graph.load(out)
    assert g["body_ids"].tolist() == [10, 20]
    assert g["indptr"].tolist() == [0, 1, 3]
    assert g["source"].tolist() == [0, 1, 1] and g["target"].tolist() == [1, 0, 1]
    assert g["count"].tolist() == [4, 2, 1]
    assert g["weight"].tolist() == pytest.approx([-0.275 * 4, 0.0, 0.0])
    assert meta["edges"] == 3 and meta["synaptic_contacts"] == 7 and meta["histamine_neurons"] == 1
    assert meta["layout"] == "csr-by-source" and g["metadata"]["signs"]["histamine"] == -1.0


def test_load_rejects_non_csr_graph(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, metadata='{"format_version": 1}')
    with pytest.raises(ValueError):
        graph.load(path)
