import math
from flypong import config


def test_defaults_has_every_param_with_its_default():
    d = config.defaults()
    assert set(d) == set(config.PARAMS)
    assert d["steps_per_tick"] == 4
    assert d["loom_strength"] == 0.3


def test_clamp_clamps_into_bounds():
    out = config.clamp_params({"steps_per_tick": 99, "motor_gain": -3})
    assert out["steps_per_tick"] == 16
    assert out["motor_gain"] == 0.0


def test_clamp_keeps_in_range_values():
    out = config.clamp_params({"loom_strength": 0.55})
    assert out["loom_strength"] == 0.55


def test_clamp_drops_unknown_and_fills_missing():
    out = config.clamp_params({"bogus": 1})
    assert "bogus" not in out
    assert out == config.defaults()


def test_clamp_ignores_non_numeric_bool_and_nan():
    out = config.clamp_params({"loom_strength": "high", "motor_gain": True, "light": math.nan})
    assert out["loom_strength"] == 0.3
    assert out["motor_gain"] == 0.5
    assert out["light"] == 0.15


def test_clamp_non_dict_returns_defaults():
    assert config.clamp_params(None) == config.defaults()
    assert config.clamp_params([1, 2]) == config.defaults()


def test_learning_params_exist():
    d = config.defaults()
    assert d["learning_enabled"] == 1 and d["learning_rate"] == 0.02
    assert config.clamp_params({"learning_rate": 9})["learning_rate"] == 0.2
    assert config.DIFFUSE_GAIN == 0.3


def test_paths_respect_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYPONG_DATA", str(tmp_path))
    assert config.upstream_graph_path() == tmp_path / "data" / "graph.npz"
    assert config.graph_path() in (config.PROJECT_GRAPH_PATH, tmp_path / "data" / "graph.npz")
    assert config.annotations_path().name == "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    assert config.raw_edges_path().name == "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def test_snap_dt():
    assert config.snap_dt(0.9) == 1.0 and config.snap_dt(0.4) == 0.5 and config.snap_dt(0.3) == 0.25
