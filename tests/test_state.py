import math
import pytest
from flypong.state import FEAR_TAU_MS, MEAL, RULES, SCARE, START_ENERGY, InternalState


def test_energy_burns_with_time_and_activity_and_sugar_refills_it():
    s = InternalState()
    assert s.energy == START_ENERGY and s.hunger == pytest.approx(1 - START_ENERGY)
    s.advance(60_000.0, metabolism_per_min=0.2)                 # one brain minute at rest
    assert s.energy == pytest.approx(START_ENERGY - 0.2)
    s.advance(60_000.0, metabolism_per_min=0.2, activity=1.0)   # active: twice the burn
    assert s.energy == pytest.approx(START_ENERGY - 0.6)
    s.advance(600_000.0, metabolism_per_min=0.2)
    assert s.energy == 0.0 and s.hunger == 1.0                  # never below empty
    s.eat()
    assert s.energy == pytest.approx(MEAL) and s.meals == 1
    for _ in range(20):
        s.eat()
    assert s.energy == 1.0 and s.hunger == 0.0                  # never above full


def test_fear_comes_from_scares_and_looming_and_fades():
    s = InternalState()
    s.scare()
    assert s.fear == pytest.approx(SCARE) and s.scares == 1
    s.advance(FEAR_TAU_MS, metabolism_per_min=0.0)
    assert s.fear == pytest.approx(SCARE * math.exp(-1))
    s = InternalState()
    s.advance(1000.0, metabolism_per_min=0.0, loom=1.0)          # a second of full looming
    assert 0.4 < s.fear < 0.6
    for _ in range(10):
        s.scare()
    assert s.fear == 1.0


def test_gains_follow_hunger_and_fear_and_switch_off():
    s = InternalState(energy=0.5)
    s.scare()
    g = s.gains()
    assert g["sugar"] == pytest.approx(1.5) and g["odor"] == pytest.approx(1.5)
    assert g["reward"] == pytest.approx(0.25 + 0.75 * 0.5) and g["loom"] == pytest.approx(1 + 0.5 * SCARE)
    assert s.gains(enabled=False) == {"sugar": 1.0, "odor": 1.0, "reward": 1.0, "loom": 1.0}
    full = InternalState(energy=1.0).gains()
    assert full["reward"] == pytest.approx(0.25) and full["sugar"] == 1.0    # a full fly barely cares about sugar
    snap = s.snapshot()
    assert snap["energy"] == 0.5 and snap["hunger"] == 0.5 and snap["gains"]["sugar"] == 1.5 and snap["enabled"]


def test_every_rule_says_what_is_measured_and_what_is_invented():
    assert len(RULES) >= 6
    for r in RULES:
        assert r["name"] and r["effect"] and r["basis"] and r["status"]
        assert ("measured" in r["status"]) or ("invented" in r["status"])
