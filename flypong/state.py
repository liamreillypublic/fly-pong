"""Internal state: the part of a fly that a wiring diagram does not contain.

The connectome is the same map whether the fly is starving or full, calm or
frightened. What changes with state is chemistry: neuromodulators and
neuropeptides that scale the sensitivity of particular circuits. This module
keeps a small energy budget and two states, hunger and fear, and turns them
into gains on the circuits whose modulation has been measured in real flies.
The existence of each effect is from the literature; the numbers are ours,
and RULES says so for every one.
"""
from __future__ import annotations

import math

START_ENERGY = 0.8
MEAL = 0.15                 # energy from one taste of sugar
SCARE = 0.3                 # fear added by one punishment (heat, a hit)
FEAR_TAU_MS = 10_000.0      # fear decays with this time constant
FEAR_LOOM_PER_S = 0.5       # fear added per second of full looming

RULES = [
    {"name": "Hunger sharpens sugar taste", "effect": "sugar drive x (1 + hunger)",
     "basis": "Inagaki et al. 2012: dopamine raises the sugar receptor neurons' sensitivity in starved flies",
     "status": "effect measured in flies; the gain size is ours"},
    {"name": "Hunger sharpens the smell of food", "effect": "food-odor drive x (1 + hunger)",
     "basis": "Root et al. 2011: sNPF raises DM1 odor receptor neurons' sensitivity in starved flies",
     "status": "effect measured in flies; the gain size is ours"},
    {"name": "Hunger sets how rewarding sugar is", "effect": "reward dopamine x (0.25 + 0.75 hunger)",
     "basis": "Krashes et al. 2009: NPF and dopamine gate sugar-memory expression by hunger",
     "status": "effect measured in flies; the gain size is ours"},
    {"name": "Fear sharpens looming vision", "effect": "looming drive x (1 + 0.5 fear)",
     "basis": "Suver et al. 2012: octopamine boosts visual motion responses in aroused, flying flies",
     "status": "effect measured in flies; the gain size is ours"},
    {"name": "Energy budget", "effect": f"burns the metabolism setting per brain minute, more when active; a taste of sugar adds {MEAL}",
     "basis": "none", "status": "invented numbers"},
    {"name": "Fear fades", "effect": f"time constant {FEAR_TAU_MS / 1000:.0f} s, a punishment adds {SCARE}",
     "basis": "none", "status": "invented numbers"},
]


class InternalState:
    def __init__(self, energy: float = START_ENERGY):
        self.energy = float(energy)
        self.fear = 0.0
        self.meals = 0
        self.scares = 0
        self.age_ms = 0.0

    @property
    def hunger(self) -> float:
        return min(1.0, max(0.0, 1.0 - self.energy))

    def advance(self, dt_ms: float, metabolism_per_min: float, activity: float = 0.0, loom: float = 0.0) -> None:
        """dt_ms of brain time: burn energy (more when active), gain fear from looming, let fear fade."""
        self.age_ms += dt_ms
        self.energy = max(0.0, self.energy - float(metabolism_per_min) * (1.0 + max(0.0, activity)) * dt_ms / 60_000.0)
        self.fear = min(1.0, self.fear * math.exp(-dt_ms / FEAR_TAU_MS) + FEAR_LOOM_PER_S * max(0.0, min(1.0, loom)) * dt_ms / 1000.0)

    def eat(self, amount: float = MEAL) -> None:
        self.energy = min(1.0, self.energy + amount)
        self.meals += 1

    def scare(self, amount: float = SCARE) -> None:
        self.fear = min(1.0, self.fear + amount)
        self.scares += 1

    def gains(self, enabled: bool = True) -> dict[str, float]:
        if not enabled:
            return {"sugar": 1.0, "odor": 1.0, "reward": 1.0, "loom": 1.0}
        h, f = self.hunger, self.fear
        return {"sugar": 1.0 + h, "odor": 1.0 + h, "reward": 0.25 + 0.75 * h, "loom": 1.0 + 0.5 * f}

    def snapshot(self, enabled: bool = True) -> dict:
        return {"energy": round(self.energy, 3), "hunger": round(self.hunger, 3), "fear": round(self.fear, 3),
                "meals": self.meals, "scares": self.scares, "age_s": round(self.age_ms / 1000.0, 1),
                "gains": {k: round(v, 3) for k, v in self.gains(enabled).items()}, "enabled": bool(enabled)}
