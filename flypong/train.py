"""Headless Fly Pong: the same field, ball, bot and fly readout as the page, run as fast as
the brain allows, to train and measure without a browser.

    uv run python -m flypong.train --probe                # does sugar reach PAM, heat reach PPL1?
    uv run python -m flypong.train --balls 100 300 100    # baseline (learning off), train, test (frozen)

Stop the game server first: two brains on the GPU halve each other's speed.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .brain import FlyBrain
from .motor import MotorReadout
from .senses import GameState, Outcomes, SensoryMap

# field constants, identical to static/app.js
W, H = 800.0, 500.0
PADDLE_W, PADDLE_H, BALL_R = 12.0, 80.0, 8.0
PADDLE_SPEED, BOT_SPEED = 7.0, 4.0
LEFT_X, RIGHT_X = 20.0, W - 20.0 - PADDLE_W
SPEEDUP, MAX_SPEED_FACTOR = 1.03, 2.0


class Pong:
    """The page's stepPhysics(), in Python. The bot plays the left paddle, the fly the right."""

    def __init__(self, base_speed: float = 5.0, seed: int = 0):
        self.rng = random.Random(seed)
        self.base_speed = base_speed
        self.left_y = self.right_y = H / 2
        self.bx = self.by = self.vx = self.vy = 0.0
        self.speed = base_speed
        self.events: list[str] = []
        self.serve(self.rng.choice((-1, 1)))

    def serve(self, direction: int) -> None:
        angle = math.radians(self.rng.uniform(-30.0, 30.0))
        self.speed = self.base_speed
        self.bx, self.by = W / 2, H / 2
        self.vx, self.vy = math.cos(angle) * self.speed * direction, math.sin(angle) * self.speed

    @staticmethod
    def clamp(y: float) -> float:
        return max(PADDLE_H / 2, min(H - PADDLE_H / 2, y))

    def _hit(self, paddle_left_x: float, paddle_y: float, moving_toward: bool) -> bool:
        if not moving_toward:
            return False
        within_x = self.bx + BALL_R >= paddle_left_x and self.bx - BALL_R <= paddle_left_x + PADDLE_W
        within_y = abs(self.by - paddle_y) <= PADDLE_H / 2 + BALL_R
        return within_x and within_y

    def _bounce(self, paddle_y: float, direction: int) -> None:
        offset = (self.by - paddle_y) / (PADDLE_H / 2)
        angle = math.radians(offset * 60.0)
        self.speed = min(self.speed * SPEEDUP, self.base_speed * MAX_SPEED_FACTOR)
        self.vx, self.vy = math.cos(angle) * self.speed * direction, math.sin(angle) * self.speed

    def step(self, fly_move: float) -> None:
        target = self.by if self.vx < 0 else H / 2
        self.left_y = self.clamp(self.left_y + max(-BOT_SPEED, min(BOT_SPEED, target - self.left_y)))
        self.right_y = self.clamp(self.right_y + fly_move * PADDLE_SPEED)
        self.bx += self.vx
        self.by += self.vy
        if self.by - BALL_R < 0:
            self.by, self.vy = BALL_R, abs(self.vy)
        if self.by + BALL_R > H:
            self.by, self.vy = H - BALL_R, -abs(self.vy)
        if self._hit(LEFT_X, self.left_y, self.vx < 0):
            self.bx = LEFT_X + PADDLE_W + BALL_R
            self._bounce(self.left_y, 1)
        if self._hit(RIGHT_X, self.right_y, self.vx > 0):
            self.bx = RIGHT_X - BALL_R
            self._bounce(self.right_y, -1)
            self.events.append("return")
        if self.bx < -BALL_R:
            self.serve(-1)
        if self.bx > W + BALL_R:
            self.events.append("miss")
            self.serve(1)

    def state(self) -> GameState:
        return GameState(ball_x=self.bx, ball_y=self.by, ball_vx=self.vx, ball_vy=self.vy,
                         paddle_x=RIGHT_X, paddle_y=self.right_y, field_w=W, field_h=H)


def load_brain(device: str, learning: bool):
    from .plasticity import Plasticity, build_dopamine_edges, dopamine_cells, load_dopamine_edges, select_plastic
    from . import annotations as ann_module, graph as graph_module
    graph_path = config.graph_path()
    g = graph_module.load(graph_path)
    ann = ann_module.load(config.annotations_path(), g["body_ids"])
    senses = SensoryMap(g["cell_types"], ann, g)
    brain = FlyBrain.from_graph(graph_path, device, dt_ms=1.0, noise_mv=config.defaults()["noise_mv"])
    brain.set_readout(senses.dn_left, senses.dn_right, senses.mn_left, senses.mn_right)
    pam, ppl1 = dopamine_cells(g["cell_types"])
    brain.set_monitors(pam=pam, ppl1=ppl1, taste=senses.taste_mouth, hot=senses.hot, **senses.monitors)
    if learning:
        if not config.DOPAMINE_PATH.exists():
            build_dopamine_edges(config.raw_edges_path(), g["body_ids"], g["cell_types"], config.DOPAMINE_PATH)
        dop_src, dop_dst, dop_cnt = load_dopamine_edges(config.DOPAMINE_PATH, brain.n)
        dop_sign = np.where(np.isin(dop_src, pam), 1.0, -1.0).astype(np.float32)
        idx, innervated, reflex = select_plastic(g["source"], g["target"], g["weight"], dop_dst, dop_cnt, ann.superclass, g["cell_types"])
        brain.attach_plasticity(Plasticity(brain.model, idx, innervated, reflex, dop_src, dop_dst, dop_cnt, dop_sign, pam, ppl1, brain.graph_sha))
    return brain, senses


def probe(brain, senses, params: dict, trials: int = 5, window_ms: int = 300) -> None:
    """Does a taste of sugar make the PAM cells fire, does heat make PPL1 fire, through the fly's own wiring?"""
    idle = np.full(brain.n, params["background_drive"], np.float32)

    def count(drive, ms):
        pam = ppl1 = taste = hot = 0
        for _ in range(ms):
            r = brain.tick(drive, 1, (), False, 0.0, 0.0, False)
            pam += r.monitors["pam"]; ppl1 += r.monitors["ppl1"]; taste += r.monitors["taste"]; hot += r.monitors["hot"]
        return pam, ppl1, taste, hot

    print(f"probe: {len(senses.taste_mouth)} labellar taste neurons, {len(senses.hot)} hot cells, "
          f"{brain.n} neurons; {window_ms} ms windows, {trials} trials each")
    rows = []
    for name, drive_fn in (("nothing", lambda d: d),
                           ("sugar", lambda d: Outcomes.__new__(Outcomes) or d),
                           ("heat", lambda d: d)):
        pass
    base = [count(idle, window_ms) for _ in range(trials)]
    sugar_drive = idle.copy(); sugar_drive[senses.taste_mouth] += params["sugar"]
    heat_drive = idle.copy(); heat_drive[senses.hot] += params["heat"]
    sugar = []
    heat = []
    for _ in range(trials):
        count(idle, 200)
        sugar.append(count(sugar_drive, window_ms))
        count(idle, 200)
        heat.append(count(heat_drive, window_ms))
    for label, rows in (("nothing", base), ("sugar on the mouth", sugar), ("heat on the antennae", heat)):
        a = np.array(rows, dtype=float)
        print(f"  {label:22s}: PAM {a[:, 0].mean():6.1f}  PPL1 {a[:, 1].mean():6.1f}  taste cells {a[:, 2].mean():7.1f}  hot cells {a[:, 3].mean():6.1f}  spikes per {window_ms} ms")


def play(brain, senses, params: dict, balls: int, learning: bool, seed: int, label: str, injection: bool = True) -> dict:
    game = Pong(base_speed=5.0, seed=seed)
    readout = MotorReadout(decay=params["motor_decay"], gain=params["motor_gain"], normalize=params["readout_normalize"] >= 0.5)
    outcomes = Outcomes()
    k = int(params["steps_per_tick"])
    faced = returns = 0
    blocks: list[int] = []
    block_returns = block_faced = 0
    t0 = time.perf_counter()
    brain_ms = 0.0
    move = 0.0
    while faced < balls:
        events = tuple(game.events); game.events.clear()
        outcomes.mark(events, brain_ms)
        drive = outcomes.add_to(senses.drive(game.state(), params), senses, brain_ms, params)
        r = brain.tick(drive, k, events, learning, params["learning_rate"], params["punish_reflex"], injection)
        brain_ms += k
        move = readout.update(r.dn_left, r.dn_right)
        game.step(move)
        for e in events:
            faced += 1; block_faced += 1
            if e == "return":
                returns += 1; block_returns += 1
            if block_faced == 50:
                blocks.append(round(100 * block_returns / block_faced)); block_returns = block_faced = 0
            if faced % 25 == 0:
                print(f"    {label}: {faced} balls, {returns} returned ({100 * returns / faced:.0f}%), "
                      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    rate = returns / max(faced, 1)
    stats = brain.plasticity.stats(learning) if brain.plasticity is not None else {}
    print(f"  {label}: returned {returns} of {faced} ({100 * rate:.0f}%), per 50 balls {blocks}, {(time.perf_counter() - t0) / 60:.1f} min"
          + (f", mushroom body {stats['mb_changed']:,} synapses changed by {100 * stats['mb_drift']:.1f}%, reflex {stats['reflex_changed']:,} by {100 * stats['reflex_drift']:.1f}%" if stats else ""))
    return {"faced": faced, "returns": returns, "rate": rate, "blocks": blocks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="only measure whether sugar and heat reach the dopamine cells")
    ap.add_argument("--balls", type=int, nargs=3, default=[100, 300, 100], metavar=("BASE", "TRAIN", "TEST"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-injection", action="store_true", help="reward and punishment only through the senses")
    ap.add_argument("--fresh", action="store_true", help="start from the original connectome, not data/learned.npz")
    ap.add_argument("--save", action="store_true", help="save the trained weights to data/learned.npz at the end")
    args = ap.parse_args()
    params = config.defaults()
    params["dan_injection"] = 0 if args.no_injection else 1
    t0 = time.perf_counter()
    brain, senses = load_brain(args.device, learning=True)
    if not args.fresh and config.LEARNED_PATH.exists():
        print("loaded", config.LEARNED_PATH if brain.load_memory(config.LEARNED_PATH) else "nothing (learned.npz did not match)")
    print(f"ready in {time.perf_counter() - t0:.0f} s: {brain.n} neurons on {brain.device}, plastic synapses {brain.plasticity.idx.numel():,}")
    if args.probe:
        probe(brain, senses, params)
        return
    base, train, test = args.balls
    injection = not args.no_injection
    print(f"\nbaseline: learning off, {base} balls")
    b = play(brain, senses, params, base, False, 1, "baseline", injection)
    print(f"\ntraining: learning on, {train} balls, dopamine injection {'on' if injection else 'off'}")
    t = play(brain, senses, params, train, True, 2, "training", injection)
    print(f"\ntest: trained weights frozen, {test} balls")
    f = play(brain, senses, params, test, False, 3, "test", injection)
    print(f"\nRESULT baseline {100 * b['rate']:.0f}% -> training {100 * t['rate']:.0f}% -> frozen test {100 * f['rate']:.0f}%")
    if args.save:
        brain.save_memory(config.LEARNED_PATH)
        print("saved", config.LEARNED_PATH)


if __name__ == "__main__":
    main()
