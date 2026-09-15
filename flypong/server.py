"""aiohttp server: static page plus the /ws game socket.

Protocol (JSON text frames):
  server -> {"type":"hello","neurons":N,"device":"mps","defaults":{...}}
  client -> {"type":"state","ball":{x,y,vx,vy},"paddle":{x,y},"field":{w,h},"params":{...}}
  server -> {"type":"command","move":m,"dn":{"left":a,"right":b},"spikes":[...],"stats":{...}}
  client -> {"type":"reset"}            server -> {"type":"reset_ok"}
  any failure                           server -> {"type":"error","message":"..."}
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
from aiohttp import WSMsgType, web

from . import config
from .brain import BrainUnstable, TickResult
from .motor import MotorReadout
from .senses import GameState

log = logging.getLogger("flypong")


class BrainLike(Protocol):
    n: int
    device: str

    def tick(self, drive: np.ndarray, k: int) -> TickResult: ...
    def reset(self) -> None: ...


class SensesLike(Protocol):
    def drive(self, state: GameState, params: dict) -> np.ndarray: ...


def parse_state(msg: dict) -> GameState:
    try:
        b, p, f = msg["ball"], msg["paddle"], msg["field"]
        return GameState(float(b["x"]), float(b["y"]), float(b["vx"]), float(b["vy"]),
                         float(p["x"]), float(p["y"]), float(f["w"]), float(f["h"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed state: {e}") from None


def create_app(brain: BrainLike, senses: SensesLike, static_dir: Path = config.STATIC_DIR) -> web.Application:
    app = web.Application()
    app["brain"], app["senses"], app["lock"] = brain, senses, asyncio.Lock()
    app["runtime"] = {"driver": None}   # mutable after startup, unlike app[...] itself

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(static_dir / "index.html")

    async def memory_saver(app: web.Application):
        """Save learned weights every SAVE_INTERVAL_S when they changed, and on shutdown."""
        async def loop_save():
            while True:
                await asyncio.sleep(config.SAVE_INTERVAL_S)
                p = getattr(app["brain"], "plasticity", None)
                if p is not None and p.dirty:
                    try:
                        async with app["lock"]:
                            app["brain"].save_memory(config.LEARNED_PATH)
                    except OSError as e:
                        log.warning("memory save failed: %s", e)
        task = asyncio.create_task(loop_save())
        yield
        task.cancel()
        p = getattr(app["brain"], "plasticity", None)
        if p is not None and p.dirty:
            try:
                app["brain"].save_memory(config.LEARNED_PATH)
            except OSError as e:
                log.warning("memory save on shutdown failed: %s", e)

    app.cleanup_ctx.append(memory_saver)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", index)
    app.router.add_static("/static", static_dir)
    return app


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    brain, senses, lock = request.app["brain"], request.app["senses"], request.app["lock"]
    readout = MotorReadout()
    loop = asyncio.get_running_loop()
    learning_info = getattr(brain, "learning_info", lambda: None)()
    model_info = getattr(brain, "model_info", lambda: None)()
    runtime = request.app["runtime"]
    runtime["driver"] = ws   # a new tab takes over; older tabs are told to reload
    await ws.send_json({"type": "hello", "neurons": brain.n, "device": brain.device,
                        "defaults": config.defaults(), "learning": learning_info, "model": model_info})
    async for msg in ws:
        if msg.type != WSMsgType.TEXT:
            continue
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("message must be a JSON object")
            kind = data.get("type")
            if kind == "reset":
                async with lock:
                    brain.reset()
                readout.reset()
                await ws.send_json({"type": "reset_ok"})
            elif kind == "state":
                # One brain, one driver: the newest connection controls the fly.
                driver = runtime["driver"]
                if driver is not None and driver is not ws and not driver.closed:
                    await ws.send_json({"type": "error", "fatal": True,
                                        "message": "another tab is driving the fly; close it or reload this one to take over"})
                    continue
                runtime["driver"] = ws
                params = config.clamp_params(data.get("params"))
                params["dt_ms"] = config.snap_dt(params["dt_ms"])
                dt = params["dt_ms"]
                k = max(1, round(params["steps_per_tick"] / dt))      # steps_per_tick is in brain ms
                state = parse_state(data)
                raw_events = data.get("events") or []
                events = tuple(e for e in raw_events if e in ("return", "miss")) if isinstance(raw_events, list) else ()
                learning = params["learning_enabled"] >= 0.5
                rate = params["learning_rate"]
                punish_reflex = params["punish_reflex"]
                drive = senses.drive(state, params)

                def run_tick():
                    configure = getattr(brain, "configure", None)
                    if configure is not None:
                        configure(dt_ms=dt, noise_mv=params["noise_mv"], depression_u=params["depression_u"])
                    return brain.tick(drive, k, events, learning, rate, punish_reflex)

                async with lock:
                    result = await loop.run_in_executor(None, run_tick)
                readout.decay, readout.gain = params["motor_decay"], params["motor_gain"]
                readout.normalize = params["readout_normalize"] >= 0.5
                if params["readout_motor"] >= 0.5:
                    move = readout.update(result.mn_left, result.mn_right)
                else:
                    move = readout.update(result.dn_left, result.dn_right)
                await ws.send_json({
                    "type": "command", "move": move,
                    "dn": {"left": result.dn_left, "right": result.dn_right},
                    "mn": {"left": result.mn_left, "right": result.mn_right},
                    "spikes": result.fired_indices.tolist(),
                    "stats": {"total_spikes": result.total_spikes, "wall_ms": round(result.wall_ms, 2),
                              "brain_ms": round(k * dt, 3), "steps": k,
                              "spikes_per_step": round(result.total_spikes / k, 1),
                              "monitors": result.monitors, "params": params, "learning": result.learning},
                })
            elif kind == "forget":
                async with lock:
                    brain.forget()
                    if config.LEARNED_PATH.exists():
                        config.LEARNED_PATH.unlink()
                readout.reset()
                await ws.send_json({"type": "forget_ok"})
            else:
                raise ValueError(f"unknown message type: {kind!r}")
        except BrainUnstable as e:
            async with lock:
                brain.reset()
            readout.reset()
            await ws.send_json({"type": "error", "message": f"brain reset: {e}"})
        except ValueError as e:   # includes json.JSONDecodeError
            await ws.send_json({"type": "error", "message": str(e)})
    return ws
