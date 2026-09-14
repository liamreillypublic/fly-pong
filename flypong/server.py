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
    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(static_dir / "index.html")

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
    await ws.send_json({"type": "hello", "neurons": brain.n, "device": brain.device,
                        "defaults": config.defaults()})
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
                params = config.clamp_params(data.get("params"))
                k = int(params["steps_per_tick"])
                state = parse_state(data)
                drive = senses.drive(state, params)
                async with lock:
                    result = await loop.run_in_executor(None, brain.tick, drive, k)
                readout.decay, readout.gain = params["motor_decay"], params["motor_gain"]
                move = readout.update(result.dn_left, result.dn_right)
                await ws.send_json({
                    "type": "command", "move": move,
                    "dn": {"left": result.dn_left, "right": result.dn_right},
                    "spikes": result.fired_indices.tolist(),
                    "stats": {"total_spikes": result.total_spikes, "wall_ms": round(result.wall_ms, 2),
                              "brain_ms": k, "params": params},
                })
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
