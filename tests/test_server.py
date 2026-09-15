import numpy as np
from flypong.brain import BrainUnstable, TickResult
from flypong.server import create_app


class FakeBrain:
    n = 5
    device = "cpu"
    plasticity = None

    def __init__(self, unstable=False):
        self.resets = 0
        self.forgets = 0
        self.ticks = []
        self.unstable = unstable

    def tick(self, drive, k, events=(), learning=True, rate=0.02, punish_reflex=0.0):
        self.ticks.append((np.array(drive), k, tuple(events), learning, rate, punish_reflex))
        if self.unstable:
            raise BrainUnstable("boom")
        return TickResult(np.array([1, 3], np.int64), dn_left=2, dn_right=0, total_spikes=4, wall_ms=1.5,
                          learning={"fake": True}, mn_left=0, mn_right=5, monitors={"kc": 7}, steps=k)

    def configure(self, dt_ms=None, noise_mv=None, depression_u=None):
        self.configured = (dt_ms, noise_mv, depression_u)

    def model_info(self):
        return {"name": "fake"}

    def reset(self):
        self.resets += 1

    def forget(self):
        self.forgets += 1

    def learning_info(self):
        return {"plastic": 3}

    def save_memory(self, path):
        pass


class FakeSenses:
    def drive(self, state, params):
        return np.full(5, params["background_drive"], np.float32)


STATE = {"type": "state", "ball": {"x": 700, "y": 100, "vx": 5, "vy": 0},
         "paddle": {"x": 788, "y": 250}, "field": {"w": 800, "h": 500},
         "params": {"steps_per_tick": 6}}


async def connect(aiohttp_client, brain=None, tmp_path=None):
    brain = brain or FakeBrain()
    static = tmp_path
    (static / "index.html").write_text("<h1>ok</h1>")
    client = await aiohttp_client(create_app(brain, FakeSenses(), static_dir=static))
    ws = await client.ws_connect("/ws")
    hello = await ws.receive_json()
    return client, ws, hello, brain


async def test_hello_on_connect(aiohttp_client, tmp_path):
    _, ws, hello, _ = await connect(aiohttp_client, tmp_path=tmp_path)
    assert hello["type"] == "hello" and hello["neurons"] == 5 and hello["device"] == "cpu"
    assert hello["defaults"]["steps_per_tick"] == 4
    await ws.close()


async def test_state_returns_command(aiohttp_client, tmp_path):
    _, ws, _, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    await ws.send_json(STATE)
    cmd = await ws.receive_json()
    assert cmd["type"] == "command"
    assert cmd["move"] < 0                       # left DN spikes move up
    assert cmd["dn"] == {"left": 2, "right": 0}
    assert cmd["spikes"] == [1, 3]
    assert cmd["stats"]["total_spikes"] == 4 and cmd["stats"]["brain_ms"] == 6
    assert cmd["stats"]["params"]["steps_per_tick"] == 6
    assert brain.ticks[0][1] == 6
    await ws.close()


async def test_params_are_clamped_and_echoed(aiohttp_client, tmp_path):
    _, ws, _, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    await ws.send_json(dict(STATE, params={"steps_per_tick": 99, "motor_gain": 50, "junk": 1}))
    cmd = await ws.receive_json()
    assert cmd["stats"]["params"]["steps_per_tick"] == 16
    assert cmd["stats"]["params"]["motor_gain"] == 5.0
    assert "junk" not in cmd["stats"]["params"]
    assert brain.ticks[0][1] == 16
    await ws.close()


async def test_reset(aiohttp_client, tmp_path):
    _, ws, _, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    await ws.send_json({"type": "reset"})
    assert (await ws.receive_json()) == {"type": "reset_ok"}
    assert brain.resets == 1
    await ws.close()


async def test_malformed_and_unknown_messages_keep_connection(aiohttp_client, tmp_path):
    _, ws, _, _ = await connect(aiohttp_client, tmp_path=tmp_path)
    await ws.send_str("not json")
    assert (await ws.receive_json())["type"] == "error"
    await ws.send_json({"type": "state", "ball": {}})
    assert (await ws.receive_json())["type"] == "error"
    await ws.send_json({"type": "dance"})
    assert (await ws.receive_json())["type"] == "error"
    await ws.send_json(STATE)
    assert (await ws.receive_json())["type"] == "command"
    await ws.close()


async def test_unstable_brain_is_reset_and_reported(aiohttp_client, tmp_path):
    _, ws, _, brain = await connect(aiohttp_client, FakeBrain(unstable=True), tmp_path)
    await ws.send_json(STATE)
    err = await ws.receive_json()
    assert err["type"] == "error" and "reset" in err["message"]
    assert brain.resets == 1
    await ws.close()


async def test_events_reach_the_brain_and_learning_stats_are_echoed(aiohttp_client, tmp_path):
    _, ws, hello, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    assert hello["learning"] == {"plastic": 3}
    await ws.send_json(dict(STATE, events=["return", "bogus"], params={"learning_rate": 0.05, "learning_enabled": 0}))
    cmd = await ws.receive_json()
    assert brain.ticks[-1][2] == ("return",) and brain.ticks[-1][3] is False and brain.ticks[-1][4] == 0.05
    assert cmd["stats"]["learning"] == {"fake": True}
    await ws.close()


async def test_model_params_readout_and_monitors(aiohttp_client, tmp_path):
    _, ws, hello, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    assert hello["model"] == {"name": "fake"}
    await ws.send_json(dict(STATE, params={"steps_per_tick": 4, "dt_ms": 0.4, "noise_mv": 0.7,
                                           "depression_u": 0.1, "readout_motor": 1, "punish_reflex": 1}))
    cmd = await ws.receive_json()
    assert brain.configured == (0.5, 0.7, 0.1)                 # dt snapped to 0.5
    assert brain.ticks[-1][1] == 8 and brain.ticks[-1][5] == 1.0   # 4 brain ms at 0.5 ms steps = 8 steps
    assert cmd["move"] > 0                                     # motor readout: right 5 vs left 0 -> down
    assert cmd["mn"] == {"left": 0, "right": 5} and cmd["stats"]["monitors"] == {"kc": 7}
    assert cmd["stats"]["steps"] == 8 and cmd["stats"]["brain_ms"] == 4.0 and cmd["stats"]["spikes_per_step"] == 0.5
    await ws.close()


async def test_forget(aiohttp_client, tmp_path):
    _, ws, _, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    await ws.send_json({"type": "forget"})
    assert (await ws.receive_json()) == {"type": "forget_ok"}
    assert brain.forgets == 1
    await ws.close()


async def test_only_the_newest_connection_drives_the_fly(aiohttp_client, tmp_path):
    client, ws1, _, brain = await connect(aiohttp_client, tmp_path=tmp_path)
    ws2 = await client.ws_connect("/ws")
    await ws2.receive_json()                                   # hello
    await ws1.send_json(STATE)
    err = await ws1.receive_json()
    assert err["type"] == "error" and err["fatal"] is True and "another tab" in err["message"]
    assert brain.ticks == []
    await ws2.send_json(STATE)
    assert (await ws2.receive_json())["type"] == "command"
    await ws2.close()
    await ws1.send_json(STATE)                                 # the survivor takes over again
    assert (await ws1.receive_json())["type"] == "command"
    await ws1.close()


async def test_index_is_served_with_cache_busting_and_no_cache(aiohttp_client, tmp_path):
    (tmp_path / "app.js").write_text("// script")
    (tmp_path / "style.css").write_text("/* css */")
    client, ws, _, _ = await connect(aiohttp_client, tmp_path=tmp_path)
    await ws.close()
    (tmp_path / "index.html").write_text('<link rel="stylesheet" href="/static/style.css"><h1>ok</h1><script src="/static/app.js"></script>')
    resp = await client.get("/")
    text = await resp.text()
    assert resp.status == 200 and "ok" in text
    assert '/static/app.js?v=' in text and '/static/style.css?v=' in text
    assert resp.headers["Cache-Control"] == "no-cache"
    static = await client.get("/static/app.js")
    assert static.status == 200 and static.headers["Cache-Control"] == "no-cache"
