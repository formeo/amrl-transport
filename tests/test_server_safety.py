"""
Safety behaviour of the DeepSPM instrument server.

Commands are dispatched directly (no TCP) against a recording transport,
so the tests check exactly which hardware calls the server makes.
"""
import asyncio
import threading
import time

import numpy as np
import pytest

from amrl_transport.deepspm.protocol import (
    ApproachCommand,
    MoveAreaCommand,
    ScanCommand,
    TipCleanCommand,
    TipShapingCommand,
)
from amrl_transport.deepspm.server import InstrumentServer, ServerConfig
from amrl_transport.transport import SimulatorTransport
from amrl_transport.transport.protocol import ScanResult, STMTransport, TipPosition


class RecordingTransport(STMTransport):
    """Non-simulator transport that records calls and their threads."""

    def __init__(self, delay_s: float = 0.0):
        self.calls = []
        self.threads = set()
        self._delay = delay_s
        self._active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def _record(self, name, *args):
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        self.threads.add(threading.get_ident())
        self.calls.append((name, args))
        time.sleep(self._delay)
        with self._guard:
            self._active -= 1

    def connect(self): ...
    def disconnect(self): ...
    def is_connected(self): return True

    def scan_image(self, size_nm, offset_nm, pixel, bias_mv, speed=None):
        self._record("scan_image", size_nm, tuple(offset_nm), pixel, bias_mv)
        img = np.zeros((pixel, pixel))
        return ScanResult(img, img, np.array(offset_nm), np.array([size_nm, size_nm]))

    def lateral_manipulation(self, *a, **kw):
        self._record("lateral_manipulation")

    def tip_form(self, z_angstrom, x_nm, y_nm):
        self._record("tip_form", z_angstrom, x_nm, y_nm)

    def get_tip_position(self): return TipPosition(0.0, 0.0)
    def set_tip_position(self, x_nm, y_nm): ...

    def ramp_bias(self, target_mv, speed=2):
        self._record("ramp_bias", target_mv)

    def get_bias(self): return 0.0
    def get_scan_speed(self): return 0.0
    def get_image_size_nm(self): return 0.0

    @property
    def name(self): return "Recording"


def _dispatch(server, *cmds):
    async def run():
        return await asyncio.gather(*(server._dispatch(c) for c in cmds))
    return asyncio.run(run())


@pytest.fixture
def cfg():
    return ServerConfig(shaping_wait_s=0.0)


class TestBiasLimits:
    def test_negative_pulse_keeps_sign(self, cfg):
        stm = RecordingTransport()
        _dispatch(InstrumentServer(stm, cfg),
                  TipShapingCommand(action_str="0,-3,20m", bias_v=-3.0, timing_s=0.02))
        pulses = [a[0] for n, a in stm.calls if n == "ramp_bias"]
        assert pulses[0] == pytest.approx(-3000.0)

    def test_pulse_magnitude_clipped_both_signs(self, cfg):
        stm = RecordingTransport()
        _dispatch(InstrumentServer(stm, cfg),
                  TipShapingCommand(action_str="x", bias_v=-20.0, timing_s=0.02),
                  TipShapingCommand(action_str="x", bias_v=0.001, timing_s=0.02))
        pulses = [a[0] for n, a in stm.calls if n == "ramp_bias"][::2]
        assert pulses == [pytest.approx(-8200.0), pytest.approx(19.0)]

    def test_zero_action_touches_no_hardware(self, cfg):
        stm = RecordingTransport()
        resp, = _dispatch(InstrumentServer(stm, cfg),
                          TipShapingCommand(action_str="0,0,20m", timing_s=0.02))
        assert resp == b"1"
        assert stm.calls == []


class TestHardwareAccess:
    def test_concurrent_commands_are_serialized(self, cfg):
        stm = RecordingTransport(delay_s=0.05)
        server = InstrumentServer(stm, cfg)
        scans = [ScanCommand(size_nm=10, pixels=8) for _ in range(4)]
        _dispatch(server, *scans)
        assert len(stm.calls) == 4
        assert stm.max_active == 1

    def test_all_calls_on_one_thread(self, cfg):
        stm = RecordingTransport()
        _dispatch(InstrumentServer(stm, cfg), *[
            ScanCommand(size_nm=10, pixels=8),
            TipShapingCommand(action_str="x", bias_v=1.0, timing_s=0.02),
        ] * 3)
        assert len(stm.threads) == 1
        assert threading.get_ident() not in stm.threads


class TestUnimplementedOnHardware:
    @pytest.mark.parametrize("cmd", [ApproachCommand(), MoveAreaCommand(), TipCleanCommand()])
    def test_refused_on_real_transport(self, cfg, cmd):
        stm = RecordingTransport()
        with pytest.raises(NotImplementedError):
            _dispatch(InstrumentServer(stm, cfg), cmd)
        assert stm.calls == []

    @pytest.mark.parametrize("cmd", [ApproachCommand(), MoveAreaCommand(), TipCleanCommand()])
    def test_still_simulated_on_simulator(self, cfg, cmd):
        stm = SimulatorTransport(seed=0)
        stm.connect()
        resp, = _dispatch(InstrumentServer(stm, cfg), cmd)
        assert resp
