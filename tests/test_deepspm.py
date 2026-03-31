"""
Tests for DeepSPM protocol, server, and client.

Tests the full stack: protocol parsing → async server → client → simulator.
No hardware or LabVIEW needed.
"""
import asyncio
import struct
import threading
import time

import numpy as np
import pytest

from amrl_transport.deepspm.client import DeepSPMClient
from amrl_transport.deepspm.protocol import (
    CommandType,
    GetParamCommand,
    MoveAreaCommand,
    ScanCommand,
    TipShapingCommand,
    encode_param_response,
    encode_scan_response,
    parse_command,
)
from amrl_transport.deepspm.server import InstrumentServer, ServerConfig
from amrl_transport.transport import SimulatorTransport

# ── Protocol parsing tests ────────────────────────────────────


class TestProtocolParser:
    def test_parse_scan(self):
        cmd = parse_command(b"scan(10n,20n,50n,64)")
        assert isinstance(cmd, ScanCommand)
        assert abs(cmd.x_nm - 10.0) < 0.01
        assert abs(cmd.y_nm - 20.0) < 0.01
        assert abs(cmd.size_nm - 50.0) < 0.01
        assert cmd.pixels == 64

    def test_parse_scan_with_spaces(self):
        cmd = parse_command(b"scan(10n, 20n, 50n , 64)")
        assert isinstance(cmd, ScanCommand)
        assert cmd.pixels == 64

    def test_parse_tipshaping(self):
        cmd = parse_command(b"tipshaping(5n,10n,-1.0n,20m,200m)")
        assert isinstance(cmd, TipShapingCommand)
        assert abs(cmd.x_nm - 5.0) < 0.01
        assert abs(cmd.y_nm - 10.0) < 0.01
        assert abs(cmd.dip_m - (-1e-9)) < 1e-12
        assert abs(cmd.bias_v - 20e-3) < 1e-6
        assert abs(cmd.timing_s - 200e-3) < 1e-6

    def test_parse_tipshaping_stall(self):
        cmd = parse_command(b"tipshaping(5n,10n,stall)")
        assert isinstance(cmd, TipShapingCommand)
        assert cmd.action_str == "stall"

    def test_parse_tipshaping_bias_only(self):
        cmd = parse_command(b"tipshaping(0n,0n,0,4,200m)")
        assert isinstance(cmd, TipShapingCommand)
        assert abs(cmd.dip_m) < 1e-15
        assert abs(cmd.bias_v - 4.0) < 1e-6

    def test_parse_getparam(self):
        cmd = parse_command(b"getparam(Range)")
        assert isinstance(cmd, GetParamCommand)
        assert cmd.param_name == "Range"

    def test_parse_getparam_zrange(self):
        cmd = parse_command(b"getparam(zRange)")
        assert isinstance(cmd, GetParamCommand)
        assert cmd.param_name == "zRange"

    def test_parse_approach(self):
        cmd = parse_command(b"approach(f)")
        assert cmd.cmd == CommandType.APPROACH

    def test_parse_movearea(self):
        cmd = parse_command(b"movearea(y+)")
        assert isinstance(cmd, MoveAreaCommand)
        assert cmd.direction == "y+"

    def test_parse_tipclean(self):
        cmd = parse_command(b"tipclean(5n,10n)")
        assert cmd.cmd == CommandType.TIPCLEAN

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError):
            parse_command(b"bogus()")


class TestProtocolEncoder:
    def test_encode_scan_response(self):
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        data = encode_scan_response(img)
        # Header: 2 x int32 big-endian
        h, w = struct.unpack(">ii", data[:8])
        assert h == 32
        assert w == 32
        # Body: 32*32*4 bytes
        assert len(data) == 8 + 32 * 32 * 4

    def test_encode_param_response(self):
        data = encode_param_response("Range", 800e-9)
        text = data.decode("utf-8")
        assert "Range:" in text
        assert "8" in text


# ── Full integration: server + client ─────────────────────────


class TestServerClientIntegration:
    """
    Spin up an async server with SimulatorTransport,
    connect a client, run commands, verify results.
    """

    @pytest.fixture(autouse=True)
    def setup_server(self):
        """Start server in a background thread, yield client, cleanup."""
        self.stm = SimulatorTransport(
            seed=42,
            atom_positions=np.array([[0.0, 0.0], [5.0, 5.0]]),
            noise_level=0.01,
        )
        self.stm.connect()

        self.config = ServerConfig(
            host="127.0.0.1",
            port=0,  # let OS pick a free port
            approach_area_m=800e-9,
            z_range_m=120e-9,
        )

        self._server_obj = InstrumentServer(self.stm, self.config)
        self._loop = asyncio.new_event_loop()
        self._actual_port = None

        def run_server():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_and_serve())

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()

        # Wait for server to start and get the actual port
        for _ in range(50):
            time.sleep(0.1)
            if self._actual_port is not None:
                break

        assert self._actual_port is not None, "Server did not start"

        self.client = DeepSPMClient(
            host="127.0.0.1",
            port=self._actual_port,
            timeout=10.0,
        )
        self.client.connect()

        yield

        self.client.disconnect()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self.stm.disconnect()

    async def _start_and_serve(self):
        server = await asyncio.start_server(
            self._server_obj._handle_client,
            "127.0.0.1",
            0,  # OS picks port
        )
        self._actual_port = server.sockets[0].getsockname()[1]
        async with server:
            await server.serve_forever()

    def test_scan(self):
        img = self.client.scan(x_nm=0, y_nm=0, size_nm=10, pixels=32)
        assert img.shape == (32, 32)
        assert img.dtype == np.float32

    def test_scan_has_atom_signal(self):
        """Atom at (0,0) should appear in center."""
        img = self.client.scan(x_nm=0, y_nm=0, size_nm=2, pixels=64)
        center = img[24:40, 24:40].mean()
        corner = img[0:8, 0:8].mean()
        assert center > corner

    def test_get_approach_area(self):
        area_nm = self.client.get_approach_area()
        assert area_nm == pytest.approx(800.0, abs=1.0)

    def test_get_z_range(self):
        z = self.client.get_z_range()
        assert z != 0.0

    def test_tipshaping(self):
        result = self.client.tipshaping(0, 0, "0,-4,200m")
        assert result  # non-empty response

    def test_tipshaping_stall(self):
        result = self.client.tipshaping(0, 0, "stall")
        assert result == "stall"

    def test_tipclean(self):
        result = self.client.tipclean(0, 0)
        assert result is True

    def test_approach(self):
        result = self.client.approach()
        assert "Approached" in result or "Z-range" in result

    def test_movearea(self):
        crashes = self.client.movearea("y+")
        assert crashes == 0

    def test_multiple_scans(self):
        """Server should handle multiple sequential commands."""
        for _ in range(5):
            img = self.client.scan(0, 0, 5, 32)
            assert img.shape == (32, 32)
