"""
Nanonis STM adapter (stub).

Nanonis (SPECS / Scienta Omicron) is the most widely used STM control
software in academia. It exposes a TCP/IP programming interface that
can be driven from any platform (no Windows COM dependency).

This is a stub implementation — the TCP protocol commands need to be
filled in based on the Nanonis Programming Interface documentation.
Contributions welcome!

Reference:
    Nanonis Programming Interface v5 — TCP command set
    DeepSPM project (github.com/abred/DeepSPM) uses similar TCP approach
"""
from __future__ import annotations

import logging
import socket
import struct

import numpy as np

from .protocol import (
    ManipulationResult,
    ScanResult,
    STMTransport,
    TipPosition,
)

logger = logging.getLogger(__name__)


class NanonisTransport(STMTransport):
    """
    Adapter for Nanonis-controlled STM systems.

    Communicates via TCP/IP with the Nanonis programming interface.
    Works on Linux, macOS, and Windows — no COM dependency.

    Parameters
    ----------
    host : str
        Nanonis TCP server host (default 'localhost').
    port : int
        Nanonis TCP server port (default 6501).
    """

    def __init__(self, host: str = "localhost", port: int = 6501) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        logger.info("Connecting to Nanonis at %s:%d", self._host, self._port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(10.0)
        self._sock.connect((self._host, self._port))
        logger.info("Nanonis connected")

    def disconnect(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None
        logger.info("Nanonis disconnected")

    def is_connected(self) -> bool:
        return self._sock is not None

    def _send_command(self, command: str, body: bytes = b"") -> bytes:
        """
        Send a Nanonis TCP command and receive response.

        Nanonis protocol:
            Request:  [command_name (32 bytes)] [body_size (4 bytes, big-endian)] [body]
            Response: [command_name (32 bytes)] [body_size (4 bytes)] [body]

        Override this if your Nanonis version uses a different protocol.
        """
        if not self._sock:
            raise ConnectionError("Not connected to Nanonis")

        # Pack command header
        cmd_bytes = command.encode("ascii").ljust(32, b"\x00")
        header = cmd_bytes + struct.pack(">I", len(body))
        self._sock.sendall(header + body)

        # Read response header
        resp_header = self._recv_exact(36)
        resp_header[:32].rstrip(b"\x00").decode("ascii")
        resp_size = struct.unpack(">I", resp_header[32:36])[0]

        # Read response body
        resp_body = self._recv_exact(resp_size) if resp_size > 0 else b""
        return resp_body

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by Nanonis")
            data += chunk
        return data

    # ── STMTransport implementation (stubs) ───────────────────

    def scan_image(
        self,
        size_nm: float,
        offset_nm: np.ndarray,
        pixel: int,
        bias_mv: float,
        speed: float | None = None,
    ) -> ScanResult:
        # TODO: Implement using Nanonis Scan module commands:
        #   Scan.FrameSet (set scan frame)
        #   Scan.BufferSet (set channels)
        #   Scan.Action (start/stop)
        #   Scan.FrameDataGrab (get image data)
        raise NotImplementedError(
            "Nanonis scan_image not yet implemented. "
            "See Nanonis Programming Interface docs, Scan module."
        )

    def lateral_manipulation(
        self,
        x_start_nm: float,
        y_start_nm: float,
        x_end_nm: float,
        y_end_nm: float,
        bias_mv: float,
        current_pa: float,
        offset_nm: np.ndarray,
        size_nm: float,
    ) -> ManipulationResult | None:
        # TODO: Implement using Nanonis AtomManip module or
        #   manual tip movement with current feedback
        raise NotImplementedError(
            "Nanonis lateral_manipulation not yet implemented. "
            "Requires AtomManip or manual tip path control."
        )

    def tip_form(self, z_angstrom: float, x_nm: float, y_nm: float) -> None:
        # TODO: Implement via ZCtrl module + tip conditioning sequence
        raise NotImplementedError("Nanonis tip_form not yet implemented.")

    def get_tip_position(self) -> TipPosition:
        # TODO: Implement via Motor.StartClosedLoop or Piezo.RangeGet
        raise NotImplementedError("Nanonis get_tip_position not yet implemented.")

    def set_tip_position(self, x_nm: float, y_nm: float) -> None:
        raise NotImplementedError("Nanonis set_tip_position not yet implemented.")

    def ramp_bias(self, target_mv: float, speed: int = 2) -> None:
        # TODO: Bias.Set command
        raise NotImplementedError("Nanonis ramp_bias not yet implemented.")

    def get_bias(self) -> float:
        raise NotImplementedError("Nanonis get_bias not yet implemented.")

    def get_scan_speed(self) -> float:
        raise NotImplementedError("Nanonis get_scan_speed not yet implemented.")

    def get_image_size_nm(self) -> float:
        raise NotImplementedError("Nanonis get_image_size_nm not yet implemented.")

    @property
    def name(self) -> str:
        return f"Nanonis ({self._host}:{self._port})"
