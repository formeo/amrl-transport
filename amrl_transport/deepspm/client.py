"""
Modern DeepSPM client + drop-in EnvClient compatibility adapter.

Usage (standalone):
    client = DeepSPMClient("localhost", 50008)
    client.connect()
    img = client.scan(0, 0, 50, 64)

Usage (drop-in for DeepSPM agent):
    from amrl_transport.deepspm.client import EnvClientCompat as EnvClient
"""
from __future__ import annotations

import logging
import select
import socket
import struct
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _recvall(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def _drain(sock: socket.socket) -> None:
    sock.setblocking(False)
    try:
        while select.select([sock], [], [], 0.0)[0]:
            sock.recv(4096)
    except (BlockingIOError, OSError):
        pass
    finally:
        sock.setblocking(True)


class DeepSPMClient:
    """Clean TCP client for the DeepSPM instrument control server."""

    def __init__(self, host: str = "localhost", port: int = 50008, timeout: float = 180.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        logger.info("Connected to %s:%d", self.host, self.port)

    def disconnect(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *a):
        self.disconnect()

    def scan(self, x_nm: float, y_nm: float, size_nm: float, pixels: int) -> np.ndarray:
        """Request an STM scan. Returns float32 image (H, W)."""
        req = f"scan({x_nm}n,{y_nm}n,{size_nm}n,{pixels})"
        _drain(self._sock)
        self._sock.sendall(req.encode())
        hdr = _recvall(self._sock, 8)
        h, w = struct.unpack(">ii", hdr)
        body = _recvall(self._sock, h * w * 4)
        img = np.frombuffer(body, dtype=np.float32).copy().byteswap().reshape(h, w)
        return img

    def tipshaping(self, x_nm: float, y_nm: float, action: str) -> str:
        if action == "stall":
            return "stall"
        req = f"tipshaping({x_nm}n,{y_nm}n,{action})"
        _drain(self._sock)
        self._sock.sendall(req.encode())
        return self._sock.recv(1024).decode()

    def tipclean(self, x_nm: float, y_nm: float) -> bool:
        req = f"tipclean({x_nm}n,{y_nm}n)"
        _drain(self._sock)
        self._sock.sendall(req.encode())
        return self._sock.recv(64).decode().strip() == "1"

    def get_approach_area(self) -> float:
        """Returns approach area size in nm."""
        _drain(self._sock)
        self._sock.sendall(b"getparam(Range)")
        resp = self._sock.recv(64).decode()
        return float(resp.split(":")[1]) * 1e9

    def get_z_range(self) -> float:
        """Returns half Z-range in meters."""
        _drain(self._sock)
        self._sock.sendall(b"getparam(zRange)")
        resp = self._sock.recv(64).decode()
        return -float(resp.split(":")[1]) / 2.0

    def approach(self) -> str:
        _drain(self._sock)
        self._sock.sendall(b"approach(f)")
        old = self._sock.gettimeout()
        self._sock.settimeout(600)
        resp = self._sock.recv(256).decode()
        self._sock.settimeout(old)
        return resp

    def movearea(self, direction: str = "y+") -> int:
        """Move to new area. Returns crash count."""
        _drain(self._sock)
        self._sock.sendall(f"movearea({direction})".encode())
        old = self._sock.gettimeout()
        self._sock.settimeout(1800)
        resp = self._sock.recv(256).decode()
        self._sock.settimeout(old)
        try:
            return int(resp.split("with ")[1].split(" crash")[0])
        except (IndexError, ValueError):
            return 0


class EnvClientCompat:
    """
    Drop-in replacement for DeepSPM's envClient.EnvClient.

    Change import in agent.py:
        from amrl_transport.deepspm.client import EnvClientCompat as EnvClient
    """

    def __init__(self, sess, params: dict, agent):
        self.agent = agent
        self.params = params
        self.terminateOnFail = False
        self.requestSendTime = time.time()
        host = params.get("host") or "localhost"
        port = params.get("port") or 50008
        self._c = DeepSPMClient(host, port)
        self._c.connect()

        import os
        fn = os.path.join(params.get("out_dir", "."), "clientLog.txt")
        self.logFile = open(fn, "a")

    def act(self, action: str, px: float, py: float):
        if action == "stall":
            return
        try:
            self._c.tipshaping(px, py, action)
            self.terminateOnFail = False
        except Exception as e:
            if self.terminateOnFail:
                self.agent.terminate()
            self.terminateOnFail = True

    def getApproachArea(self) -> float:
        return self._c.get_approach_area()

    def getZRange(self) -> float:
        return self._c.get_z_range()

    def getState(self, currX: float, currY: float, size: float, pxRes: int) -> np.ndarray:
        return self._c.scan(currX, currY, size, pxRes)

    def newApproach(self):
        self._c.approach()

    def switchApproachArea(self):
        self._c.movearea("y+")

    def cleanTip(self, px: float, py: float) -> bool:
        return self._c.tipclean(px, py)

    def sendRequest(self, request: str):
        self._c._sock.sendall(request.encode())

    def logRequest(self, request: str):
        import datetime
        self.logFile.write(f"{datetime.datetime.now()}\t{time.time()}\t >> {request}\n")
        self.requestSendTime = time.time()
        self.logFile.flush()

    def logResponse(self, response: str):
        import datetime
        self.terminateOnFail = False
        self.logFile.write(f"{datetime.datetime.now()}\t{time.time()}\t << {response}\t{time.time()-self.requestSendTime}\n")
        self.logFile.flush()
