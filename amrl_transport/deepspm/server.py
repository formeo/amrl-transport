"""
DeepSPM Instrument Server — pure Python/asyncio replacement for LabVIEW.

Speaks the same TCP protocol as the original LabVIEW TCPIP.Server v2.vi.
The DeepSPM agent code works WITHOUT MODIFICATION — just point it here.
"""
from __future__ import annotations

import asyncio
import configparser
import logging
import signal
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..transport.protocol import STMTransport
from .protocol import (
    ApproachCommand, Command, GetParamCommand, MoveAreaCommand,
    ScanCommand, TipCleanCommand, TipShapingCommand,
    encode_approach_response, encode_movearea_response,
    encode_param_response, encode_scan_response, encode_text_response,
    parse_command,
)

logger = logging.getLogger(__name__)


def _ini_val(s: str) -> float:
    """Parse INI value with SI suffix."""
    s = s.strip().strip('"')
    for suffix, mult in {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3}.items():
        if s.endswith(suffix):
            return float(s[:-1]) * mult
    return float(s)


@dataclass
class ServerConfig:
    """Server configuration matching deepSPM_server.ini."""
    host: str = "0.0.0.0"
    port: int = 50008
    scan_bias_v: float = -1.0
    approach_area_m: float = 800e-9
    z_range_m: float = 120e-9
    rough_threshold_m: float = 600e-12
    shaping_wait_s: float = 0.5
    lim_dip_min_m: float = -30e-9
    lim_dip_max_m: float = 30e-9
    lim_bias_min_v: float = 19e-3
    lim_bias_max_v: float = 8.2
    lim_time_min_s: float = 20e-3
    lim_time_max_s: float = 5.0
    lim_image_min_m: float = 1e-9
    lim_image_max_m: float = 200e-9
    lim_pixel_min: int = 8
    lim_pixel_max: int = 1024

    @classmethod
    def from_ini(cls, path: str) -> "ServerConfig":
        cfg = configparser.ConfigParser()
        cfg.read(path)
        c = cls()
        if cfg.has_section("Settings"):
            s = cfg["Settings"]
            sp = s.get("setpoint", "-1;25p").strip('"').split(";")
            c.scan_bias_v = float(sp[0])
            c.rough_threshold_m = _ini_val(s.get("rough_threshold", "600p").strip('"'))
            c.shaping_wait_s = float(s.get("shaping_wait", "500").strip('"')) / 1000
        if cfg.has_section("Limits"):
            lim = cfg["Limits"]
            dp = lim.get("dip", "-30n;30n").strip('"').split(";")
            c.lim_dip_min_m = _ini_val(dp[0])
            c.lim_dip_max_m = _ini_val(dp[1])
            bp = lim.get("bias", "19m;8.2").strip('"').split(";")
            c.lim_bias_min_v = _ini_val(bp[0])
            c.lim_bias_max_v = _ini_val(bp[1])
        return c


class InstrumentServer:
    """
    Async TCP server replacing the LabVIEW DeepSPM instrument control server.

    Usage:
        stm = SimulatorTransport(seed=42); stm.connect()
        server = InstrumentServer(stm, ServerConfig(port=50008))
        asyncio.run(server.serve_forever())
    """

    def __init__(self, transport: STMTransport, config: Optional[ServerConfig] = None):
        self.transport = transport
        self.config = config or ServerConfig()
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self.config.host, self.config.port,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("DeepSPM server on %s:%d (%s)", addr[0], addr[1], self.transport.name)

    async def serve_forever(self) -> None:
        if not self._server:
            await self.start()
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        logger.info("Agent connected from %s", addr)
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                try:
                    cmd = parse_command(data)
                    logger.info("< %s", type(cmd).__name__)
                    t0 = time.monotonic()
                    resp = await self._dispatch(cmd)
                    logger.info("> %d bytes (%.2fs)", len(resp), time.monotonic() - t0)
                    writer.write(resp)
                    await writer.drain()
                except Exception as e:
                    logger.error("Error: %s", e, exc_info=True)
                    writer.write(encode_text_response(f"ERROR:{e}"))
                    await writer.drain()
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            writer.close()
            logger.info("Agent %s disconnected", addr)

    async def _dispatch(self, cmd: Command) -> bytes:
        loop = asyncio.get_event_loop()
        c = self.config

        if isinstance(cmd, ScanCommand):
            size_m = np.clip(cmd.size_nm * 1e-9, c.lim_image_min_m, c.lim_image_max_m)
            px = int(np.clip(cmd.pixels, c.lim_pixel_min, c.lim_pixel_max))
            result = await loop.run_in_executor(None, lambda: self.transport.scan_image(
                size_nm=size_m * 1e9,
                offset_nm=np.array([cmd.x_nm, cmd.y_nm]),
                pixel=px,
                bias_mv=c.scan_bias_v * 1000,
            ))
            return encode_scan_response(result.img_forward)

        elif isinstance(cmd, TipShapingCommand):
            if cmd.action_str == "stall":
                return encode_text_response("0.5")
            dip = np.clip(cmd.dip_m, c.lim_dip_min_m, c.lim_dip_max_m)
            bias = np.clip(cmd.bias_v, c.lim_bias_min_v, c.lim_bias_max_v)
            timing = np.clip(cmd.timing_s, c.lim_time_min_s, c.lim_time_max_s)
            if abs(dip) > 1e-12:
                await loop.run_in_executor(None, lambda: self.transport.tip_form(
                    dip * 1e10, cmd.x_nm, cmd.y_nm,
                ))
            elif abs(bias) > 1e-6:
                await loop.run_in_executor(None, lambda: self.transport.ramp_bias(bias * 1000))
                await asyncio.sleep(timing)
                await loop.run_in_executor(None, lambda: self.transport.ramp_bias(c.scan_bias_v * 1000))
            await asyncio.sleep(c.shaping_wait_s)
            return encode_text_response("1")

        elif isinstance(cmd, TipCleanCommand):
            await loop.run_in_executor(None, lambda: self.transport.tip_form(-50.0, cmd.x_nm, cmd.y_nm))
            return encode_text_response("1")

        elif isinstance(cmd, GetParamCommand):
            if cmd.param_name == "Range":
                return encode_param_response("Range", c.approach_area_m)
            elif cmd.param_name == "zRange":
                return encode_param_response("zRange", c.z_range_m)
            return encode_param_response(cmd.param_name, 0.0)

        elif isinstance(cmd, ApproachCommand):
            return encode_approach_response(c.z_range_m)

        elif isinstance(cmd, MoveAreaCommand):
            return encode_movearea_response(crashes=0)

        return encode_text_response("ERROR:unknown")
