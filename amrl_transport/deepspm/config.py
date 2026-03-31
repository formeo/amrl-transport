"""
DeepSPM server configuration.

Reads the original deepSPM_server.ini format for backward compatibility,
and also supports YAML configuration with Pydantic validation.
"""
from __future__ import annotations

import configparser
from pathlib import Path

from pydantic import BaseModel, Field


def _parse_pair(s: str) -> tuple[float, float]:
    """Parse '80n;160n' -> (80e-9, 160e-9)."""
    from .protocol import _parse_val
    parts = s.strip().strip('"').split(";")
    a, _ = _parse_val(parts[0])
    b, _ = _parse_val(parts[1])
    return a, b


class ScanSettings(BaseModel):
    speed_fwd: float = Field(80e-9, description="Forward scan speed (m/s)")
    speed_bwd: float = Field(160e-9, description="Backward scan speed (m/s)")
    setpoint_bias_v: float = Field(-1.0, description="Tunneling bias (V)")
    setpoint_current_a: float = Field(25e-12, description="Tunneling current (A)")
    z_ctrl_setpoint: float = Field(10e-12)
    z_ctrl_time: float = Field(120e-6)

class ApproachSettings(BaseModel):
    wait_ms: int = Field(1000)
    motor_freq: int = Field(1000)
    motor_voltage: int = Field(235)
    optimal_z_min: float = Field(10e-9)
    optimal_z_max: float = Field(-62e-9)

class ShapingSettings(BaseModel):
    draw_actions: bool = Field(True)
    wait_ms: int = Field(500)
    rough_threshold_m: float = Field(600e-12)

class SafetyLimits(BaseModel):
    """Safety limits from [Limits] section — values the server will clamp to."""
    time_min_s: float = Field(20e-3)
    time_max_s: float = Field(5.0)
    image_min_m: float = Field(1e-9)
    image_max_m: float = Field(200e-9)
    pixel_min: int = Field(8)
    pixel_max: int = Field(1024)
    bias_min_v: float = Field(19e-3)
    bias_max_v: float = Field(8.2)
    dip_min_m: float = Field(-30e-9)
    dip_max_m: float = Field(30e-9)
    position_fraction: float = Field(0.95)

class MoveAreaSettings(BaseModel):
    steps_z: int = Field(250)
    steps_lateral: int = Field(200)
    retract_freq: int = Field(1000)
    retract_voltage: int = Field(250)

class ServerConfig(BaseModel):
    """Full DeepSPM server configuration."""
    host: str = Field("0.0.0.0")
    port: int = Field(50008)
    test_ip: str = Field("0.0.0.0")
    scan: ScanSettings = Field(default_factory=ScanSettings)
    approach: ApproachSettings = Field(default_factory=ApproachSettings)
    shaping: ShapingSettings = Field(default_factory=ShapingSettings)
    limits: SafetyLimits = Field(default_factory=SafetyLimits)
    movearea: MoveAreaSettings = Field(default_factory=MoveAreaSettings)
    # Transport backend
    transport_type: str = Field("simulator", description="simulator | nanonis | amrl_transport")
    nanonis_host: str = Field("localhost")
    nanonis_port: int = Field(6501)
    log_file: str | None = Field(None)

    @classmethod
    def from_ini(cls, path: str | Path) -> ServerConfig:
        """
        Load from a deepSPM_server.ini file (original format).

        This provides backward compatibility — labs with existing INI
        configs can switch from LabVIEW to this Python server by just
        changing the executable, keeping the same config file.
        """
        cp = configparser.ConfigParser()
        cp.read(str(path))

        cfg = cls()

        if cp.has_section("Settings"):
            s = cp["Settings"]
            cfg.test_ip = s.get("test_IP", cfg.test_ip).strip('"')

            if "speed" in s:
                fwd, bwd = _parse_pair(s["speed"])
                cfg.scan.speed_fwd = fwd
                cfg.scan.speed_bwd = bwd

            if "setpoint" in s:
                bias_v, current_a = _parse_pair(s["setpoint"])
                cfg.scan.setpoint_bias_v = bias_v
                cfg.scan.setpoint_current_a = current_a

            if "z_ctrl" in s:
                sp, t = _parse_pair(s["z_ctrl"])
                cfg.scan.z_ctrl_setpoint = sp
                cfg.scan.z_ctrl_time = t

            if "approach_wait" in s:
                cfg.approach.wait_ms = int(s["approach_wait"].strip('"'))

            if "approach_motor" in s:
                freq, volt = _parse_pair(s["approach_motor"])
                cfg.approach.motor_freq = int(freq)
                cfg.approach.motor_voltage = int(volt)

            if "optimal_z_range" in s:
                zmin, zmax = _parse_pair(s["optimal_z_range"])
                cfg.approach.optimal_z_min = zmin
                cfg.approach.optimal_z_max = zmax

            if "shaping_wait" in s:
                cfg.shaping.wait_ms = int(s["shaping_wait"].strip('"'))

            if "rough_threshold" in s:
                from .protocol import _parse_val
                val, _ = _parse_val(s["rough_threshold"].strip('"'))
                cfg.shaping.rough_threshold_m = val

            if "movearea_steps" in s:
                z, lat = _parse_pair(s["movearea_steps"])
                cfg.movearea.steps_z = int(z)
                cfg.movearea.steps_lateral = int(lat)

        if cp.has_section("Limits"):
            lim = cp["Limits"]
            if "time" in lim:
                tmin, tmax = _parse_pair(lim["time"])
                cfg.limits.time_min_s = tmin
                cfg.limits.time_max_s = tmax
            if "image" in lim:
                imin, imax = _parse_pair(lim["image"])
                cfg.limits.image_min_m = imin
                cfg.limits.image_max_m = imax
            if "pixel" in lim:
                pmin, pmax = _parse_pair(lim["pixel"])
                cfg.limits.pixel_min = int(pmin)
                cfg.limits.pixel_max = int(pmax)
            if "bias" in lim:
                bmin, bmax = _parse_pair(lim["bias"])
                cfg.limits.bias_min_v = bmin
                cfg.limits.bias_max_v = bmax
            if "dip" in lim:
                dmin, dmax = _parse_pair(lim["dip"])
                cfg.limits.dip_min_m = dmin
                cfg.limits.dip_max_m = dmax
            if "position" in lim:
                cfg.limits.position_fraction = float(lim["position"])

        return cfg
