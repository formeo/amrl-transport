"""
Createc STM adapter.

Wraps the Createc COM-based control into the STMTransport interface.
Requires Windows with Createc software installed (win32com).

This is a direct port of the original AMRL/Environment/createc_control.py,
preserving all the DAC/piezo math but hiding it behind the unified interface.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .protocol import (
    ConnectionState,
    ManipulationResult,
    ScanResult,
    STMTransport,
    TipPosition,
)

logger = logging.getLogger(__name__)

# Constants shared across Createc hardware
_DAC_UNIT = 2**19
_VOLT_UNIT = 10


class CreatecTransport(STMTransport):
    """
    Adapter for Createc LT-STM / STM-AFM systems.

    Communicates via Windows COM (win32com) with the Createc remote control
    interface ``pstmafm.stmafmrem``.

    Parameters
    ----------
    dispatch_mode : str
        'Dispatch' (shared) or 'DispatchEx' (isolated). Default 'Dispatch'.
    """

    def __init__(self, dispatch_mode: str = "Dispatch") -> None:
        self._dispatch_mode = dispatch_mode
        self._stm = None
        self._state = ConnectionState.DISCONNECTED

    # ── Connection ────────────────────────────────────────────

    def connect(self) -> None:
        try:
            import win32com.client
        except ImportError:
            raise RuntimeError(
                "win32com is required for Createc. "
                "Install pywin32: pip install pywin32"
            )

        logger.info("Connecting to Createc STM via COM...")
        if self._dispatch_mode == "DispatchEx":
            self._stm = win32com.client.DispatchEx("pstmafm.stmafmrem")
        else:
            self._stm = win32com.client.Dispatch("pstmafm.stmafmrem")

        if self._stm.stmready() == 1:
            self._state = ConnectionState.CONNECTED
            logger.info("Createc STM connected")
        else:
            # Fallback to DispatchEx
            self._stm = win32com.client.DispatchEx("pstmafm.stmafmrem")
            if self._stm.stmready() == 1:
                self._state = ConnectionState.CONNECTED
                logger.info("Createc STM connected (DispatchEx fallback)")
            else:
                self._state = ConnectionState.ERROR
                raise ConnectionError("Failed to connect to Createc STM")

    def disconnect(self) -> None:
        self._stm = None
        self._state = ConnectionState.DISCONNECTED
        logger.info("Createc STM disconnected")

    def is_connected(self) -> bool:
        if self._stm is None:
            return False
        try:
            return self._stm.stmready() == 1
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    # ── Internal helpers (DAC/piezo math) ─────────────────────

    def _getparam(self, key: str) -> str:
        return self._stm.getparam(key)

    def _getparam_f(self, key: str) -> float:
        return float(self._stm.getparam(key))

    def _setparam(self, key: str, value) -> None:
        self._stm.setparam(key, value)

    @property
    def _gain_x(self) -> float:
        return self._getparam_f("GainX")

    @property
    def _gain_y(self) -> float:
        return self._getparam_f("GainY")

    @property
    def _x_piezo(self) -> float:
        return self._getparam_f("Xpiezoconst")

    @property
    def _y_piezo(self) -> float:
        return self._getparam_f("Ypiezoconst")

    @property
    def _z_piezo(self) -> float:
        return self._getparam_f("Zpiezoconst")

    def _nm_to_pixel(
        self,
        x_start_nm: float,
        y_start_nm: float,
        x_end_nm: float | None,
        y_end_nm: float | None,
        offset_nm: np.ndarray,
        len_nm: float,
    ) -> tuple:
        delta_x = self._getparam_f("Delta X [Dac]")
        gain_x = self._gain_x
        pixel_to_a = delta_x * _VOLT_UNIT * gain_x * self._x_piezo / _DAC_UNIT

        x_start_px = int(np.rint(
            (x_start_nm - (offset_nm[0] - 0.5 * len_nm)) * 10 / pixel_to_a
        ))
        y_start_px = int(np.rint(
            (y_start_nm - offset_nm[1]) * 10 / pixel_to_a
        ))

        x_end_px = (
            int(np.rint((x_end_nm - (offset_nm[0] - 0.5 * len_nm)) * 10 / pixel_to_a))
            if x_end_nm is not None else None
        )
        y_end_px = (
            int(np.rint((y_end_nm - offset_nm[1]) * 10 / pixel_to_a))
            if y_end_nm is not None else None
        )
        return x_start_px, y_start_px, x_end_px, y_end_px

    def _get_delta_x(self, size_nm: float, num_x: float) -> int:
        gain_x = self._gain_x
        assert gain_x != 0, "GainX must not be 0"
        x_piezo = self._x_piezo
        delta_x = size_nm * 10 / (num_x * _VOLT_UNIT * gain_x * x_piezo / _DAC_UNIT)
        return int(delta_x)

    def _estimate_scan_time(self) -> float:
        scan_time = self._getparam_f("Sec/Image:")
        delay_y = self._getparam_f("Delay Y")
        return scan_time / 2 * (1 + 1 / delay_y)

    # ── STMTransport implementation ───────────────────────────

    def scan_image(
        self,
        size_nm: float,
        offset_nm: np.ndarray,
        pixel: int,
        bias_mv: float,
        speed: float | None = None,
    ) -> ScanResult:
        self.ramp_bias(bias_mv)

        self._setparam("Num.X", pixel)
        delta_x = self._get_delta_x(size_nm, pixel)
        self._setparam("Delta X [Dac]", delta_x)
        time.sleep(0.1)

        scan_time = self._estimate_scan_time()
        logger.info("Scan will take %.1f seconds", scan_time)

        gain_x = self._gain_x
        x_v = gain_x * offset_nm[0] / self._x_piezo
        y_v = gain_x * offset_nm[1] / self._y_piezo
        self._stm.setxyoffvolt(x_v, y_v)

        if speed is not None:
            self._set_speed_internal(speed)

        self._stm.scanstart()
        time.sleep(max(0, scan_time - 2))
        while self._stm.scanstatus == 2:
            time.sleep(2)

        img_forward = np.array(self._stm.scandata(1, 4))
        img_backward = np.array(self._stm.scandata(257, 4))

        x_length = 0.1 * self._getparam_f("Length x[A]")
        y_length = 0.1 * self._getparam_f("Length y[A]")

        return ScanResult(
            img_forward=img_forward,
            img_backward=img_backward,
            offset_nm=np.array(offset_nm),
            size_nm=np.array([x_length, y_length]),
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
        pixels = self._nm_to_pixel(
            x_start_nm, y_start_nm, x_end_nm, y_end_nm, offset_nm, size_nm
        )
        x_s, y_s, x_e, y_e = pixels

        if [x_s, y_s] == [x_e, y_e]:
            return None

        self.ramp_bias(bias_mv)
        preamp_gain = 10 ** self._getparam_f("Latmangain")
        self._setparam("LatmanVolt", bias_mv)
        self._setparam("Latmanlgi", current_pa * 1e-9 * preamp_gain)
        self._stm.latmanip(x_s, y_s, x_e, y_e)

        return ManipulationResult(
            time=np.array(self._stm.latmandata(0, 0)),
            x=np.array(self._stm.latmandata(1, 4)),
            y=np.array(self._stm.latmandata(2, 4)),
            current=np.array(self._stm.latmandata(3, 3)),
            di_dv=np.array(self._stm.latmandata(4, 0)),
            topography=np.array(self._stm.latmandata(15, 4)),
        )

    def tip_form(self, z_angstrom: float, x_nm: float, y_nm: float) -> None:
        offset_nm = self._get_offset_internal()
        len_nm = self.get_image_size_nm()
        self._setparam("TipForm_Z", 1000 * z_angstrom / self._z_piezo)
        x_px, y_px, _, _ = self._nm_to_pixel(
            x_nm, y_nm, None, None, offset_nm, len_nm
        )
        self._stm.btn_tipform(x_px, y_px)
        self._stm.waitms(50)

    def get_tip_position(self) -> TipPosition:
        offset_x = self._getparam_f("OffsetX")
        offset_y = self._getparam_f("OffsetY")
        x_nm = -0.1 * self._x_piezo * _VOLT_UNIT * offset_x * self._gain_x / _DAC_UNIT
        y_nm = -0.1 * self._y_piezo * _VOLT_UNIT * offset_y * self._gain_y / _DAC_UNIT
        return TipPosition(x_nm=x_nm, y_nm=y_nm)

    def set_tip_position(self, x_nm: float, y_nm: float) -> None:
        self._stm.setxyoffvolt(
            10 * x_nm / self._x_piezo,
            10 * y_nm / self._y_piezo,
        )

    def ramp_bias(self, target_mv: float, speed: int = 2) -> None:
        speed = max(1, int(speed))
        init_mv = self._getparam_f("Biasvolt.[mV]")

        if init_mv * target_mv == 0 or init_mv == target_mv:
            return

        if init_mv * target_mv > 0:
            self._ramp_same_pole(target_mv, init_mv, speed)
        else:
            if abs(init_mv) > abs(target_mv):
                self._setparam("Biasvolt.[mV]", -init_mv)
                self._ramp_same_pole(target_mv, -init_mv, speed)
            elif abs(init_mv) < abs(target_mv):
                self._ramp_same_pole(-target_mv, init_mv, speed)
                self._setparam("Biasvolt.[mV]", target_mv)
            else:
                self._setparam("Biasvolt.[mV]", target_mv)

    def _ramp_same_pole(self, end_mv: float, init_mv: float, spd: int) -> None:
        pole = np.sign(init_mv)
        start = int(spd * np.log10(abs(init_mv)))
        end = int(spd * np.log10(abs(end_mv)))
        sign = int(np.sign(end - start))
        for i in range(start + sign, end + sign, sign):
            time.sleep(0.01)
            self._setparam("Biasvolt.[mV]", pole * 10 ** (i / spd))
        self._setparam("Biasvolt.[mV]", end_mv)

    def get_bias(self) -> float:
        return self._getparam_f("Biasvolt.[mV]")

    def get_scan_speed(self) -> float:
        gain_x = self._gain_x
        delta_x = self._getparam_f("Delta X [Dac]")
        x_piezo = self._x_piezo
        dx_d = self._getparam_f("DX/DDeltaX")
        return delta_x * _VOLT_UNIT * gain_x * x_piezo / (_DAC_UNIT * dx_d * 20e-6)

    def get_image_size_nm(self) -> float:
        gain_x = self._gain_x
        x_piezo = self._x_piezo
        num_x = self._getparam_f("Num.X")
        delta_x = self._getparam_f("Delta X [Dac]")
        return delta_x * num_x * _VOLT_UNIT * gain_x * x_piezo / (10 * _DAC_UNIT)

    def _get_offset_internal(self) -> np.ndarray:
        pos = self.get_tip_position()
        return np.array([pos.x_nm, pos.y_nm])

    def _set_speed_internal(self, speed_a_per_s: float) -> None:
        gain_x = self._gain_x
        delta_x = self._getparam_f("Delta X [Dac]")
        x_piezo = self._x_piezo
        dx_d = (delta_x * _VOLT_UNIT * gain_x * x_piezo) / (
            speed_a_per_s * _DAC_UNIT * 20e-6
        )
        self._setparam("DX/DDeltaX", int(dx_d))

    @property
    def name(self) -> str:
        return "Createc LT-STM"
