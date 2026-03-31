"""
Abstract transport protocol for STM/AFM controllers.

This module defines the interface that any STM hardware adapter must implement.
The goal is to decouple the RL environment from specific hardware, enabling:
- Multiple STM brands (Createc, Nanonis, Scienta Omicron, RHK, etc.)
- A simulation backend for testing without hardware
- Remote STM access over the network

Design follows the same pattern as database drivers (DB-API 2.0) or
storage backends — a thin, hardware-agnostic interface that adapters implement.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class ScanResult:
    """Immutable result of an STM scan."""
    img_forward: np.ndarray
    img_backward: np.ndarray
    offset_nm: np.ndarray
    size_nm: np.ndarray  # (x_length, y_length)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ManipulationResult:
    """Immutable result of a lateral manipulation."""
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    current: np.ndarray
    di_dv: np.ndarray
    topography: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TipPosition:
    """Current tip position in nm."""
    x_nm: float
    y_nm: float


class STMTransport(abc.ABC):
    """
    Abstract base class for STM hardware controllers.

    Every adapter must implement these methods. The interface uses
    physical units (nm, mV, pA) — conversion to hardware-specific
    units (DAC, pixels, volts) is the adapter's responsibility.

    Lifecycle:
        controller = SomeAdapter(config)
        controller.connect()
        try:
            result = controller.scan_image(...)
            ...
        finally:
            controller.disconnect()

    Or as a context manager:
        with SomeAdapter(config) as stm:
            result = stm.scan_image(...)
    """

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish connection to the STM hardware."""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Gracefully close connection to the STM hardware."""
        ...

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if connection to hardware is active."""
        ...

    # ── Scanning ──────────────────────────────────────────────

    @abc.abstractmethod
    def scan_image(
        self,
        size_nm: float,
        offset_nm: np.ndarray,
        pixel: int,
        bias_mv: float,
        speed: float | None = None,
    ) -> ScanResult:
        """
        Perform an STM scan.

        Parameters
        ----------
        size_nm : float
            Image size in nm.
        offset_nm : ndarray, shape (2,)
            XY offset in nm.
        pixel : int
            Number of scan pixels.
        bias_mv : float
            Scan bias in mV.
        speed : float, optional
            Scan speed in Å/s.

        Returns
        -------
        ScanResult
        """
        ...

    # ── Manipulation ──────────────────────────────────────────

    @abc.abstractmethod
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
        """
        Execute lateral manipulation (push/pull atom).

        Parameters
        ----------
        x_start_nm, y_start_nm : float
            Start position in nm (absolute STM coordinates).
        x_end_nm, y_end_nm : float
            End position in nm (absolute STM coordinates).
        bias_mv : float
            Manipulation bias in mV.
        current_pa : float
            Manipulation current setpoint in pA.
        offset_nm : ndarray, shape (2,)
            Current XY offset in nm.
        size_nm : float
            Current image size in nm.

        Returns
        -------
        ManipulationResult or None
            None if start == end (no movement needed).
        """
        ...

    # ── Tip management ────────────────────────────────────────

    @abc.abstractmethod
    def tip_form(self, z_angstrom: float, x_nm: float, y_nm: float) -> None:
        """
        Perform tip forming / conditioning.

        Parameters
        ----------
        z_angstrom : float
            Z approach depth in Å.
        x_nm, y_nm : float
            Position for tip forming in nm.
        """
        ...

    # ── Position ──────────────────────────────────────────────

    @abc.abstractmethod
    def get_tip_position(self) -> TipPosition:
        """Get current tip position in nm."""
        ...

    @abc.abstractmethod
    def set_tip_position(self, x_nm: float, y_nm: float) -> None:
        """Move tip to given position in nm."""
        ...

    # ── Bias ──────────────────────────────────────────────────

    @abc.abstractmethod
    def ramp_bias(self, target_mv: float, speed: int = 2) -> None:
        """Gradually ramp bias to target value."""
        ...

    @abc.abstractmethod
    def get_bias(self) -> float:
        """Get current bias in mV."""
        ...

    # ── Calibration info ──────────────────────────────────────

    @abc.abstractmethod
    def get_scan_speed(self) -> float:
        """Get current scan speed in Å/s."""
        ...

    @abc.abstractmethod
    def get_image_size_nm(self) -> float:
        """Get current image size in nm."""
        ...

    # ── Context manager ───────────────────────────────────────

    def __enter__(self) -> STMTransport:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ── Capabilities ──────────────────────────────────────────

    @property
    def capabilities(self) -> dict:
        """
        Report what this transport supports.
        Override in subclasses to indicate special features.
        """
        return {
            "lateral_manipulation": True,
            "vertical_manipulation": False,
            "tip_forming": True,
            "spectroscopy": False,
            "bias_ramp": True,
        }

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'Createc LT-STM'."""
        ...
