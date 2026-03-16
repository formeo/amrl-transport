"""
Simulated STM transport for development and testing.

Provides a physically plausible (but simplified) simulation of an STM:
- Generates synthetic scan images with Gaussian atom blobs
- Simulates lateral manipulation with configurable success probability
- Tracks atom positions on a virtual surface
- Adds realistic noise

This allows:
- Running the full RL training loop without hardware
- Unit testing the entire pipeline
- Benchmarking new algorithms before lab time
- CI/CD integration
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .protocol import (
    ManipulationResult,
    ScanResult,
    STMTransport,
    TipPosition,
)

logger = logging.getLogger(__name__)


class SimulatorTransport(STMTransport):
    """
    Simulated STM environment.

    Models a flat surface with adatoms that can be imaged and manipulated.
    Atoms are represented as 2D Gaussians in scan images.

    Parameters
    ----------
    lattice_constant : float
        Lattice constant in nm (default 0.288 for Ag(111)).
    atom_positions : ndarray, optional
        Initial atom positions in nm, shape (N, 2). If None, places
        a single atom at the origin.
    noise_level : float
        Gaussian noise std added to scan images (default 0.05).
    manipulation_success_rate : float
        Probability that a lateral manipulation actually moves the atom
        (default 0.7 — realistic for room temperature).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        lattice_constant: float = 0.288,
        atom_positions: Optional[np.ndarray] = None,
        noise_level: float = 0.05,
        manipulation_success_rate: float = 0.7,
        seed: Optional[int] = None,
    ) -> None:
        self._lattice = lattice_constant
        self._noise = noise_level
        self._success_rate = manipulation_success_rate
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._bias_mv = 100.0
        self._tip_pos = np.array([0.0, 0.0])

        if atom_positions is not None:
            self._atoms = np.array(atom_positions, dtype=float)
        else:
            self._atoms = np.array([[0.0, 0.0]])

    def connect(self) -> None:
        self._connected = True
        logger.info(
            "Simulator connected: %d atoms on surface", len(self._atoms)
        )

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Simulator disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def _render_image(
        self, size_nm: float, offset_nm: np.ndarray, pixel: int
    ) -> np.ndarray:
        """Render a synthetic STM image with atom blobs."""
        img = np.zeros((pixel, pixel), dtype=float)

        # Coordinate grid
        x_min = offset_nm[0] - size_nm / 2
        x_max = offset_nm[0] + size_nm / 2
        y_min = offset_nm[1] - size_nm / 2  
        y_max = offset_nm[1] + size_nm / 2

        xs = np.linspace(x_min, x_max, pixel)
        ys = np.linspace(y_min, y_max, pixel)
        xx, yy = np.meshgrid(xs, ys)

        # Atom FWHM ~ 0.15 nm for typical adatom
        sigma = 0.065  # nm

        for atom in self._atoms:
            dx = xx - atom[0]
            dy = yy - atom[1]
            r2 = dx**2 + dy**2
            img += np.exp(-r2 / (2 * sigma**2))

        # Add background corrugation (lattice)
        k = 2 * np.pi / self._lattice
        img += 0.03 * (np.cos(k * xx) + np.cos(k * yy))

        # Add noise
        img += self._rng.normal(0, self._noise, img.shape)

        return img

    # ── STMTransport implementation ───────────────────────────

    def scan_image(
        self,
        size_nm: float,
        offset_nm: np.ndarray,
        pixel: int,
        bias_mv: float,
        speed: Optional[float] = None,
    ) -> ScanResult:
        self._bias_mv = bias_mv
        img_fwd = self._render_image(size_nm, offset_nm, pixel)
        # Backward scan is slightly different (hysteresis)
        img_bwd = img_fwd + self._rng.normal(0, self._noise * 0.3, img_fwd.shape)

        return ScanResult(
            img_forward=img_fwd,
            img_backward=img_bwd,
            offset_nm=np.array(offset_nm),
            size_nm=np.array([size_nm, size_nm]),
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
    ) -> Optional[ManipulationResult]:
        if (x_start_nm == x_end_nm) and (y_start_nm == y_end_nm):
            return None

        # Find closest atom to manipulation start
        start = np.array([x_start_nm, y_start_nm])
        dists = np.linalg.norm(self._atoms - start, axis=1)
        closest_idx = np.argmin(dists)

        # Decide if manipulation succeeds
        # Higher current = higher success rate (simplified model)
        base_rate = self._success_rate
        current_factor = min(1.0, current_pa / 50000)  # saturates at 50 nA
        effective_rate = base_rate * current_factor
        success = self._rng.random() < effective_rate

        # Generate synthetic current trace
        n_points = 256
        t = np.linspace(0, 1, n_points)
        base_current = current_pa * 1e-12 * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))

        if success and dists[closest_idx] < 0.5:  # atom within 0.5 nm
            # Simulate atom jump — current spike
            jump_pos = self._rng.integers(n_points // 3, 2 * n_points // 3)
            base_current[jump_pos:jump_pos + 5] *= 3.0

            # Move atom toward end position (with some randomness)
            direction = np.array([x_end_nm, y_end_nm]) - self._atoms[closest_idx]
            move_frac = 0.5 + 0.5 * self._rng.random()  # move 50-100% of distance
            noise = self._rng.normal(0, 0.02, 2)  # positional noise
            self._atoms[closest_idx] += direction * move_frac + noise
            logger.debug(
                "Atom %d moved to (%.3f, %.3f)",
                closest_idx,
                self._atoms[closest_idx][0],
                self._atoms[closest_idx][1],
            )

        # Synthetic spatial trace
        x_trace = np.linspace(x_start_nm, x_end_nm, n_points)
        y_trace = np.linspace(y_start_nm, y_end_nm, n_points)

        return ManipulationResult(
            time=t,
            x=x_trace,
            y=y_trace,
            current=base_current,
            di_dv=np.gradient(base_current),
            topography=self._rng.normal(0, 0.01, n_points),
        )

    def tip_form(self, z_angstrom: float, x_nm: float, y_nm: float) -> None:
        logger.info("Simulator: tip form at (%.2f, %.2f), z=%.1f Å", x_nm, y_nm, z_angstrom)

    def get_tip_position(self) -> TipPosition:
        return TipPosition(x_nm=self._tip_pos[0], y_nm=self._tip_pos[1])

    def set_tip_position(self, x_nm: float, y_nm: float) -> None:
        self._tip_pos = np.array([x_nm, y_nm])

    def ramp_bias(self, target_mv: float, speed: int = 2) -> None:
        self._bias_mv = target_mv

    def get_bias(self) -> float:
        return self._bias_mv

    def get_scan_speed(self) -> float:
        return 500.0  # Å/s

    def get_image_size_nm(self) -> float:
        return 10.0  # default

    @property
    def name(self) -> str:
        return "Simulator"

    # ── Simulator-specific helpers ────────────────────────────

    @property
    def atom_positions(self) -> np.ndarray:
        """Direct access to atom positions (for testing)."""
        return self._atoms.copy()

    def add_atom(self, x_nm: float, y_nm: float) -> None:
        """Add an atom to the surface."""
        self._atoms = np.vstack([self._atoms, [x_nm, y_nm]])

    def reset_atoms(self, positions: np.ndarray) -> None:
        """Reset all atom positions."""
        self._atoms = np.array(positions, dtype=float)
