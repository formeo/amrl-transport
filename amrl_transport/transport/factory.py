"""
Factory for creating STM transport instances from configuration.
"""
from __future__ import annotations

import numpy as np

from ..config.models import TransportConfig, TransportType
from .protocol import STMTransport


def create_transport(config: TransportConfig) -> STMTransport:
    """
    Create an STM transport instance from configuration.

    Parameters
    ----------
    config : TransportConfig
        Transport configuration with type and backend-specific settings.

    Returns
    -------
    STMTransport
        A configured transport instance (not yet connected).

    Example
    -------
    >>> from amrl_transport.config.models import TransportConfig, TransportType
    >>> cfg = TransportConfig(type=TransportType.SIMULATOR)
    >>> stm = create_transport(cfg)
    >>> stm.connect()
    """
    if config.type == TransportType.CREATEC:
        from .createc import CreatecTransport
        return CreatecTransport(
            dispatch_mode=config.createc.dispatch_mode,
        )

    elif config.type == TransportType.NANONIS:
        from .nanonis import NanonisTransport
        return NanonisTransport(
            host=config.nanonis.host,
            port=config.nanonis.port,
        )

    elif config.type == TransportType.SIMULATOR:
        from .simulator import SimulatorTransport
        atoms = None
        if config.simulator.initial_atoms:
            atoms = np.array(config.simulator.initial_atoms)
        return SimulatorTransport(
            lattice_constant=config.simulator.lattice_constant,
            noise_level=config.simulator.noise_level,
            manipulation_success_rate=config.simulator.manipulation_success_rate,
            seed=config.simulator.seed,
            atom_positions=atoms,
        )

    else:
        raise ValueError(f"Unknown transport type: {config.type}")
