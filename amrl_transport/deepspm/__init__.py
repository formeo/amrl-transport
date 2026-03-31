"""
DeepSPM compatibility layer — Python replacement for LabVIEW server.

- InstrumentServer: async TCP server (replaces LabVIEW)
- DeepSPMClient: clean Python client
- EnvClientCompat: drop-in for envClient.EnvClient
"""
from .client import DeepSPMClient, EnvClientCompat
from .server import InstrumentServer, ServerConfig

__all__ = [
    "InstrumentServer", "ServerConfig",
    "DeepSPMClient", "EnvClientCompat",
]
