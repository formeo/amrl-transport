"""
Configuration models for the transport layer.

Uses Pydantic for validation, serialization, and environment variable support.
Configs can be loaded from YAML/JSON files or environment variables.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TransportType(str, Enum):
    CREATEC = "createc"
    NANONIS = "nanonis"
    SIMULATOR = "simulator"


class CreatecConfig(BaseModel):
    """Configuration for Createc STM."""
    dispatch_mode: str = Field("Dispatch", description="COM dispatch mode")


class NanonisConfig(BaseModel):
    """Configuration for Nanonis STM."""
    host: str = Field("localhost", description="Nanonis TCP server host")
    port: int = Field(6501, description="Nanonis TCP server port")


class SimulatorConfig(BaseModel):
    """Configuration for the simulator."""
    lattice_constant: float = Field(0.288, description="Lattice constant in nm")
    noise_level: float = Field(0.05, description="Scan noise std")
    manipulation_success_rate: float = Field(0.7, description="Manipulation success probability")
    seed: Optional[int] = Field(None, description="Random seed")
    initial_atoms: Optional[List[List[float]]] = Field(
        None, description="Initial atom positions [[x,y], ...]"
    )


class TransportConfig(BaseModel):
    """
    Top-level transport configuration.

    Example YAML:
        transport:
          type: simulator
          simulator:
            noise_level: 0.03
            seed: 42
    """
    type: TransportType = Field(TransportType.SIMULATOR, description="Transport backend")
    createc: CreatecConfig = Field(default_factory=CreatecConfig)
    nanonis: NanonisConfig = Field(default_factory=NanonisConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)


class ManipulationTaskConfig(BaseModel):
    """Configuration for a single manipulation task."""
    task_id: str = Field(..., description="Unique task identifier")
    design_positions_nm: List[List[float]] = Field(
        ..., description="Target positions for atoms [[x,y], ...]"
    )
    scan_size_nm: float = Field(10.0, description="Scan area size in nm")
    scan_pixel: int = Field(128, description="Scan pixels")
    scan_bias_mv: float = Field(100.0, description="Scan bias in mV")
    max_episodes: int = Field(100, description="Max manipulation episodes")
    manipulation_limit_nm: Optional[List[float]] = Field(
        None, description="[x_min, x_max, y_min, y_max] in nm"
    )


class QueueConfig(BaseModel):
    """Configuration for the task queue."""
    broker_url: str = Field(
        "amqp://guest:guest@localhost:5672/",
        description="RabbitMQ broker URL",
    )
    result_backend: str = Field(
        "redis://localhost:6379/0",
        description="Redis URL for storing results",
    )
    queue_name: str = Field(
        "amrl.tasks",
        description="Queue name for manipulation tasks",
    )
    prefetch_count: int = Field(1, description="Messages to prefetch (1 = fair dispatch)")
    heartbeat: int = Field(60, description="AMQP heartbeat interval in seconds")


class WorkerConfig(BaseModel):
    """Full worker configuration."""
    transport: TransportConfig = Field(default_factory=TransportConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    worker_id: str = Field("worker-01", description="Unique worker identifier")
    log_level: str = Field("INFO", description="Logging level")
