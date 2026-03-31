"""
Task and result models for the manipulation queue.

These models define the message format exchanged between
the task client (lab operator / orchestrator) and workers
(STM controllers running RL agents).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AtomTarget(BaseModel):
    """Single atom target position."""
    x_nm: float
    y_nm: float
    element: str = Field("Ag", description="Atom element symbol")


class ManipulationTask(BaseModel):
    """
    A task to assemble atoms into a target structure.

    Published to the queue by the client, consumed by workers.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # What to build
    targets: list[AtomTarget] = Field(..., description="Target atom positions")

    # Scan parameters
    scan_size_nm: float = Field(10.0)
    scan_pixel: int = Field(128)
    scan_bias_mv: float = Field(100.0)

    # Manipulation parameters
    max_episodes_per_atom: int = Field(50, description="Max RL episodes per atom")
    manipulation_voltage_mv: float = Field(20.0)
    manipulation_current_pa: float = Field(57000.0)

    # Safety limits
    manipulation_limit_nm: list[float] | None = Field(
        None, description="[x_min, x_max, y_min, y_max]"
    )

    # Metadata
    requester: str = Field("unknown", description="Who submitted the task")
    priority: int = Field(5, description="1 (highest) to 10 (lowest)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AtomResult(BaseModel):
    """Result for a single atom placement."""
    target: AtomTarget
    final_position_nm: list[float]
    distance_to_target_nm: float
    episodes_used: int
    success: bool


class ManipulationResult(BaseModel):
    """
    Result of a completed manipulation task.

    Stored in Redis and published back to the result queue.
    """
    task_id: str
    worker_id: str
    status: TaskStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Per-atom results
    atom_results: list[AtomResult] = Field(default_factory=list)

    # Aggregate stats
    total_episodes: int = 0
    atoms_placed: int = 0
    atoms_total: int = 0
    mean_precision_nm: float = 0.0

    # Error info
    error_message: str | None = None
    error_traceback: str | None = None

    # Raw data paths (for later analysis)
    scan_data_path: str | None = None
    trajectory_data_path: str | None = None
