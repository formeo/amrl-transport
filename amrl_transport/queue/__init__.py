"""
EXPERIMENTAL / FROZEN — not for use with real instruments.

The default worker handler is a placeholder (reports COMPLETED, places no
atoms). Long tasks run inside the pika callback, so heartbeats and the
RabbitMQ consumer timeout can cause redelivery and repeated execution;
malformed messages are requeued forever. Kept for reference only.
"""
from .client import TaskClient
from .models import (
    AtomResult,
    AtomTarget,
    ManipulationResult,
    ManipulationTask,
    TaskStatus,
)
from .worker import Worker

__all__ = [
    "AtomResult",
    "AtomTarget",
    "ManipulationResult",
    "ManipulationTask",
    "TaskStatus",
    "TaskClient",
    "Worker",
]
