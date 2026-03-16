"""
CLI entry points for AMRL Transport.

Usage:
    # Start a worker with simulator backend
    python -m amrl_transport.cli worker --transport simulator

    # Submit a task
    python -m amrl_transport.cli submit --atoms '[[0,0],[1,0],[0.5,0.866]]'

    # Check queue status
    python -m amrl_transport.cli status
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np


def run_worker(args=None):
    """Start a task queue worker."""
    parser = argparse.ArgumentParser(description="AMRL Task Worker")
    parser.add_argument(
        "--transport", choices=["createc", "nanonis", "simulator"],
        default="simulator", help="STM transport backend",
    )
    parser.add_argument("--broker", default="amqp://guest:guest@localhost:5672/")
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    parser.add_argument("--queue", default="amrl.tasks")
    parser.add_argument("--worker-id", default="worker-01")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--seed", type=int, default=None, help="Simulator seed")
    opts = parser.parse_args(args)

    from .config.models import (
        QueueConfig,
        SimulatorConfig,
        TransportConfig,
        TransportType,
        WorkerConfig,
    )
    from .queue.worker import Worker

    transport_cfg = TransportConfig(
        type=TransportType(opts.transport),
        simulator=SimulatorConfig(seed=opts.seed),
    )
    queue_cfg = QueueConfig(
        broker_url=opts.broker,
        result_backend=opts.redis,
        queue_name=opts.queue,
    )
    worker_cfg = WorkerConfig(
        transport=transport_cfg,
        queue=queue_cfg,
        worker_id=opts.worker_id,
        log_level=opts.log_level,
    )

    worker = Worker(worker_cfg)
    worker.run()


def submit_task(args=None):
    """Submit a manipulation task."""
    parser = argparse.ArgumentParser(description="Submit AMRL Task")
    parser.add_argument(
        "--atoms", required=True, type=str,
        help='Target atom positions as JSON: [[x1,y1],[x2,y2],...]',
    )
    parser.add_argument("--broker", default="amqp://guest:guest@localhost:5672/")
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    parser.add_argument("--queue", default="amrl.tasks")
    parser.add_argument("--requester", default="cli")
    parser.add_argument("--priority", type=int, default=5)
    opts = parser.parse_args(args)

    from .config.models import QueueConfig
    from .queue.client import TaskClient
    from .queue.models import AtomTarget, ManipulationTask

    positions = json.loads(opts.atoms)
    targets = [AtomTarget(x_nm=p[0], y_nm=p[1]) for p in positions]
    task = ManipulationTask(
        targets=targets,
        requester=opts.requester,
        priority=opts.priority,
    )

    queue_cfg = QueueConfig(
        broker_url=opts.broker,
        result_backend=opts.redis,
        queue_name=opts.queue,
    )
    client = TaskClient(queue_cfg)
    task_id = client.submit(task)
    print(f"Submitted task {task_id} ({len(targets)} atoms)")


def check_status(args=None):
    """Check queue status."""
    parser = argparse.ArgumentParser(description="AMRL Queue Status")
    parser.add_argument("--broker", default="amqp://guest:guest@localhost:5672/")
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    parser.add_argument("--queue", default="amrl.tasks")
    opts = parser.parse_args(args)

    from .config.models import QueueConfig
    from .queue.client import TaskClient

    queue_cfg = QueueConfig(
        broker_url=opts.broker,
        result_backend=opts.redis,
        queue_name=opts.queue,
    )
    client = TaskClient(queue_cfg)
    pending = client.list_pending()
    print(f"Queue '{opts.queue}': {pending} pending task(s)")


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        print("Usage: python -m amrl_transport.cli {worker|submit|status}")
        sys.exit(1)

    cmd = sys.argv[1]
    remaining = sys.argv[2:]

    if cmd == "worker":
        run_worker(remaining)
    elif cmd == "submit":
        submit_task(remaining)
    elif cmd == "status":
        check_status(remaining)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
