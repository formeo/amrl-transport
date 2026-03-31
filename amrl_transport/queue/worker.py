"""
Task queue worker for STM manipulation.

Connects to RabbitMQ, consumes ManipulationTask messages, executes them
on the configured STM transport, and publishes results to Redis.

Follows the patterns from Roman's dp_rabbit work:
- Graceful shutdown via SIGTERM/SIGINT
- Heartbeat-aware connection management
- Exponential backoff on reconnection
- Structured logging
"""
from __future__ import annotations

import logging
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Callable

from ..config.models import WorkerConfig
from ..transport.factory import create_transport
from ..transport.protocol import STMTransport
from .models import ManipulationResult, ManipulationTask, TaskStatus

logger = logging.getLogger(__name__)


class Worker:
    """
    RabbitMQ-based task worker for STM manipulation.

    Usage:
        config = WorkerConfig(...)
        worker = Worker(config)
        worker.run()  # blocks until SIGTERM

    The worker handles:
    1. Connecting to RabbitMQ and STM hardware
    2. Consuming tasks from the queue
    3. Executing manipulation via the transport layer
    4. Publishing results to Redis
    5. Graceful shutdown on SIGTERM/SIGINT

    Parameters
    ----------
    config : WorkerConfig
        Full worker configuration.
    task_handler : callable, optional
        Custom task handler function. If None, uses a default
        placeholder. In production, this would integrate with
        the RL agent from Atom_manipulation_with_RL.
    """

    def __init__(
        self,
        config: WorkerConfig,
        task_handler: Callable | None = None,
    ) -> None:
        self._config = config
        self._task_handler = task_handler or self._default_task_handler
        self._transport: STMTransport | None = None
        self._connection = None
        self._channel = None
        self._should_stop = False
        self._current_task_id: str | None = None

    def run(self) -> None:
        """Start the worker. Blocks until SIGTERM/SIGINT."""
        self._setup_signals()
        self._setup_logging()

        logger.info(
            "Worker %s starting (transport: %s, queue: %s)",
            self._config.worker_id,
            self._config.transport.type.value,
            self._config.queue.queue_name,
        )

        # Connect to STM hardware
        self._transport = create_transport(self._config.transport)
        self._transport.connect()
        logger.info("STM transport connected: %s", self._transport.name)

        # Connect to RabbitMQ with retry
        backoff = 1
        while not self._should_stop:
            try:
                self._connect_broker()
                logger.info("Broker connected, starting consumption")
                backoff = 1  # reset on success
                self._consume()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error("Broker error: %s, reconnecting in %ds", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

        self._shutdown()

    def _connect_broker(self) -> None:
        """Connect to RabbitMQ."""
        try:
            import pika
        except ImportError:
            raise RuntimeError(
                "pika is required for the task queue. "
                "Install: pip install pika"
            )

        params = pika.URLParameters(self._config.queue.broker_url)
        params.heartbeat = self._config.queue.heartbeat
        params.blocked_connection_timeout = 300

        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()

        # Declare queue (idempotent)
        self._channel.queue_declare(
            queue=self._config.queue.queue_name,
            durable=True,
            arguments={
                "x-max-priority": 10,  # priority queue
            },
        )

        # Fair dispatch — one message at a time
        self._channel.basic_qos(prefetch_count=self._config.queue.prefetch_count)

    def _consume(self) -> None:
        """Consume and process messages."""
        def on_message(ch, method, properties, body):
            if self._should_stop:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                return

            try:
                task = ManipulationTask.model_validate_json(body)
                self._current_task_id = task.task_id
                logger.info(
                    "Task %s received: %d atoms, requester=%s",
                    task.task_id,
                    len(task.targets),
                    task.requester,
                )

                result = self._execute_task(task)
                self._store_result(result)

                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(
                    "Task %s completed: %d/%d atoms placed, mean precision %.3f nm",
                    result.task_id,
                    result.atoms_placed,
                    result.atoms_total,
                    result.mean_precision_nm,
                )

            except Exception as e:
                logger.error("Task processing error: %s", e, exc_info=True)
                # Nack and requeue on failure (up to broker retry policy)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            finally:
                self._current_task_id = None

        self._channel.basic_consume(
            queue=self._config.queue.queue_name,
            on_message_callback=on_message,
        )

        try:
            self._channel.start_consuming()
        except Exception:
            if not self._should_stop:
                raise

    def _execute_task(self, task: ManipulationTask) -> ManipulationResult:
        """Execute a manipulation task using the transport and RL agent."""
        started_at = datetime.now(timezone.utc)
        try:
            result = self._task_handler(task, self._transport)
            result.worker_id = self._config.worker_id
            result.started_at = started_at
            result.completed_at = datetime.now(timezone.utc)
            return result
        except Exception as e:
            return ManipulationResult(
                task_id=task.task_id,
                worker_id=self._config.worker_id,
                status=TaskStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                atoms_total=len(task.targets),
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )

    def _store_result(self, result: ManipulationResult) -> None:
        """Store result in Redis."""
        try:
            import redis
        except ImportError:
            logger.warning("redis not installed, skipping result storage")
            return

        try:
            r = redis.from_url(self._config.queue.result_backend)
            key = f"amrl:result:{result.task_id}"
            r.set(key, result.model_dump_json(), ex=86400 * 7)  # TTL 7 days
            logger.debug("Result stored: %s", key)
        except Exception as e:
            logger.error("Failed to store result: %s", e)

    def _default_task_handler(
        self, task: ManipulationTask, transport: STMTransport
    ) -> ManipulationResult:
        """
        Placeholder task handler.

        In production, this integrates with the SINGROUP RL agent:
            from AMRL import sac_agent
            from AMRL import RealExpEnv  # → replaced by transport-backed env

        For now, just logs the task and returns a mock result.
        """
        logger.info(
            "Default handler: would assemble %d atoms using %s",
            len(task.targets),
            transport.name,
        )

        # Placeholder: pretend we processed it
        return ManipulationResult(
            task_id=task.task_id,
            worker_id=self._config.worker_id,
            status=TaskStatus.COMPLETED,
            atoms_total=len(task.targets),
            atoms_placed=0,
            mean_precision_nm=0.0,
        )

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def handle_stop(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info("Received %s, initiating graceful shutdown...", sig_name)
            self._should_stop = True
            if self._channel and self._channel.is_open:
                self._channel.stop_consuming()

        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)

    def _setup_logging(self) -> None:
        """Configure structured logging."""
        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper(), logging.INFO),
            format=(
                "%(asctime)s | %(levelname)-8s | "
                f"worker={self._config.worker_id} | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
            stream=sys.stdout,
        )

    def _shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down worker %s...", self._config.worker_id)

        if self._channel and self._channel.is_open:
            try:
                self._channel.close()
            except Exception:
                pass

        if self._connection and self._connection.is_open:
            try:
                self._connection.close()
            except Exception:
                pass

        if self._transport:
            try:
                self._transport.disconnect()
            except Exception:
                pass

        logger.info("Worker %s stopped", self._config.worker_id)
