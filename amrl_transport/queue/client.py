"""
Client for submitting manipulation tasks and retrieving results.

Usage:
    client = TaskClient(queue_config)
    task_id = client.submit(task)
    result = client.wait_for_result(task_id, timeout=3600)
"""
from __future__ import annotations

import logging
import time

from ..config.models import QueueConfig
from .models import ManipulationResult, ManipulationTask

logger = logging.getLogger(__name__)


class TaskClient:
    """
    Client for the AMRL task queue.

    Publishes ManipulationTask messages to RabbitMQ and polls
    Redis for results.

    Parameters
    ----------
    config : QueueConfig
        Queue configuration (broker URL, result backend, etc.)
    """

    def __init__(self, config: QueueConfig) -> None:
        self._config = config

    def submit(self, task: ManipulationTask) -> str:
        """
        Submit a manipulation task to the queue.

        Parameters
        ----------
        task : ManipulationTask
            The task to submit.

        Returns
        -------
        str
            The task ID.
        """
        import pika

        params = pika.URLParameters(self._config.broker_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        # Ensure queue exists
        channel.queue_declare(
            queue=self._config.queue_name,
            durable=True,
            arguments={"x-max-priority": 10},
        )

        channel.basic_publish(
            exchange="",
            routing_key=self._config.queue_name,
            body=task.model_dump_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,  # persistent
                priority=task.priority,
                content_type="application/json",
                message_id=task.task_id,
            ),
        )

        connection.close()
        logger.info("Task %s submitted (%d atoms)", task.task_id, len(task.targets))
        return task.task_id

    def get_result(self, task_id: str) -> ManipulationResult | None:
        """
        Get the result for a task (non-blocking).

        Returns None if not yet available.
        """
        import redis

        r = redis.from_url(self._config.result_backend)
        key = f"amrl:result:{task_id}"
        data = r.get(key)
        if data is None:
            return None
        return ManipulationResult.model_validate_json(data)

    def wait_for_result(
        self,
        task_id: str,
        timeout: float = 3600,
        poll_interval: float = 5.0,
    ) -> ManipulationResult:
        """
        Wait for a task result (blocking).

        Parameters
        ----------
        task_id : str
            Task ID to wait for.
        timeout : float
            Maximum wait time in seconds (default 1 hour).
        poll_interval : float
            Polling interval in seconds.

        Returns
        -------
        ManipulationResult

        Raises
        ------
        TimeoutError
            If result not available within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.get_result(task_id)
            if result is not None:
                return result
            time.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    def list_pending(self) -> int:
        """Get number of pending messages in the queue."""
        import pika

        params = pika.URLParameters(self._config.broker_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        result = channel.queue_declare(
            queue=self._config.queue_name,
            durable=True,
            passive=True,  # don't create, just check
        )
        count = result.method.message_count
        connection.close()
        return count
