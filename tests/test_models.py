"""
Tests for queue models, config models, and edge cases.

Run: pytest tests/ -v
"""
from amrl_transport.config.models import (
    QueueConfig,
    SimulatorConfig,
    TransportConfig,
    TransportType,
    WorkerConfig,
)
from amrl_transport.queue.models import (
    AtomResult,
    AtomTarget,
    ManipulationResult,
    ManipulationTask,
    TaskStatus,
)


class TestAtomTarget:
    def test_create_default(self):
        t = AtomTarget(x_nm=1.0, y_nm=2.0)
        assert t.x_nm == 1.0
        assert t.y_nm == 2.0
        assert t.element == "Ag"

    def test_create_custom_element(self):
        t = AtomTarget(x_nm=0.0, y_nm=0.0, element="Cu")
        assert t.element == "Cu"


class TestManipulationTask:
    def test_create_minimal(self):
        task = ManipulationTask(
            targets=[AtomTarget(x_nm=0.0, y_nm=0.0)]
        )
        assert len(task.targets) == 1
        assert task.task_id  # auto-generated UUID
        assert task.created_at
        assert task.scan_size_nm == 10.0
        assert task.priority == 5

    def test_create_with_limits(self):
        task = ManipulationTask(
            targets=[
                AtomTarget(x_nm=0.0, y_nm=0.0),
                AtomTarget(x_nm=1.0, y_nm=0.0),
            ],
            manipulation_limit_nm=[-5.0, 5.0, -5.0, 5.0],
            requester="test_user",
            priority=1,
        )
        assert len(task.targets) == 2
        assert task.manipulation_limit_nm == [-5.0, 5.0, -5.0, 5.0]
        assert task.requester == "test_user"

    def test_serialization_roundtrip(self):
        task = ManipulationTask(
            targets=[AtomTarget(x_nm=1.5, y_nm=2.5)]
        )
        json_str = task.model_dump_json()
        restored = ManipulationTask.model_validate_json(json_str)
        assert restored.task_id == task.task_id
        assert restored.targets[0].x_nm == 1.5

    def test_metadata(self):
        task = ManipulationTask(
            targets=[AtomTarget(x_nm=0.0, y_nm=0.0)],
            metadata={"experiment": "test_run_01", "temperature_K": 4.2},
        )
        assert task.metadata["temperature_K"] == 4.2


class TestManipulationResult:
    def test_create_completed(self):
        result = ManipulationResult(
            task_id="test-123",
            worker_id="sim-01",
            status=TaskStatus.COMPLETED,
            atoms_placed=3,
            atoms_total=3,
            mean_precision_nm=0.05,
        )
        assert result.status == TaskStatus.COMPLETED
        assert result.atoms_placed == 3
        assert result.error_message is None

    def test_create_failed(self):
        result = ManipulationResult(
            task_id="test-456",
            worker_id="sim-01",
            status=TaskStatus.FAILED,
            error_message="Tip crashed",
            error_traceback="...",
        )
        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Tip crashed"

    def test_with_atom_results(self):
        atom_result = AtomResult(
            target=AtomTarget(x_nm=1.0, y_nm=2.0),
            final_position_nm=[1.02, 1.98],
            distance_to_target_nm=0.028,
            episodes_used=5,
            success=True,
        )
        result = ManipulationResult(
            task_id="test-789",
            worker_id="sim-01",
            status=TaskStatus.COMPLETED,
            atom_results=[atom_result],
            atoms_placed=1,
            atoms_total=1,
        )
        assert len(result.atom_results) == 1
        assert result.atom_results[0].success is True
        assert result.atom_results[0].distance_to_target_nm == 0.028


class TestTaskStatus:
    def test_enum_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"


class TestConfigModels:
    def test_simulator_config_defaults(self):
        cfg = SimulatorConfig()
        assert cfg.noise_level == 0.05
        assert cfg.manipulation_success_rate == 0.7
        assert cfg.seed is None

    def test_simulator_config_custom(self):
        cfg = SimulatorConfig(
            noise_level=0.1,
            seed=42,
            initial_atoms=[[0.0, 0.0], [1.0, 1.0]],
        )
        assert cfg.seed == 42
        assert len(cfg.initial_atoms) == 2

    def test_transport_config_simulator(self):
        cfg = TransportConfig(type=TransportType.SIMULATOR)
        assert cfg.type == TransportType.SIMULATOR

    def test_queue_config_defaults(self):
        cfg = QueueConfig()
        assert cfg.broker_url == "amqp://guest:guest@localhost:5672/"
        assert cfg.queue_name == "amrl.tasks"
        assert cfg.heartbeat == 60

    def test_worker_config(self):
        cfg = WorkerConfig(worker_id="test-01")
        assert cfg.worker_id == "test-01"
        assert cfg.log_level == "INFO"
        assert cfg.queue.broker_url == "amqp://guest:guest@localhost:5672/"
