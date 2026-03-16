"""
Tests for the transport layer.

All tests use the SimulatorTransport — no hardware needed.
Run: pytest tests/ -v
"""
import numpy as np
import pytest

from amrl_transport.config.models import TransportConfig, TransportType, SimulatorConfig
from amrl_transport.transport import SimulatorTransport, create_transport
from amrl_transport.transport.protocol import ScanResult, ManipulationResult, TipPosition


# ── Simulator basic operations ────────────────────────────────


class TestSimulatorTransport:
    def setup_method(self):
        self.stm = SimulatorTransport(seed=42)
        self.stm.connect()

    def teardown_method(self):
        self.stm.disconnect()

    def test_connect_disconnect(self):
        assert self.stm.is_connected()
        self.stm.disconnect()
        assert not self.stm.is_connected()

    def test_context_manager(self):
        with SimulatorTransport(seed=42) as stm:
            assert stm.is_connected()
        assert not stm.is_connected()

    def test_scan_image_shape(self):
        result = self.stm.scan_image(
            size_nm=5.0,
            offset_nm=np.array([0.0, 0.0]),
            pixel=64,
            bias_mv=100,
        )
        assert isinstance(result, ScanResult)
        assert result.img_forward.shape == (64, 64)
        assert result.img_backward.shape == (64, 64)
        np.testing.assert_array_equal(result.size_nm, [5.0, 5.0])

    def test_scan_image_has_atom_signal(self):
        """Atom at (0,0) should produce a peak in center of image."""
        result = self.stm.scan_image(
            size_nm=2.0,
            offset_nm=np.array([0.0, 0.0]),
            pixel=128,
            bias_mv=100,
        )
        img = result.img_forward
        center_region = img[48:80, 48:80]
        edge_region = img[0:16, 0:16]
        assert center_region.mean() > edge_region.mean()

    def test_scan_image_deterministic_with_seed(self):
        """Same seed → same images."""
        stm1 = SimulatorTransport(seed=123)
        stm1.connect()
        stm2 = SimulatorTransport(seed=123)
        stm2.connect()
        r1 = stm1.scan_image(5.0, np.array([0, 0]), 64, 100)
        r2 = stm2.scan_image(5.0, np.array([0, 0]), 64, 100)
        np.testing.assert_array_equal(r1.img_forward, r2.img_forward)
        stm1.disconnect()
        stm2.disconnect()

    def test_lateral_manipulation_returns_result(self):
        result = self.stm.lateral_manipulation(
            x_start_nm=0.0,
            y_start_nm=0.0,
            x_end_nm=1.0,
            y_end_nm=0.0,
            bias_mv=20,
            current_pa=57000,
            offset_nm=np.array([0.0, 0.0]),
            size_nm=10.0,
        )
        assert result is not None
        assert len(result.current) == 256
        assert len(result.x) == 256
        assert len(result.time) == 256

    def test_lateral_manipulation_same_point_returns_none(self):
        result = self.stm.lateral_manipulation(
            x_start_nm=1.0, y_start_nm=1.0,
            x_end_nm=1.0, y_end_nm=1.0,
            bias_mv=20, current_pa=50000,
            offset_nm=np.array([0, 0]), size_nm=10,
        )
        assert result is None

    def test_tip_position(self):
        self.stm.set_tip_position(3.5, -2.1)
        pos = self.stm.get_tip_position()
        assert isinstance(pos, TipPosition)
        assert abs(pos.x_nm - 3.5) < 1e-10
        assert abs(pos.y_nm - (-2.1)) < 1e-10

    def test_bias_ramp(self):
        self.stm.ramp_bias(200.0)
        assert abs(self.stm.get_bias() - 200.0) < 1e-10

    def test_name(self):
        assert self.stm.name == "Simulator"

    def test_capabilities(self):
        caps = self.stm.capabilities
        assert caps["lateral_manipulation"] is True
        assert caps["tip_forming"] is True

    def test_atom_positions_tracking(self):
        """Manipulation should change atom positions."""
        initial = self.stm.atom_positions.copy()
        # Run many manipulations to overcome success rate randomness
        for _ in range(20):
            self.stm.lateral_manipulation(
                x_start_nm=0.0, y_start_nm=0.0,
                x_end_nm=2.0, y_end_nm=0.0,
                bias_mv=20, current_pa=100000,
                offset_nm=np.array([0, 0]), size_nm=10,
            )
        final = self.stm.atom_positions
        # With high current and 20 attempts, at least one should succeed
        assert not np.allclose(initial, final, atol=0.01)

    def test_add_atom(self):
        self.stm.add_atom(5.0, 5.0)
        assert len(self.stm.atom_positions) == 2

    def test_reset_atoms(self):
        new_pos = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        self.stm.reset_atoms(new_pos)
        assert len(self.stm.atom_positions) == 3


# ── Factory ───────────────────────────────────────────────────


class TestFactory:
    def test_create_simulator(self):
        cfg = TransportConfig(
            type=TransportType.SIMULATOR,
            simulator=SimulatorConfig(seed=42, noise_level=0.01),
        )
        stm = create_transport(cfg)
        assert isinstance(stm, SimulatorTransport)

    def test_create_with_initial_atoms(self):
        cfg = TransportConfig(
            type=TransportType.SIMULATOR,
            simulator=SimulatorConfig(
                initial_atoms=[[1.0, 2.0], [3.0, 4.0]],
            ),
        )
        stm = create_transport(cfg)
        stm.connect()
        assert len(stm.atom_positions) == 2
        stm.disconnect()

    def test_create_unknown_raises(self):
        cfg = TransportConfig(type=TransportType.SIMULATOR)
        cfg.type = "bogus"
        with pytest.raises(ValueError):
            create_transport(cfg)


# ── Integration: TransportEnv ─────────────────────────────────


class TestTransportEnv:
    """Test the RL environment integration layer with the simulator."""

    def test_full_episode(self):
        """Run a complete reset → step → done cycle."""
        from amrl_transport.integration import TransportEnv

        stm = SimulatorTransport(
            seed=42,
            atom_positions=np.array([[0.0, 0.0]]),
            manipulation_success_rate=1.0,  # always succeed for test
            noise_level=0.001,
        )
        stm.connect()

        template = np.zeros((16, 16))  # dummy template

        env = TransportEnv(
            transport=stm,
            step_nm=0.2,
            max_mvolt=20,
            max_pcurrent_to_mvolt_ratio=2850,
            goal_nm=1.0,
            template=template,
            current_jump=4,
            im_size_nm=5.0,
            offset_nm=np.array([0.0, 0.0]),
            manip_limit_nm=np.array([-3.0, 3.0, -3.0, 3.0]),
            pixel=64,
            template_max_y=30,
            scan_mV=100,
            max_len=5,
            load_weight="dummy.pth",
        )

        state, info = env.reset()
        assert state is not None
        assert len(state) == 4  # [goal_x, goal_y, displacement_x, displacement_y]
        assert "start_absolute_nm" in info

        # Take a random step
        action = np.array([0.1, 0.1, 0.5, 0.0, 0.8, 0.7])
        next_state, reward, done, info = env.step(action)
        assert next_state is not None
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, bool)

        stm.disconnect()
