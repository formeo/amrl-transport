"""
Scan-frame convention shared by all transports (Createc/SINGROUP):
X spans [offset_x - L/2, offset_x + L/2), Y spans [offset_y, offset_y + L),
row index grows with Y, column index grows with X.
"""
import numpy as np
import pytest

from amrl_transport.integration import TransportEnv
from amrl_transport.transport import SimulatorTransport
from amrl_transport.transport.createc import _DAC_UNIT, _VOLT_UNIT, CreatecTransport

SIZE_NM = 5.0
PIXEL = 64
OFFSET = np.array([1.0, -2.0])


class FakeCreatecCOM:
    """Only the parameters `_nm_to_pixel` reads."""

    def __init__(self, size_nm, pixel, gain=10.0, piezo=10.0):
        pixel_to_a = size_nm * 10 / pixel
        self.params = {
            "GainX": gain,
            "Xpiezoconst": piezo,
            "Delta X [Dac]": pixel_to_a * _DAC_UNIT / (_VOLT_UNIT * gain * piezo),
        }

    def getparam(self, key):
        return str(self.params[key])


def createc_pixel(x_nm, y_nm):
    stm = CreatecTransport()
    stm._stm = FakeCreatecCOM(SIZE_NM, PIXEL)
    col, row, _, _ = stm._nm_to_pixel(x_nm, y_nm, None, None, OFFSET, SIZE_NM)
    return row, col


def simulator_peak(atom):
    stm = SimulatorTransport(atom_positions=np.array([atom]), noise_level=0.0, seed=0)
    stm.connect()
    img = stm.scan_image(SIZE_NM, OFFSET, PIXEL, 100).img_forward
    return np.unravel_index(np.argmax(img), img.shape)


@pytest.mark.parametrize("atom", [(1.0, -2.0), (0.2, 0.5), (2.9, 1.7), (-0.55, -1.05)])
def test_simulator_matches_createc_pixel_frame(atom):
    row, col = simulator_peak(atom)
    assert (row, col) == createc_pixel(*atom)


def test_atom_at_offset_is_top_centre():
    row, col = simulator_peak(tuple(OFFSET))
    assert row == 0
    assert col == PIXEL // 2


@pytest.mark.parametrize("atom", [(0.7, -0.3), (2.0, 2.2)])
def test_transport_env_fallback_detector_uses_same_frame(atom):
    stm = SimulatorTransport(atom_positions=np.array([atom]), noise_level=0.001, seed=0)
    stm.connect()
    env = TransportEnv(
        transport=stm, step_nm=0.2, max_mvolt=20, max_pcurrent_to_mvolt_ratio=2850,
        goal_nm=1.0, template=np.zeros((16, 16)), current_jump=4, im_size_nm=SIZE_NM,
        offset_nm=OFFSET, manip_limit_nm=np.array([-3.0, 3.0, -3.0, 3.0]), pixel=PIXEL,
        template_max_y=30, scan_mV=100, max_len=5, load_weight="dummy.pth",
    )
    abs_nm, _ = env._scan_atom()
    assert np.allclose(abs_nm, atom, atol=SIZE_NM / PIXEL)
