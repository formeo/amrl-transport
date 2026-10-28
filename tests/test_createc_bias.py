"""CreatecTransport.ramp_bias against a fake COM object (no Windows needed)."""
import numpy as np
import pytest

from amrl_transport.transport import createc
from amrl_transport.transport.createc import CreatecTransport

KEY = "Biasvolt.[mV]"


class FakeCOM:
    def __init__(self, bias_mv):
        self.bias = float(bias_mv)
        self.history = []

    def getparam(self, key):
        assert key == KEY
        return str(self.bias)

    def setparam(self, key, value):
        assert key == KEY
        self.bias = float(value)
        self.history.append(self.bias)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(createc.time, "sleep", lambda s: None)


def ramp(init, target):
    stm = CreatecTransport()
    stm._stm = FakeCOM(init)
    stm.ramp_bias(target)
    return stm._stm


@pytest.mark.parametrize("init,target", [
    (100, 0), (-100, 0), (0, 100), (0, -100), (100, 50), (-100, 100), (2000, -300),
])
def test_reaches_target(init, target):
    assert ramp(init, target).bias == pytest.approx(target)


@pytest.mark.parametrize("init,target", [(100, 0), (0, 100), (-500, 0), (0, -500)])
def test_ramp_through_zero_is_gradual_and_monotonic(init, target):
    hist = np.array(ramp(init, target).history)
    assert len(hist) > 2
    steps = np.diff(np.concatenate([[init], hist]))
    assert np.all(steps * np.sign(target - init) >= 0)


def test_same_value_is_noop():
    assert ramp(100, 100).history == []
