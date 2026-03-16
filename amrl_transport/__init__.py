"""
AMRL Transport — Hardware abstraction and task queue for autonomous atom manipulation.

Provides:
- A unified STMTransport interface for any scanning probe microscope
- Adapters for Createc, Nanonis, and a built-in simulator
- A RabbitMQ/Redis task queue for distributed manipulation workflows
- Drop-in replacement for SINGROUP's Atom_manipulation_with_RL hardware layer

Quick start (simulator, no hardware needed):
    >>> from amrl_transport.transport import SimulatorTransport
    >>> with SimulatorTransport(seed=42) as stm:
    ...     result = stm.scan_image(size_nm=5.0, offset_nm=[0,0], pixel=128, bias_mv=100)
    ...     print(result.img_forward.shape)
    (128, 128)
"""

__version__ = "0.1.0"
