# AMRL Transport

**Hardware abstraction layer and task queue for autonomous atom manipulation.**

A proposed extension to [SINGROUP/Atom_manipulation_with_RL](https://github.com/SINGROUP/Atom_manipulation_with_RL) that decouples the RL agent from specific STM hardware, enabling:

- **Multi-vendor support** — swap between Createc, Nanonis, or any STM without changing RL code
- **Simulation-first development** — train and test RL agents without lab access
- **Distributed workflows** — queue manipulation tasks across multiple STM instruments
- **Lower barrier to entry** — labs with different hardware can contribute to and use the same RL models

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                    Task Client                        │
│  amrl-submit --atoms '[[0,0],[1,0],[0.5,0.87]]'      │
└──────────────────────┬────────────────────────────────┘
                       │ ManipulationTask (JSON)
                       ▼
              ┌─────────────────┐
              │    RabbitMQ     │
              │  amrl.tasks Q   │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Worker 01│   │Worker 02│   │Worker 03│
   │ Createc │   │ Nanonis │   │Simulator│
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        ▼              ▼              ▼
   STMTransport   STMTransport   STMTransport
   (protocol.py)  (protocol.py)  (protocol.py)
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌──────────┐
   │Createc  │   │Nanonis  │   │Simulated │
   │COM/Win  │   │TCP/IP   │   │Surface   │
   └─────────┘   └─────────┘   └──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     Redis       │
              │  Results + TTL  │
              └─────────────────┘
```

## Quick Start

### Install

```bash
# Core (numpy + pydantic only)
pip install -e .

# With task queue support
pip install -e ".[queue]"

# With ML dependencies (for SINGROUP RL agent integration)
pip install -e ".[all]"
```

### Run with simulator (no hardware needed)

```python
from amrl_transport.transport import SimulatorTransport
import numpy as np

with SimulatorTransport(seed=42) as stm:
    # Scan
    result = stm.scan_image(
        size_nm=5.0,
        offset_nm=np.array([0.0, 0.0]),
        pixel=128,
        bias_mv=100,
    )
    print(f"Image shape: {result.img_forward.shape}")

    # Manipulate
    manip = stm.lateral_manipulation(
        x_start_nm=0.0, y_start_nm=0.0,
        x_end_nm=1.0, y_end_nm=0.0,
        bias_mv=20, current_pa=57000,
        offset_nm=np.array([0, 0]), size_nm=5.0,
    )
    print(f"Current trace length: {len(manip.current)}")
```

### Use as RL environment (drop-in for RealExpEnv)

```python
from amrl_transport.transport import SimulatorTransport
from amrl_transport.integration import TransportEnv
from AMRL import sac_agent  # SINGROUP's RL agent

stm = SimulatorTransport(seed=42)
stm.connect()

env = TransportEnv(
    transport=stm,
    step_nm=0.2,
    max_mvolt=20,
    max_pcurrent_to_mvolt_ratio=2850,
    goal_nm=2.0,
    # ... same parameters as RealExpEnv
)

# Existing training code works as-is
state, info = env.reset()
next_state, reward, done, info = env.step(action)
```

### Distributed task queue

```bash
# Start infrastructure
docker compose up -d

# Start a worker
python -m amrl_transport.cli worker --transport simulator --worker-id sim-01

# Submit a task (from another terminal)
python -m amrl_transport.cli submit --atoms '[[0,0],[1,0],[0.5,0.866]]'

# Check status
python -m amrl_transport.cli status
```

## Implementing a new adapter

To support a new STM brand, implement the `STMTransport` ABC:

```python
from amrl_transport.transport.protocol import STMTransport, ScanResult

class MySTMTransport(STMTransport):
    def connect(self) -> None:
        # Your connection logic
        ...

    def scan_image(self, size_nm, offset_nm, pixel, bias_mv, speed=None) -> ScanResult:
        # Your scan logic — return ScanResult with numpy arrays
        ...

    def lateral_manipulation(self, x_start_nm, y_start_nm, ...) -> ManipulationResult:
        # Your manipulation logic
        ...

    # ... implement remaining abstract methods

    @property
    def name(self) -> str:
        return "My Custom STM"
```

Then register it in `transport/factory.py` and you're done. All RL code, the task queue, and the CLI will work with your adapter.

## Project Structure

```
amrl_transport/
├── transport/
│   ├── protocol.py     # STMTransport ABC — the core interface
│   ├── createc.py      # Createc COM adapter (Windows)
│   ├── nanonis.py      # Nanonis TCP adapter (stub, cross-platform)
│   ├── simulator.py    # Full simulator for dev/testing
│   └── factory.py      # Config → transport instance
├── queue/
│   ├── models.py       # Task/Result Pydantic models
│   ├── worker.py       # RabbitMQ consumer with graceful shutdown
│   └── client.py       # Task submission + result retrieval
├── config/
│   └── models.py       # Pydantic configuration models
├── integration.py      # TransportEnv — drop-in for RealExpEnv
└── cli.py              # CLI entry points
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests use the simulator backend — no hardware or external services needed.

## Contributing

This project aims to accelerate autonomous atom manipulation by making the
software infrastructure accessible to all labs. Contributions welcome:

1. **New adapters** — Scienta Omicron, RHK, SPECS, your custom setup
2. **Nanonis implementation** — fill in the TCP command stubs
3. **Improved simulator** — multi-atom physics, thermal drift, tip degradation
4. **RL improvements** — integrate with the latest SINGROUP training code
5. **Documentation** — tutorials for specific STM brands

## Acknowledgments

Built on top of the foundational work by SINGROUP (Aalto University):
- [Atom_manipulation_with_RL](https://github.com/SINGROUP/Atom_manipulation_with_RL)
- [AutoOSS](https://github.com/SINGROUP/AutoOSS)

Inspired by [DeepSPM](https://github.com/abred/DeepSPM) autonomous operation framework.

## License

MIT
