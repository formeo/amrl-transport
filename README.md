[![PyPI](https://img.shields.io/pypi/v/amrl-transport.svg)](https://pypi.org/project/amrl-transport/)
[![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19049387.svg)](https://doi.org/10.5281/zenodo.19049387)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/amrl-transport/month)](https://pepy.tech/projects/amrl-transport)

# AMRL Transport

**Hardware abstraction layer for autonomous atom manipulation with STM/AFM.**

Drop-in replacement for [SINGROUP](https://github.com/SINGROUP/Atom_manipulation_with_RL)'s `RealExpEnv` — train RL agents on a simulator, deploy on real hardware without code changes. Also replaces the [DeepSPM](https://github.com/abred/DeepSPM) LabVIEW instrument server with a pure Python asyncio implementation.

## The problem

RL frameworks for atom manipulation ([DeepSPM](https://github.com/abred/DeepSPM), [SINGROUP/AMRL](https://github.com/SINGROUP/Atom_manipulation_with_RL)) are hardwired to specific STM controllers — Createc COM on Windows, Nanonis TCP, or custom LabVIEW servers. If your lab has different hardware, you rewrite everything. If you want to develop without a $500K microscope, you can't.

## What this solves

- **`STMTransport` ABC** — 12 methods with physical units (nm, mV, pA). Implement once per vendor, use any RL agent.
- **Simulator backend** — Gaussian atom physics, tip-sample interaction, configurable noise. No hardware, no Windows, no license fees.
- **DeepSPM server replacement** — pure Python asyncio server that speaks the DeepSPM wire protocol. Replaces LabVIEW (BSD-3, Windows-only, ~$3500/yr). Existing DeepSPM agent code works unchanged — just point it at the Python server.
- **Distributed task queue** — RabbitMQ + Redis for queuing manipulation tasks across multiple instruments with retry, backoff, and graceful shutdown.
- **`TransportEnv`** — drop-in for SINGROUP's `RealExpEnv`. Existing training code works as-is.

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

## Quick start

### Install

```bash
# Core (numpy + pydantic only)
pip install amrl-transport

# With task queue support
pip install amrl-transport[queue]

# With ML dependencies (for SINGROUP RL agent integration)
pip install amrl-transport[all]
```

### Run with simulator (no hardware needed)

```python
from amrl_transport.transport import SimulatorTransport
import numpy as np

with SimulatorTransport(seed=42) as stm:
    # Scan a 5nm × 5nm area
    result = stm.scan_image(
        size_nm=5.0,
        offset_nm=np.array([0.0, 0.0]),
        pixel=128,
        bias_mv=100,
    )
    print(f"Image shape: {result.img_forward.shape}")

    # Lateral manipulation
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

stm = SimulatorTransport(seed=42)
stm.connect()

env = TransportEnv(
    transport=stm,
    step_nm=0.2,
    max_mvolt=20,
    max_pcurrent_to_mvolt_ratio=2850,
    goal_nm=2.0,
)

# Existing SINGROUP training code works as-is
state, info = env.reset()
next_state, reward, done, info = env.step(action)
```

### DeepSPM server (replaces LabVIEW)

```bash
# Start the Python instrument server with simulator backend
python -m amrl_transport.deepspm --transport simulator

# Existing DeepSPM agent connects to localhost:5556 — no code changes
```

### Distributed task queue

```bash
# Start infrastructure
docker compose up -d

# Start a worker
python -m amrl_transport.cli worker --transport simulator --worker-id sim-01

# Submit a task
python -m amrl_transport.cli submit --atoms '[[0,0],[1,0],[0.5,0.866]]'
```

## Implementing a new adapter

To support a new STM brand, implement the `STMTransport` ABC:

```python
from amrl_transport.transport.protocol import STMTransport, ScanResult

class MySTMTransport(STMTransport):
    def connect(self) -> None: ...
    def scan_image(self, size_nm, offset_nm, pixel, bias_mv) -> ScanResult: ...
    def lateral_manipulation(self, ...) -> ManipResult: ...
    # ... 12 methods total, all in physical units (nm, mV, pA)
```

See [amrl_transport/transport/simulator.py](amrl_transport/transport/simulator.py) for a complete reference implementation.

## Adapters

| Backend | Status | Platform | Notes |
|---------|--------|----------|-------|
| **Simulator** | ✅ Ready | Any | Gaussian atoms, noise, tip physics |
| **Createc** | ✅ Ready | Windows | COM interface via `pyvisa` |
| **Nanonis** | 🔧 Stub | Any | TCP socket protocol |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
# 41 tests passing — simulator backend, no hardware required
```

## Related work

- [SINGROUP/Atom_manipulation_with_RL](https://github.com/SINGROUP/Atom_manipulation_with_RL) — RL agent for atom manipulation (Aalto University)
- [abred/DeepSPM](https://github.com/abred/DeepSPM) — automated SPM with deep learning
- [Probe-Particle/ppafm](https://github.com/Probe-Particle/ppafm) — AFM image simulator
- [bluesky/ophyd-async](https://github.com/bluesky/ophyd-async) — hardware abstraction for synchrotron beamlines (similar pattern at larger scale)

## License

MIT
