[![CI](https://github.com/formeo/amrl-transport/actions/workflows/ci.yml/badge.svg)](https://github.com/formeo/amrl-transport/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/amrl-transport.svg)](https://pypi.org/project/amrl-transport/)
[![codecov](https://codecov.io/gh/formeo/amrl-transport/branch/main/graph/badge.svg)](https://codecov.io/gh/formeo/amrl-transport)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19049387.svg)](https://doi.org/10.5281/zenodo.19049387)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/amrl-transport/month)](https://pepy.tech/projects/amrl-transport)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)


# AMRL Transport

**Hardware abstraction layer for autonomous atom manipulation with STM/AFM.**

Same API as [SINGROUP](https://github.com/SINGROUP/Atom_manipulation_with_RL)'s `RealExpEnv`, backed by a pluggable transport — develop RL agents against a simulator, then swap in a hardware adapter. Also includes a pure Python asyncio server that speaks the [DeepSPM](https://github.com/abred/DeepSPM) LabVIEW instrument-server protocol.

> **Status: alpha.** Everything runs end-to-end against the built-in simulator. The Createc adapter is a port of SINGROUP's controller and has **not yet been run on an instrument**; the Nanonis adapter is a stub. Hardware testers and lab collaborators are very welcome — please open an issue.

## The problem

RL frameworks for atom manipulation ([DeepSPM](https://github.com/abred/DeepSPM), [SINGROUP/AMRL](https://github.com/SINGROUP/Atom_manipulation_with_RL)) are hardwired to specific STM controllers — Createc COM on Windows, Nanonis TCP, or custom LabVIEW servers. If your lab has different hardware, you rewrite everything. If you want to develop without a $500K microscope, you can't.

## What this solves

- **`STMTransport` ABC** — 12 methods with physical units (nm, mV, pA) and one scan-frame convention (below). Implement once per vendor, use any RL agent.
- **Simulator backend** — synthetic images (Gaussian adatoms, lattice corrugation, noise) and a simple stochastic manipulation model. No hardware, no Windows, no license fees. It is a development and CI tool, not a physical model: atoms move continuously rather than hopping between lattice sites, and there is no drift or tip change.
- **DeepSPM-protocol server** — pure Python asyncio server for the DeepSPM agent. `scan()`, `tipshaping()` and `getparam()` go through the transport; `approach()`, `movearea()` and `tipclean()` are simulated only and are refused on real hardware. All instrument calls are serialized on one thread; bias pulses are limited by magnitude and keep their sign.
- **`TransportEnv`** — same `reset()`/`step()` contract as SINGROUP's `RealExpEnv`. To use SINGROUP's atom detector and jump-detection CNN, the `AMRL` package must be importable; otherwise simple fallbacks are used.
- **Task queue (experimental, frozen)** — RabbitMQ + Redis prototype. Not for use with instruments; see [below](#task-queue-experimental-frozen).

## Architecture

The queue layer at the top is an experimental prototype (see [below](#task-queue-experimental-frozen)); the transport layer, simulator and DeepSPM server are the core.

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

## Scan-frame convention

All transports use the Createc / SINGROUP frame: for a scan of size `L` at `offset_nm`, X spans `[offset_x − L/2, offset_x + L/2)` and Y spans `[offset_y, offset_y + L)`. Image columns grow with X, rows grow with Y. (The DeepSPM server converts DeepSPM's frame-centre coordinates to this convention.)

## Quick start

### Install

```bash
# Core (numpy + pydantic only)
pip install amrl-transport

# Createc adapter (Windows, COM)
pip install amrl-transport[createc]

# With ML dependencies (for SINGROUP RL agent integration)
pip install amrl-transport[all]
```

### Run with simulator (no hardware needed)

```python
from amrl_transport.transport import SimulatorTransport
import numpy as np

with SimulatorTransport(seed=42) as stm:
    # Scan a 5nm × 5nm area with the atom at (0, 0) in the centre
    result = stm.scan_image(
        size_nm=5.0,
        offset_nm=np.array([0.0, -2.5]),  # X centre, Y start of frame
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

### Use as RL environment (RealExpEnv API)

```python
import numpy as np
from amrl_transport.transport import SimulatorTransport
from amrl_transport.integration import TransportEnv

stm = SimulatorTransport(seed=42)
stm.connect()

env = TransportEnv(
    transport=stm,
    step_nm=0.2,
    max_mvolt=20,
    max_pcurrent_to_mvolt_ratio=2850,
    goal_nm=1.5,
    template=np.zeros((16, 16)),          # SINGROUP template-matching image
    current_jump=4,
    im_size_nm=5.0,
    offset_nm=np.array([0.0, -2.5]),
    manip_limit_nm=np.array([-2.0, 2.0, -2.0, 2.0]),
    pixel=64,
    template_max_y=30,
    scan_mV=100,
    max_len=10,
    load_weight="atom_move_detector.pth",  # used only if AMRL is installed
)

state, info = env.reset()
action = np.random.uniform(-1, 1, size=6)  # or your SAC agent
next_state, reward, done, info = env.step(action)
```

A complete loop is in [`examples/train_with_simulator.py`](examples/train_with_simulator.py).

### DeepSPM-protocol server

```bash
# Start the Python instrument server with simulator backend (default port 50008)
python -m amrl_transport.deepspm --transport simulator

# Point the DeepSPM agent at localhost:50008
```

With a hardware transport, `approach()`, `movearea()` and `tipclean()` are not implemented yet: the server logs an error and closes the connection instead of reporting a fake success.

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
| **Simulator** | ✅ Ready | Any | Gaussian atoms, lattice corrugation, noise |
| **Createc** | 🧪 Ported, untested on hardware | Windows | COM interface via `pywin32` |
| **Nanonis** | 🔧 Stub | Any | TCP socket protocol — not implemented |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=amrl_transport
```

Tests run on the simulator and on fake COM objects — no hardware needed.

## Task queue (experimental, frozen)

`amrl_transport.queue` is a RabbitMQ + Redis prototype kept for reference. **Do not connect it to an instrument.** Known issues:

- the default task handler is a placeholder: it reports `COMPLETED` without placing any atoms;
- tasks run inside the pika callback, so missed heartbeats or the RabbitMQ consumer timeout can redeliver a task and execute it again;
- malformed messages are requeued forever (no retry limit, no dead-letter queue);
- task `priority` is documented as "1 = highest", but RabbitMQ treats larger numbers as higher priority.

```bash
pip install amrl-transport[queue]
docker compose up -d   # RabbitMQ + Redis only
python -m amrl_transport.cli worker --transport simulator --broker amqp://amrl:amrl@localhost:5672/
python -m amrl_transport.cli submit --atoms '[[0,0],[1,0],[0.5,0.866]]' --broker amqp://amrl:amrl@localhost:5672/
```

## Related work

- [SINGROUP/Atom_manipulation_with_RL](https://github.com/SINGROUP/Atom_manipulation_with_RL) — RL agent for atom manipulation (Aalto University)
- [abred/DeepSPM](https://github.com/abred/DeepSPM) — automated SPM with deep learning
- [Probe-Particle/ppafm](https://github.com/Probe-Particle/ppafm) — AFM image simulator
- [bluesky/ophyd-async](https://github.com/bluesky/ophyd-async) — hardware abstraction for synchrotron beamlines (similar pattern at larger scale)


## Keywords

`STM` · `AFM` · `scanning probe microscopy` · `atom manipulation` ·
`reinforcement learning` · `nanofabrication` · `laboratory automation` ·
`hardware abstraction` · `DeepSPM` · `Nanonis` · `Createc` ·
`quantum computing fabrication` · `nanoassembly` · `tip-induced manipulation`

## License

MIT
