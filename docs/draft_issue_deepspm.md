# Issue Title:
# [Proposal] Python replacement for LabVIEW Instrument Control Server

## Summary

I've implemented a pure Python/asyncio replacement for the LabVIEW-based 
Instrument Control Server (`TCPIP.Server v2.vi` + all NanonisCommand VIs). 

It speaks the **exact same TCP protocol** — the DeepSPM agent code 
(`agent.py`, `envClient.py`) works **without any modifications**.

## Motivation

DeepSPM is one of the most cited works on autonomous SPM, but adoption 
is limited by infrastructure requirements:

1. **LabVIEW license** (~$3500/year academic) — many labs can't justify this 
   for a single project
2. **Windows-only** — the LabVIEW VIs + COM interface locks out Linux/macOS users
3. **No simulation mode** — without physical STM hardware, it's impossible to 
   develop, test, or even understand the agent's behavior
4. **Binary VI files** — researchers can't read or modify the server logic 
   without LabVIEW IDE
5. **TensorFlow 1.x** dependency in `envClient.py` — deprecated, painful to install

These barriers mean that most researchers who want to build on DeepSPM 
end up reimplementing the server from scratch (or giving up).

## What I Built

A drop-in Python package (`amrl_transport`) with:

### 1. Instrument Control Server (replaces all 15 LabVIEW VIs)
- `InstrumentServer` — asyncio TCP server, ~150 lines of readable Python
- Parses all 6 command types: `scan()`, `tipshaping()`, `tipclean()`, 
  `getparam()`, `approach()`, `movearea()`
- Returns responses in identical binary/text format
- Reads `deepSPM_server.ini` for configuration
- Cross-platform (Linux, macOS, Windows)

### 2. Hardware Abstraction (`STMTransport` interface)
- Abstract base class for any STM controller
- Adapters: Nanonis TCP (stub), Createc COM (full), **Simulator** (full)
- Simulator generates physically plausible scan images with Gaussian atom 
  blobs, lattice corrugation, and noise

### 3. Modernized Client
- `DeepSPMClient` — clean Python client, zero TF dependency
- `EnvClientCompat` — drop-in replacement for `envClient.EnvClient`
  (just change one import line)

### 4. Tests
- 41 tests passing, including full server↔client integration tests
- All tests run on the simulator — no hardware needed, CI-friendly

## Usage

```bash
# Start Python server (no LabVIEW, no hardware)
python -m amrl_transport.deepspm --transport simulator --port 50008

# In another terminal: run the original DeepSPM agent — unchanged
cd DeepSPM/agent && ./prepareAndRun.sh 1
```

## Proposed Integration

I see two options and would love your input:

**Option A (minimal):** Add the Python server as an alternative in `labview/`, 
with documentation. Existing LabVIEW setup remains default.

**Option B (recommended):** Add as a `pyserver/` directory alongside `labview/`, 
update README to offer both options, make simulator the default for development.

## Compatibility

- BSD 3-Clause compatible (my code is MIT)
- No changes to existing agent/classifier code
- Fully backward compatible with the LabVIEW server
- Python 3.9+ only dependencies: numpy, pydantic (both standard in scientific Python)

I have a working implementation ready. Happy to submit a PR if you're 
interested, or discuss the approach first.

## References

Your README says: *"If you do write additional code or modify ours please 
contribute by sending us a pull request."* — here I am :)

Related work using similar approaches:
- SINGROUP/Atom_manipulation_with_RL (Aalto University)
- AI-SPM systems (Diao et al., Small Methods 2024)
