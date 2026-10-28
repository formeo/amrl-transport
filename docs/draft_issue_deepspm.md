# Issue Title:
# [Proposal] Python implementation of the instrument-server protocol + simulator

## Summary

I've written a Python/asyncio server that speaks the DeepSPM instrument-server
TCP protocol (the one implemented by `TCPIP.Server v2.vi`), together with a
simulated STM backend. With the simulator, the DeepSPM agent can be run
end-to-end without LabVIEW or an instrument.

I'd like your feedback before going further — especially on the parts I could
not verify without your setup (see "Current status").

## Motivation

- Developing or debugging the agent currently requires LabVIEW and a working
  instrument.
- A simulator makes it possible to test agent changes, run CI and onboard new
  people without lab time.
- Binary VI files are hard to read or review outside LabVIEW.

## What exists

- `InstrumentServer` — asyncio TCP server; parses `scan()`, `tipshaping()`,
  `tipclean()`, `getparam()`, `approach()`, `movearea()` and answers in the
  same text/binary format. Reads the main values from `deepSPM_server.ini`.
- `STMTransport` — small hardware-abstraction interface; simulator backend,
  a Createc (COM) adapter ported from SINGROUP, and a Nanonis stub.
- `DeepSPMClient` / `EnvClientCompat` — a TF-free Python client with the
  `envClient.EnvClient` method names.
- Tests for the protocol parser and server ↔ client round-trips on the
  simulator.

## Current status (honest version)

- **Simulator: works end-to-end.**
- **Real hardware: not yet.** `approach()`, `movearea()` and `tipclean()` are
  only simulated; with a hardware transport the server refuses them. The
  Nanonis adapter (the controller DeepSPM targets) is a stub, and the Createc
  adapter has not been run on an instrument.
- Assumption to confirm: `scan(x, y, size, px)` coordinates are the frame
  centre, as in Nanonis `Scan.FrameSet`.

## Questions for you

1. Would a simulator / Python server be useful to have alongside `labview/`
   (e.g. a `pyserver/` directory), or better kept as a separate package?
2. What exactly do `approach()`, `movearea()` and `tipclean()` do on the
   instrument (sequence of Nanonis commands, limits)? That is the main gap
   before the server can be tried on hardware.
3. Would anyone with a Nanonis setup be willing to try it in a supervised
   session once the adapter exists?

Code: https://github.com/formeo/amrl-transport (MIT).
