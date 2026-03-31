"""
python -m amrl_transport.deepspm --transport simulator --port 50008
"""
import argparse
import asyncio
import logging
import sys


def main():
    p = argparse.ArgumentParser(description="DeepSPM Server (Python)")
    p.add_argument("--transport", choices=["createc","nanonis","simulator"], default="simulator")
    p.add_argument("--port", type=int, default=50008)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--config", default=None, help="deepSPM_server.ini path")
    p.add_argument("--seed", type=int, default=None)
    opts = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(levelname)s|%(message)s",
        stream=sys.stdout,
    )

    from ..config.models import SimulatorConfig, TransportConfig, TransportType
    from ..transport.factory import create_transport
    from .server import InstrumentServer, ServerConfig

    tcfg = TransportConfig(
        type=TransportType(opts.transport),
        simulator=SimulatorConfig(
            seed=opts.seed, initial_atoms=[[0, 0], [5, 5]]
        ),
    )
    transport = create_transport(tcfg)
    transport.connect()

    scfg = ServerConfig.from_ini(opts.config) if opts.config else ServerConfig()
    scfg.host = opts.host
    scfg.port = opts.port

    print(f"DeepSPM Server | {transport.name} | {opts.host}:{opts.port}")
    try:
        asyncio.run(InstrumentServer(transport, scfg).serve_forever())
    except KeyboardInterrupt:
        pass
    finally:
        transport.disconnect()

if __name__ == "__main__":
    main()
