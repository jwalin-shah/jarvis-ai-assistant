from __future__ import annotations

import asyncio
import signal

from jarvis.interfaces.desktop.server import JarvisSocketServer


async def _serve() -> None:
    server = JarvisSocketServer()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Fallback for platforms without add_signal_handler support.
            signal.signal(sig, lambda *_: _request_stop())

    await server.start()
    try:
        await stop_event.wait()
    finally:
        await server.stop()


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
