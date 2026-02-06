#!/usr/bin/env python3
"""Minimal MLX Embedding Server - standalone JSON-RPC wrapper.

This is a standalone debugging tool that wraps the in-process BERT embedder
(models.bert_embedder) in a JSON-RPC Unix socket server. Useful for testing
embeddings from external tools or the command line.

For production use, embeddings run in-process via:
    from jarvis.embedding_adapter import get_embedder

Usage:
    uv run python scripts/minimal_mlx_embed_server.py

Protocol: JSON-RPC 2.0 over Unix socket
Socket: /tmp/jarvis-embed-minimal.sock
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

import numpy as np

from models.bert_embedder import MODEL_REGISTRY, InProcessEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("minimal-mlx-embed")

SOCKET_PATH = Path(os.environ.get("MLX_EMBED_SOCKET", "/tmp/jarvis-embed-minimal.sock"))


# =============================================================================
# JSON-RPC Server
# =============================================================================


class MinimalEmbedServer:
    """Async Unix socket server wrapping InProcessEmbedder."""

    def __init__(self, embedder: InProcessEmbedder):
        self._embedder = embedder
        self._server = None
        self._running = False
        self._start_time = time.time()

    async def start(self) -> None:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        self._server = await asyncio.start_unix_server(self._handle_client, path=str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o600)
        self._running = True

        logger.info("Minimal MLX Embed Server listening on %s", SOCKET_PATH)

        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        self._embedder.unload()
        logger.info("Server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while self._running:
                line = await asyncio.wait_for(reader.readline(), timeout=300)
                if not line:
                    break
                response = await self._process(line.decode())
                writer.write(response.encode() + b"\n")
                await writer.drain()
        except TimeoutError:
            pass
        except Exception as e:
            logger.debug("Client error: %s", e)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process(self, message: str) -> str:
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return json.dumps(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}, "id": None}
            )

        method = data.get("method")
        params = data.get("params", {})
        req_id = data.get("id")

        handlers = {
            "health": self._health,
            "embed": self._embed,
            "unload": self._unload,
            "list_models": self._list_models,
            "ping": lambda p: {"status": "ok"},
        }

        handler = handlers.get(method)
        if not handler:
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                    "id": req_id,
                }
            )

        try:
            result = await asyncio.to_thread(handler, params)
            return json.dumps({"jsonrpc": "2.0", "result": result, "id": req_id})
        except Exception as e:
            return json.dumps(
                {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": req_id}
            )

    def _health(self, params: dict) -> dict:
        return {
            "status": "healthy",
            "model_loaded": self._embedder.is_loaded,
            "model_name": self._embedder.model_name,
            "uptime_seconds": time.time() - self._start_time,
            "backend": "minimal-mlx",
        }

    def _embed(self, params: dict) -> dict:
        texts = params.get("texts", [])
        model = params.get("model")
        normalize = params.get("normalize", True)
        binary = params.get("binary", False)

        if not texts:
            raise ValueError("No texts provided")

        if model:
            self._embedder.load_model(model)

        embeddings = self._embedder.encode(texts, normalize)

        if binary:
            import base64

            emb_bytes = embeddings.astype(np.float32).tobytes()
            return {
                "embeddings_b64": base64.b64encode(emb_bytes).decode("ascii"),
                "model": self._embedder.model_name,
                "dimension": embeddings.shape[1],
                "count": len(embeddings),
                "dtype": "float32",
            }
        else:
            return {
                "embeddings": embeddings.tolist(),
                "model": self._embedder.model_name,
                "dimension": embeddings.shape[1],
                "count": len(embeddings),
            }

    def _unload(self, params: dict) -> dict:
        self._embedder.unload()
        return {"status": "unloaded"}

    def _list_models(self, params: dict) -> dict:
        return {
            "models": [
                {"name": name, "pooling": pooling} for name, (_, pooling) in MODEL_REGISTRY.items()
            ],
            "current": self._embedder.model_name,
        }


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    embedder = InProcessEmbedder()
    server = MinimalEmbedServer(embedder)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

    try:
        await server.start()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
