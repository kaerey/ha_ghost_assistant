"""Minimal Wyoming Protocol TCP server for Home Assistant integration."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)


@dataclass
class WyomingInfo:
    mic_rate: int = 16000
    mic_width: int = 2
    mic_channels: int = 1
    snd_rate: int = 16000
    snd_width: int = 2
    snd_channels: int = 1
    supports_trigger: bool = True
    has_vad: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "satellite": {
                "supports_trigger": self.supports_trigger,
                "has_vad": self.has_vad,
            },
            "mic": {
                "mic_format": {
                    "rate": self.mic_rate,
                    "width": self.mic_width,
                    "channels": self.mic_channels,
                }
            },
            "snd": {
                "snd_format": {
                    "rate": self.snd_rate,
                    "width": self.snd_width,
                    "channels": self.snd_channels,
                }
            },
        }


StateCallback = Callable[[str], None]


class WyomingServer:
    """Minimal Wyoming Protocol server supporting describe and satellite control."""

    def __init__(
        self,
        host: str,
        port: int,
        info: WyomingInfo | None = None,
        on_state: StateCallback | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._info = info or WyomingInfo()
        self._on_state = on_state
        self._server: asyncio.AbstractServer | None = None
        self._state = "idle"

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, host=self._host, port=self._port
        )
        sockets = self._server.sockets or []
        listen = ", ".join(
            f"{sock.getsockname()[0]}:{sock.getsockname()[1]}" for sock in sockets
        )
        LOGGER.info("Wyoming server listening on %s", listen)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        LOGGER.info("Wyoming server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        LOGGER.info("Wyoming client connected: %s", peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                event = self._decode_event(line)
                if event is None:
                    continue
                LOGGER.info("Wyoming event received: %s", event.get("type"))
                await self._handle_event(event, writer)
        except asyncio.IncompleteReadError:
            LOGGER.info("Wyoming client disconnected: %s", peer)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_event(
        self, event: dict[str, object], writer: asyncio.StreamWriter
    ) -> None:
        event_type = event.get("type")
        if event_type == "describe":
            await self._send_event(writer, {"type": "info", "data": self._info.as_dict()})
            return
        if event_type == "run-satellite":
            self._set_state("listening")
            return
        if event_type == "pause-satellite":
            self._set_state("idle")
            return
        LOGGER.warning("Unhandled Wyoming event: %s", event_type)

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        LOGGER.info("State transition: %s", state)
        if self._on_state is not None:
            self._on_state(state)

    async def _send_event(self, writer: asyncio.StreamWriter, event: dict[str, object]) -> None:
        payload = json.dumps(event).encode("utf-8") + b"\n"
        writer.write(payload)
        await writer.drain()

    def _decode_event(self, line: bytes) -> dict[str, object] | None:
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Failed to decode Wyoming event")
            return None
