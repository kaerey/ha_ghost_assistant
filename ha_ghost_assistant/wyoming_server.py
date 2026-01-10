"""Wyoming Protocol TCP server for Home Assistant integration."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable

from ha_ghost_assistant.audio import AudioCapture

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
    """Wyoming Protocol server supporting describe, satellite control, and audio."""

    def __init__(
        self,
        host: str,
        port: int,
        audio: AudioCapture,
        info: WyomingInfo | None = None,
        on_state: StateCallback | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._audio = audio
        self._info = info or WyomingInfo()
        self._on_state = on_state
        self._server: asyncio.AbstractServer | None = None
        self._state = "idle"
        self._writer: asyncio.StreamWriter | None = None
        self._writer_lock = asyncio.Lock()
        self._stream_task: asyncio.Task[None] | None = None
        self._stop_stream = asyncio.Event()

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
        await self._stop_streaming()
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        LOGGER.info("Wyoming server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        LOGGER.info("Wyoming client connected: %s", peer)
        if self._writer is not None and self._writer is not writer:
            LOGGER.info("Replacing existing Wyoming client connection")
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                LOGGER.exception("Failed to close previous Wyoming client")
        self._writer = writer
        try:
            while True:
                event = await self._read_event(reader)
                if event is None:
                    break
                LOGGER.info("Wyoming event received: %s", event.get("type"))
                await self._handle_event(event, writer)
        except asyncio.IncompleteReadError:
            LOGGER.info("Wyoming client disconnected: %s", peer)
        finally:
            await self._stop_streaming()
            writer.close()
            await writer.wait_closed()
            if self._writer is writer:
                self._writer = None

    async def _handle_event(
        self, event: dict[str, object], writer: asyncio.StreamWriter
    ) -> None:
        event_type = event.get("type")
        if event_type == "describe":
            await self._send_event(writer, {"type": "info", "data": self._info.as_dict()})
            return
        if event_type == "run-satellite":
            await self._start_streaming()
            return
        if event_type == "pause-satellite":
            await self._stop_streaming()
            return
        LOGGER.warning("Unhandled Wyoming event: %s", event_type)

    async def trigger(
        self, name: str = "push_to_talk", start_stream: bool = True
    ) -> None:
        if self._writer is None:
            LOGGER.warning("No Wyoming client connected; trigger ignored")
            return
        data = {"name": name, "model": "ha_ghost_assistant"}
        await self._send_event(self._writer, {"type": "wake-word-detected", "data": data})
        if start_stream:
            await self._start_streaming()

    async def _start_streaming(self) -> None:
        if self._writer is None:
            LOGGER.warning("No Wyoming client connected; cannot start streaming")
            return
        if self._stream_task is not None and not self._stream_task.done():
            return
        self._audio.clear_audio()
        self._stop_stream.clear()
        self._stream_task = asyncio.create_task(self._stream_audio(self._writer))
        self._set_state("listening")

    async def _stop_streaming(self) -> None:
        if self._stream_task is None:
            return
        self._stop_stream.set()
        self._stream_task.cancel()
        try:
            await self._stream_task
        except asyncio.CancelledError:
            pass
        self._stream_task = None
        if self._writer is not None and not self._writer.is_closing():
            await self._send_event(self._writer, {"type": "audio-stop"})
        self._set_state("idle")

    async def _stream_audio(self, writer: asyncio.StreamWriter) -> None:
        await self._send_event(
            writer,
            {
                "type": "audio-start",
                "data": {
                    "rate": self._info.mic_rate,
                    "width": self._info.mic_width,
                    "channels": self._info.mic_channels,
                },
            },
        )
        try:
            while not self._stop_stream.is_set():
                chunk = await self._audio.next_chunk()
                await self._send_event(writer, {"type": "audio-chunk"}, payload=chunk)
        except asyncio.CancelledError:
            return

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        LOGGER.info("State transition: %s", state)
        if self._on_state is not None:
            self._on_state(state)

    async def _send_event(
        self,
        writer: asyncio.StreamWriter,
        event: dict[str, object],
        payload: bytes | None = None,
    ) -> None:
        if payload is not None:
            event = dict(event)
            event["data_length"] = len(payload)
        message = json.dumps(event).encode("utf-8") + b"\n"
        async with self._writer_lock:
            writer.write(message)
            if payload is not None:
                writer.write(payload)
            await writer.drain()

    async def _read_event(
        self, reader: asyncio.StreamReader
    ) -> dict[str, object] | None:
        line = await reader.readline()
        if not line:
            return None
        try:
            event = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Failed to decode Wyoming event")
            return None
        data_length = event.get("data_length")
        if isinstance(data_length, int) and data_length > 0:
            try:
                await reader.readexactly(data_length)
            except asyncio.IncompleteReadError:
                LOGGER.warning("Incomplete Wyoming payload read")
        return event
