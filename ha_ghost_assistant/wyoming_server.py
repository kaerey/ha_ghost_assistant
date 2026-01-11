"""Wyoming Protocol TCP server for Home Assistant integration."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ha_ghost_assistant.audio import AudioCapture
from ha_ghost_assistant.playback import AudioPlayback

LOGGER = logging.getLogger(__name__)
PING_SEND_DELAY = 2.0
PONG_TIMEOUT = 5.0
SILENCE_THRESHOLD = float(os.getenv("HA_GHOST_ASSISTANT_SILENCE_THRESHOLD", "0.005"))
SILENCE_TIMEOUT = float(os.getenv("HA_GHOST_ASSISTANT_SILENCE_TIMEOUT", "3.0"))


@dataclass
class WyomingInfo:
    name: str = "HA Ghost Assistant"
    description: str = "Home Assistant Wyoming satellite"
    version: str = "0.0.1"
    attribution: str = "OpenAI"
    satellite_id: str = field(default_factory=lambda: socket.gethostname())
    software: str = "ha_ghost_assistant"
    mic_rate: int = 16000
    mic_width: int = 2
    mic_channels: int = 1
    snd_rate: int = 16000
    snd_width: int = 2
    snd_channels: int = 1
    supports_trigger: bool = True
    has_vad: bool = False
    wake_name: str = "push_to_talk"
    wake_model: str = "ha_ghost_assistant"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "id": self.satellite_id,
            "software": self.software,
            "satellite": {
                # Required by Home Assistant wyoming parser
                "name": self.name,
                "installed": True,
                "attribution": {
                    "name": self.software,
                    "url": "https://github.com/kaerey/ha_ghost_assistant",
                },
                # Satellite capabilities
                "supports_trigger": self.supports_trigger,
                "has_vad": self.has_vad,
                "active_wake_words": [],
                "max_active_wake_words": 0,
                "area": "",
            },
            "mic": [
                {
                    "name": f"{self.software}_mic",
                    "installed": True,
                    "attribution": {
                        "name": self.software,
                        "url": "https://github.com/kaerey/ha_ghost_assistant",
                    },
                    "mic_format": {
                        "rate": self.mic_rate,
                        "width": self.mic_width,
                        "channels": self.mic_channels,
                    }
                }
            ],
            "snd": [
                {
                    "name": f"{self.software}_snd",
                    "installed": True,
                    "attribution": {
                        "name": self.software,
                        "url": "https://github.com/kaerey/ha_ghost_assistant",
                    },
                    "snd_format": {
                        "rate": 22050,
                        "width": self.snd_width,
                        "channels": self.snd_channels,
                    }
                }
            ],
        }


StateCallback = Callable[[str], None]


class WyomingServer:
    """Wyoming Protocol server supporting describe, satellite control, and audio."""

    def __init__(
        self,
        host: str,
        port: int,
        audio: AudioCapture,
        playback: AudioPlayback,
        info: WyomingInfo | None = None,
        on_state: StateCallback | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._audio = audio
        self._playback = playback
        self._info = info or WyomingInfo()
        self._on_state = on_state
        self._server: asyncio.AbstractServer | None = None
        self._state = "idle"
        self._server_id: str | None = None
        self._server_writer: asyncio.StreamWriter | None = None
        self._writer_lock = asyncio.Lock()
        self._stream_task: asyncio.Task[None] | None = None
        self._stop_stream = asyncio.Event()
        self._pending_trigger: str | None = None
        self._client_connected = asyncio.Event()
        self._ping_enabled = False
        self._pong_event = asyncio.Event()
        self._ping_task: asyncio.Task[None] | None = None

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
        client_id = str(time.monotonic_ns())
        if self._pending_trigger is not None:
            pending = self._pending_trigger
            self._pending_trigger = None
            await self.trigger(name=pending)
        try:
            while not reader.at_eof():
                event = await self._read_event(reader)
                if event is None:
                    break
                event_type = event.get("type")
                if event_type != "ping":
                    LOGGER.info("Wyoming event received: %s", event_type)
                if event_type != "describe":
                    if self._server_id != client_id:
                        self._server_id = client_id
                        self._server_writer = writer
                        self._client_connected.set()
                await self._handle_event(event, writer)
        except asyncio.IncompleteReadError:
            LOGGER.info("Wyoming client disconnected: %s", peer)
        except asyncio.CancelledError:
            LOGGER.exception("Wyoming client handler cancelled: %s", peer)
            raise
        except Exception:
            LOGGER.exception("Unhandled Wyoming client error: %s", peer)
        finally:
            LOGGER.info("Wyoming client disconnected: %s", peer)
            await self._stop_streaming()
            await self._playback.stop()
            writer.close()
            await writer.wait_closed()
            if self._server_id == client_id:
                self._server_id = None
                self._server_writer = None
                self._client_connected.clear()
                self._disable_ping()

    async def _handle_event(
        self, event: dict[str, object], writer: asyncio.StreamWriter
    ) -> None:
        event_type = event.get("type")
        if event_type == "describe":
            LOGGER.info("Wyoming describe request: %s", event)
            info_event = {"type": "info", "data": self._info.as_dict()}
            LOGGER.debug("Wyoming info data: %s", info_event["data"])
            LOGGER.info("Wyoming info response: %s", json.dumps(info_event))
            await self._send_event(writer, info_event)
            return
        if event_type == "ping":
            await self._send_event(writer, {"type": "pong"})
            return
        if event_type == "pong":
            self._pong_event.set()
            return
        if event_type == "audio-start":
            data = event.get("data")
            rate = None
            channels = None
            if isinstance(data, dict):
                rate = data.get("rate")
                channels = data.get("channels")
            await self._playback.start(
                int(rate) if rate is not None else 22050,
                int(channels) if channels is not None else self._info.snd_channels,
            )
            return
        if event_type == "audio-chunk":
            payload = event.get("_payload")
            if isinstance(payload, (bytes, bytearray)):
                self._playback.write(payload)
            return
        if event_type == "audio-stop":
            await self._playback.stop()
            return
        if event_type == "run-satellite":
            LOGGER.info("Wyoming run-satellite received")
            await self._start_streaming()
            return
        if event_type == "pause-satellite":
            LOGGER.info("Wyoming pause-satellite received")
            await self._stop_streaming()
            return
        LOGGER.warning("Unhandled Wyoming event: %s", event_type)

    async def trigger(
        self, name: str = "push_to_talk", start_stream: bool = True
    ) -> None:
        if self._server_writer is None:
            self._pending_trigger = name
            LOGGER.info("Queued trigger '%s' until a Wyoming client connects", name)
            return
        run_pipeline = {
            "type": "run-pipeline",
            "data": {
                "start_stage": "asr",
                "end_stage": "tts",
                "restart_on_end": False,
                "snd_format": {
                    "rate": 22050,
                    "width": self._info.snd_width,
                    "channels": self._info.snd_channels,
                },
                "wake_word_name": name,
            },
        }
        await self._send_event(self._server_writer, run_pipeline)
        if start_stream:
            LOGGER.debug("Waiting for run-satellite before starting audio streaming")

    async def stop_streaming(self) -> None:
        await self._stop_streaming()

    async def wait_for_client(self, timeout: float | None = None) -> bool:
        try:
            if timeout is None:
                await self._client_connected.wait()
                return True
            await asyncio.wait_for(self._client_connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _enable_ping(self) -> None:
        self._ping_enabled = True
        if self._ping_task is None:
            self._ping_task = asyncio.create_task(self._ping_server())

    def _disable_ping(self) -> None:
        self._ping_enabled = False
        if self._ping_task is not None:
            self._ping_task.cancel()
            self._ping_task = None

    async def _ping_server(self) -> None:
        try:
            while True:
                await asyncio.sleep(PING_SEND_DELAY)
                if not self._ping_enabled or self._server_writer is None:
                    continue
                self._pong_event.clear()
                await self._send_event(
                    self._server_writer,
                    {"type": "ping", "data": {"text": str(time.monotonic())}},
                )
                try:
                    await asyncio.wait_for(self._pong_event.wait(), timeout=PONG_TIMEOUT)
                except asyncio.TimeoutError:
                    if self._server_writer is None:
                        continue
                    LOGGER.warning("Wyoming ping timeout; clearing server connection")
                    self._server_id = None
                    self._server_writer = None
                    self._client_connected.clear()
                    self._disable_ping()
        except asyncio.CancelledError:
            LOGGER.exception("Wyoming ping task cancelled")
            raise

    async def _start_streaming(self) -> None:
        if self._server_writer is None:
            LOGGER.warning("No Wyoming client connected; cannot start streaming")
            return
        if self._stream_task is not None and not self._stream_task.done():
            return
        self._audio.clear_audio()
        self._stop_stream.clear()
        self._stream_task = asyncio.create_task(self._stream_audio(self._server_writer))
        self._set_state("listening")

    async def _stop_streaming(self) -> None:
        if self._stream_task is None:
            return
        if self._stream_task.done():
            self._stream_task = None
            return
        self._stop_stream.set()
        self._stream_task.cancel()
        try:
            await self._stream_task
        except asyncio.CancelledError:
            pass
        self._stream_task = None
        if self._server_writer is not None and not self._server_writer.is_closing():
            await self._send_event(self._server_writer, {"type": "audio-stop"})
        self._set_state("idle")

    async def _stream_audio(self, writer: asyncio.StreamWriter) -> None:
        self._stop_stream.clear()
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
        LOGGER.debug("Wyoming audio streaming started")
        first_chunk = True
        silence_duration = 0.0
        try:
            while not self._stop_stream.is_set():
                chunk = await self._audio.next_chunk()
                await self._send_event(writer, {"type": "audio-chunk"}, payload=chunk)
                if SILENCE_TIMEOUT > 0:
                    audio = (
                        np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    rms = float(np.sqrt(np.mean(np.square(audio))))
                    chunk_duration = (
                        len(chunk) / (self._info.mic_width * self._info.mic_channels)
                    ) / self._info.mic_rate
                    if rms < SILENCE_THRESHOLD:
                        silence_duration += chunk_duration
                        if silence_duration >= SILENCE_TIMEOUT:
                            LOGGER.info("Silence detected; stopping audio stream")
                            self._stop_stream.set()
                            break
                    else:
                        silence_duration = 0.0
                if first_chunk:
                    LOGGER.debug("Wyoming audio-chunk sent")
                    first_chunk = False
        except asyncio.CancelledError:
            LOGGER.exception("Wyoming audio stream task cancelled")
            raise
        if writer is not None and not writer.is_closing():
            await self._send_event(writer, {"type": "audio-stop"})
        self._set_state("idle")
        self._stream_task = None

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
            event["payload_length"] = len(payload)
        if event.get("type") != "pong":
            LOGGER.info("Wyoming event sent: %s", event.get("type"))
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
        LOGGER.debug("Wyoming raw event line: %s", line)
        if b'"describe"' in line:
            LOGGER.info("Wyoming raw describe request: %s", line)
        try:
            event = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Failed to decode Wyoming event")
            return None
        data_length = event.get("data_length")
        payload_length = event.get("payload_length")
        if isinstance(data_length, int) and data_length > 0:
            try:
                await reader.readexactly(data_length)
            except asyncio.IncompleteReadError:
                LOGGER.warning("Incomplete Wyoming data payload read")
        if isinstance(payload_length, int) and payload_length > 0:
            try:
                event["_payload"] = await reader.readexactly(payload_length)
            except asyncio.IncompleteReadError:
                LOGGER.warning("Incomplete Wyoming payload read")
        return event
