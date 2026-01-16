"""Audio playback helper."""
from __future__ import annotations

import logging
import threading

import sounddevice as sd

LOGGER = logging.getLogger(__name__)


class AudioPlayback:
    """Play PCM audio through the default speaker."""

    def __init__(self) -> None:
        # Use RawOutputStream + callback so the device pulls audio at a steady pace.
        # write() simply appends bytes into a buffer (fast, non-blocking).
        self._stream: sd.RawOutputStream | None = None
        self._rate: int | None = None
        self._channels: int | None = None
        self._width_bytes: int = 2  # int16 => 2 bytes

        self._lock = threading.Lock()
        self._buf = bytearray()

        # Safety buffer: keep up to ~2 seconds of audio before dropping oldest.
        self._max_buffer_seconds = 2.0

    async def start(self, rate: int, channels: int) -> None:
        if (
            self._stream is not None
            and self._rate == rate
            and self._channels == channels
        ):
            return
        await self.stop()
        self._rate = rate
        self._channels = channels

        # Clear any old buffered data
        with self._lock:
            self._buf.clear()

        def callback(outdata, frames, time, status):  # pylint: disable=unused-argument
            if status:
                # Don't spam; sounddevice can report under/overruns transiently
                LOGGER.debug("Audio output status: %s", status)

            # outdata is a writable buffer of bytes for RawOutputStream
            nbytes = frames * (channels * self._width_bytes)
            with self._lock:
                available = len(self._buf)
                if available >= nbytes:
                    outdata[:] = self._buf[:nbytes]
                    del self._buf[:nbytes]
                    return
                # Not enough data: output what we have + pad silence
                if available > 0:
                    outdata[:available] = self._buf
                    del self._buf[:]
                # Pad the rest with zeros (silence)
                outdata[available:nbytes] = b"\x00" * (nbytes - available)

        self._stream = sd.RawOutputStream(
            samplerate=rate,
            channels=channels,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        LOGGER.info("Audio playback started (rate=%s, channels=%s)", rate, channels)

    def write(self, pcm_bytes: bytes) -> None:
        """Append PCM bytes to the playback buffer (non-blocking)."""
        stream = self._stream
        if stream is None or self._rate is None or self._channels is None:
            return

        # Safety valve: cap buffer size to avoid unbounded growth if something goes wrong.
        max_bytes = int(self._max_buffer_seconds * self._rate * self._channels * self._width_bytes)
        with self._lock:
            if len(self._buf) + len(pcm_bytes) > max_bytes:
                # Drop oldest audio to make room (rare). This is better than exploding RAM.
                overflow = (len(self._buf) + len(pcm_bytes)) - max_bytes
                if overflow > 0:
                    del self._buf[:overflow]
                    LOGGER.warning("Audio buffer overflow: dropped %d bytes", overflow)
            self._buf.extend(pcm_bytes)

    async def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            with self._lock:
                self._buf.clear()
        LOGGER.info("Audio playback stopped")
