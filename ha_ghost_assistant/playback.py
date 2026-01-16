"""Audio playback helper."""
from __future__ import annotations

import logging
import threading
from collections import deque

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
        self._buf: deque[bytes] = deque()
        self._buf_size = 0
        self._head_offset = 0

        # Safety buffer: keep up to ~6 seconds of audio before dropping oldest.
        self._max_buffer_seconds = 6.0

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
            self._buf_size = 0
            self._head_offset = 0

        def callback(outdata, frames, time, status):  # pylint: disable=unused-argument
            if status:
                # Don't spam; sounddevice can report under/overruns transiently
                LOGGER.debug("Audio output status: %s", status)

            # outdata is a writable buffer of bytes for RawOutputStream
            nbytes = frames * (channels * self._width_bytes)
            with self._lock:
                available = self._buf_size
                if available <= 0:
                    outdata[:] = b"\x00" * nbytes
                    return

                write_pos = 0
                to_copy = min(nbytes, available)
                while to_copy > 0 and self._buf:
                    chunk = self._buf[0]
                    start = self._head_offset
                    chunk_avail = len(chunk) - start
                    take = min(to_copy, chunk_avail)
                    outdata[write_pos:write_pos + take] = chunk[start:start + take]
                    write_pos += take
                    to_copy -= take
                    self._buf_size -= take
                    if take == chunk_avail:
                        self._buf.popleft()
                        self._head_offset = 0
                    else:
                        self._head_offset += take
                        break

                # Pad the rest with zeros (silence)
                if write_pos < nbytes:
                    outdata[write_pos:nbytes] = b"\x00" * (nbytes - write_pos)

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
            incoming = len(pcm_bytes)
            if self._buf_size + incoming > max_bytes:
                # Drop oldest audio to make room (rare). This is better than exploding RAM.
                overflow = (self._buf_size + incoming) - max_bytes
                if overflow > 0:
                    self._drop_oldest(overflow)
                    LOGGER.warning("Audio buffer overflow: dropped %d bytes", overflow)
            self._buf.append(pcm_bytes)
            self._buf_size += incoming

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
                self._buf_size = 0
                self._head_offset = 0
        LOGGER.info("Audio playback stopped")

    def _drop_oldest(self, nbytes: int) -> None:
        """Drop the oldest bytes from the buffer."""
        remaining = nbytes
        while remaining > 0 and self._buf:
            chunk = self._buf[0]
            start = self._head_offset
            chunk_avail = len(chunk) - start
            if remaining < chunk_avail:
                self._head_offset += remaining
                self._buf_size -= remaining
                return
            remaining -= chunk_avail
            self._buf_size -= chunk_avail
            self._buf.popleft()
            self._head_offset = 0
