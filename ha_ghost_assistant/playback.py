"""Audio playback helper."""
from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

LOGGER = logging.getLogger(__name__)


class AudioPlayback:
    """Play PCM audio through the default speaker."""

    def __init__(self) -> None:
        self._stream: sd.OutputStream | None = None
        self._rate: int | None = None
        self._channels: int | None = None

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
        self._stream = sd.OutputStream(
            samplerate=rate,
            channels=channels,
            dtype="int16",
        )
        self._stream.start()
        LOGGER.info("Audio playback started (rate=%s, channels=%s)", rate, channels)

    def write(self, pcm_bytes: bytes) -> None:
        if self._stream is None:
            return
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        if self._channels and self._channels > 1:
            data = data.reshape(-1, self._channels)
        self._stream.write(data)

    async def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        LOGGER.info("Audio playback stopped")
