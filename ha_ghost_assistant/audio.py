"""Audio capture and level monitoring."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

LOGGER = logging.getLogger(__name__)


@dataclass
class AudioLevel:
    rms: float


class AudioCapture:
    """Capture audio from the default microphone and provide level updates."""

    def __init__(self, samplerate: int = 16000, blocksize: int = 1024) -> None:
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._queue: asyncio.Queue[AudioLevel] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

        def callback(indata: np.ndarray, frames: int, time, status) -> None:
            if status:
                LOGGER.warning("Audio input status: %s", status)
            rms = float(np.sqrt(np.mean(np.square(indata))))
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait, AudioLevel(rms=rms)
                )

        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            blocksize=self._blocksize,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        LOGGER.info("Audio capture started (samplerate=%s)", self._samplerate)

    async def next_level(self) -> AudioLevel:
        return await self._queue.get()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            LOGGER.info("Audio capture stopped")
