"""Audio capture and level monitoring."""
from __future__ import annotations

import asyncio
import contextlib
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

    def __init__(self, samplerate: int = 16000, blocksize: int = 2048) -> None:
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._raw_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=20)
        self._level_queue: asyncio.Queue[AudioLevel] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=40)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._processor_task: asyncio.Task[None] | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

        def callback(indata: np.ndarray, frames: int, time, status) -> None:
            if status:
                LOGGER.warning("Audio input status: %s", status)
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._enqueue_raw, indata.copy())

        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            blocksize=self._blocksize,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        self._processor_task = loop.create_task(self._process_audio())
        LOGGER.info("Audio capture started (samplerate=%s)", self._samplerate)

    async def next_level(self) -> AudioLevel:
        return await self._level_queue.get()

    async def next_chunk(self) -> bytes:
        return await self._audio_queue.get()

    def clear_audio(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _enqueue_raw(self, data: np.ndarray) -> None:
        if self._raw_queue.full():
            try:
                self._raw_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
        try:
            self._raw_queue.put_nowait(data)
        except asyncio.QueueFull:
            return
        self._enqueue_audio(data)

    def _enqueue_audio(self, data: np.ndarray) -> None:
        if self._audio_queue.full():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
        try:
            self._audio_queue.put_nowait(data.copy().tobytes())
        except asyncio.QueueFull:
            return

    async def _process_audio(self) -> None:
        try:
            while True:
                data = await self._raw_queue.get()
                float_data = data.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(np.square(float_data))))
                await self._level_queue.put(AudioLevel(rms=rms))
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            LOGGER.info("Audio capture stopped")
        if self._processor_task is not None:
            self._processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processor_task
            self._processor_task = None
