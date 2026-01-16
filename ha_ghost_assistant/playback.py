"""Audio playback helper."""
from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

LOGGER = logging.getLogger(__name__)


class AudioPlayback:
    """Play PCM audio through the default speaker."""

    def __init__(self) -> None:
        self._stream: sd.OutputStream | None = None
        self._rate: int | None = None
        self._channels: int | None = None

        # NOTE: sounddevice stream.write(...) is blocking. If we call it on the
        # asyncio event loop thread, visuals will stutter badly during TTS.
        # We push PCM chunks through a small queue and write them from a single
        # background thread instead.
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

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
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker, name="AudioPlayback", daemon=True)
        self._thread.start()
        LOGGER.info("Audio playback started (rate=%s, channels=%s)", rate, channels)

    def write(self, pcm_bytes: bytes) -> None:
        # Non-blocking: enqueue audio for the worker thread.
        if self._stream is None:
            return
        try:
            self._q.put_nowait(pcm_bytes)
        except queue.Full:
            # Prefer realtime responsiveness over perfect audio: drop oldest.
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(pcm_bytes)
            except queue.Full:
                pass

    def _worker(self) -> None:
        """Write PCM chunks to the audio device from a background thread."""
        while not self._stop_evt.is_set():
            try:
                pcm = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            stream = self._stream
            if stream is None:
                continue
            try:
                data = np.frombuffer(pcm, dtype=np.int16)
                if self._channels and self._channels > 1:
                    data = data.reshape(-1, self._channels)
                stream.write(data)
            except Exception:
                LOGGER.exception("Audio playback write failed")
                time.sleep(0.01)

    async def stop(self) -> None:
        if self._stream is None:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Clear any pending audio
        try:
            while True:
                _ = self._q.get_nowait()
        except queue.Empty:
            pass
        self._stream.stop()
        self._stream.close()
        self._stream = None
        LOGGER.info("Audio playback stopped")
