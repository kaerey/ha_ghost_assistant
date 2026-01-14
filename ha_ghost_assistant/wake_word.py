"""Wake word detection."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable

import numpy as np

try:
    from openwakeword.model import Model as OpenWakeWordModel
    from openwakeword.utils import download_models
except ImportError:  # pragma: no cover - optional dependency
    OpenWakeWordModel = None
    download_models = None

from ha_ghost_assistant.audio import AudioCapture

LOGGER = logging.getLogger(__name__)


class WakeWordDetector:
    """Detect wake words using openWakeWord."""

    def __init__(
        self,
        audio: AudioCapture,
        on_detected: Callable[[str], None] | None = None,
        model_name: str = "samantha",
    ) -> None:
        self._audio = audio
        self._on_detected = on_detected
        self._model_name = os.getenv(
            "HA_GHOST_ASSISTANT_WAKE_WORD_MODEL_NAME", model_name
        )
        self._wake_word_name = os.getenv(
            "HA_GHOST_ASSISTANT_WAKE_WORD_NAME", "Samantha"
        )
        self._model_path = os.getenv("HA_GHOST_ASSISTANT_WAKE_WORD_MODEL")
        self._threshold = float(
            os.getenv("HA_GHOST_ASSISTANT_WAKE_WORD_THRESHOLD", "0.6")
        )
        self._cooldown_seconds = float(
            os.getenv("HA_GHOST_ASSISTANT_WAKE_WORD_COOLDOWN", "2.0")
        )
        self._enabled = os.getenv("HA_GHOST_ASSISTANT_WAKE_WORD", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        self._model: OpenWakeWordModel | None = None
        self._task: asyncio.Task[None] | None = None
        self._audio_queue: asyncio.Queue[bytes] | None = None
        self._last_detection = 0.0

    async def start(self) -> None:
        if not self._enabled:
            LOGGER.info("Wake word detector disabled")
            return
        if OpenWakeWordModel is None:
            LOGGER.warning(
                "openWakeWord not installed; wake word detection disabled"
            )
            return
        self._audio_queue = self._audio.create_audio_queue()
        self._model = self._build_model()
        if self._model is None:
            return
        self._task = asyncio.create_task(self._run())
        LOGGER.info(
            "Wake word detector started (name=%s, model=%s)",
            self._wake_word_name,
            self._model_name,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._audio_queue is not None:
            self._audio.remove_audio_queue(self._audio_queue)
            self._audio_queue = None
        LOGGER.info("Wake word detector stopped")

    def is_active(self) -> bool:
        return self._enabled and self._model is not None

    def is_configured(self) -> bool:
        if not self._enabled or OpenWakeWordModel is None:
            return False
        if self._model_path:
            return True
        return self._model_name.lower() != "samantha"

    def wake_word_name(self) -> str:
        return self._wake_word_name

    def wake_word_model(self) -> str:
        return self._model_path or self._model_name

    def notify_detected(self, name: str) -> None:
        if self._on_detected is None:
            return
        self._on_detected(name)

    def _build_model(self) -> OpenWakeWordModel | None:
        model_list: list[str] = []
        if self._model_path:
            model_list = [self._model_path]
        else:
            if download_models is None:
                LOGGER.warning(
                    "No wake word model configured; set HA_GHOST_ASSISTANT_WAKE_WORD_MODEL"
                )
                return None
            if self._model_name.lower() == "samantha":
                LOGGER.warning(
                    "Custom wake word model required for '%s'; set HA_GHOST_ASSISTANT_WAKE_WORD_MODEL",
                    self._model_name,
                )
                return None
            LOGGER.info("Downloading openWakeWord base models for %s", self._model_name)
            download_models([self._model_name])
            model_list = [self._model_name]
        try:
            return OpenWakeWordModel(wakeword_models=model_list)
        except Exception:
            LOGGER.exception("Failed to load openWakeWord model")
            return None

    async def _run(self) -> None:
        if self._audio_queue is None or self._model is None:
            return
        while True:
            chunk = await self._audio_queue.get()
            scores = self._model.predict(np.frombuffer(chunk, dtype=np.int16))
            if not scores:
                continue
            best_name, best_score = max(scores.items(), key=lambda item: item[1])
            if best_score < self._threshold:
                continue
            now = time.monotonic()
            if now - self._last_detection < self._cooldown_seconds:
                continue
            self._last_detection = now
            LOGGER.info(
                "Wake word detected (%s=%.3f); triggering %s",
                best_name,
                best_score,
                self._wake_word_name,
            )
            self.notify_detected(self._wake_word_name)
