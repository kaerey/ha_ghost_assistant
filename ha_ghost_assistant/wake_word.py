"""Wake word detection (stub)."""
from __future__ import annotations

import logging
from typing import Callable

LOGGER = logging.getLogger(__name__)


class WakeWordDetector:
    """Placeholder for openWakeWord integration."""

    def __init__(
        self,
        on_detected: Callable[[str], None] | None = None,
        model_name: str = "ha_ghost_assistant",
    ) -> None:
        self._on_detected = on_detected
        self._model_name = model_name

    async def start(self) -> None:
        LOGGER.info("Wake word detector stub started (model=%s)", self._model_name)

    async def stop(self) -> None:
        LOGGER.info("Wake word detector stub stopped")

    def notify_detected(self, name: str) -> None:
        if self._on_detected is None:
            return
        self._on_detected(name)
