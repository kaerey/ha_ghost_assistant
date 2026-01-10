"""Audio playback placeholder."""
from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


class AudioPlayback:
    """Placeholder for default speaker playback."""

    async def start(self) -> None:
        LOGGER.info("Audio playback stub started")

    async def stop(self) -> None:
        LOGGER.info("Audio playback stub stopped")
