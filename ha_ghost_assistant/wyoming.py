"""Home Assistant Wyoming satellite client (stub)."""
from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


class WyomingClient:
    """Placeholder for Wyoming protocol client."""

    async def connect(self) -> None:
        LOGGER.info("Wyoming client stub connect")

    async def disconnect(self) -> None:
        LOGGER.info("Wyoming client stub disconnect")
