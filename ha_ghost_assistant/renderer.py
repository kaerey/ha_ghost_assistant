"""Fullscreen renderer placeholder."""
from __future__ import annotations

import asyncio
import logging

import pygame

LOGGER = logging.getLogger(__name__)


class FullscreenRenderer:
    """Render a fullscreen placeholder window."""

    def __init__(self) -> None:
        self._screen: pygame.Surface | None = None

    async def run(self, stop_event: asyncio.Event) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("HA Ghost Assistant")
        self._screen.fill((0, 0, 0))
        pygame.display.flip()
        LOGGER.info("Fullscreen renderer started")

        try:
            while not stop_event.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop_event.set()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        stop_event.set()
                await asyncio.sleep(0.05)
        finally:
            self.close()

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            LOGGER.info("Fullscreen renderer stopped")
