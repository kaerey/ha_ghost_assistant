"""Fullscreen renderer placeholder."""
from __future__ import annotations

import asyncio
import logging

import pygame

LOGGER = logging.getLogger(__name__)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (90, 90, 90),
    "listening": (0, 160, 255),
    "thinking": (255, 200, 0),
    "speaking": (0, 220, 120),
}


class FullscreenRenderer:
    """Render a fullscreen placeholder window."""

    def __init__(self) -> None:
        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._rms: float = 0.0
        self._smoothed_rms: float = 0.0
        self._state: str = "idle"

    async def run(self, stop_event: asyncio.Event) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("HA Ghost Assistant")
        self._font = pygame.font.SysFont(None, 48)
        LOGGER.info("Fullscreen renderer started")

        try:
            while not stop_event.is_set():
                for event in pygame.event.get():
                    self._handle_event(event, stop_event)
                self._draw_frame()
                await asyncio.sleep(0.05)
        finally:
            self.close()

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        LOGGER.info("Renderer state set to %s", state)

    def set_rms(self, rms: float) -> None:
        self._rms = rms

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._font = None
            LOGGER.info("Fullscreen renderer stopped")

    def _handle_event(self, event: pygame.event.Event, stop_event: asyncio.Event) -> None:
        if event.type == pygame.QUIT:
            stop_event.set()
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            stop_event.set()
        elif event.key == pygame.K_w:
            self.set_state("listening")
        elif event.key == pygame.K_l:
            self.set_state("listening")
        elif event.key == pygame.K_t:
            self.set_state("thinking")
        elif event.key == pygame.K_s:
            self.set_state("speaking")
        elif event.key == pygame.K_i:
            self.set_state("idle")

    def _draw_frame(self) -> None:
        if self._screen is None or self._font is None:
            return
        self._screen.fill((0, 0, 0))
        self._smoothed_rms = (self._smoothed_rms * 0.85) + (self._rms * 0.15)
        width, height = self._screen.get_size()
        center = (width // 2, height // 2)
        base_radius = min(width, height) * 0.08
        radius = int(base_radius + (self._smoothed_rms * 600))
        color = STATE_COLORS.get(self._state, (255, 255, 255))
        pygame.draw.circle(self._screen, color, center, max(10, radius))

        label = self._font.render(self._state.upper(), True, (255, 255, 255))
        label_rect = label.get_rect(center=(center[0], center[1] + base_radius + 60))
        self._screen.blit(label, label_rect)
        pygame.display.flip()
