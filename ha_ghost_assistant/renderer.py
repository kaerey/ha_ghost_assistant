"""Fullscreen renderer placeholder."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from typing import Callable, Protocol

import pygame

LOGGER = logging.getLogger(__name__)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (90, 90, 90),
    "listening": (0, 160, 255),
    "thinking": (255, 200, 0),
    "responding": (0, 220, 120),
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _cheap_blur(src: pygame.Surface, factor: int = 4) -> pygame.Surface:
    """Poor-man's blur: downscale then upscale (fast enough for bloom)."""
    w, h = src.get_size()
    sw, sh = max(1, w // factor), max(1, h // factor)
    small = pygame.transform.smoothscale(src, (sw, sh))
    return pygame.transform.smoothscale(small, (w, h))


class Renderer(Protocol):
    """Renderer protocol for swap-in visuals."""

    def set_density(self, density_1_to_10: int) -> None: ...

    def set_orbit(self, orbit_1_to_10: int) -> None: ...

    def set_style(self, style: str) -> None: ...

    async def run(self, stop_event: asyncio.Event) -> None: ...

    def set_state(self, state: str) -> None: ...

    def set_trigger(self, on_trigger: Callable[[], None] | None) -> None: ...

    def set_stop(self, on_stop: Callable[[], None] | None) -> None: ...

    def set_rms(self, rms: float) -> None: ...

    def close(self) -> None: ...


class FullscreenRenderer:
    """Render a fullscreen placeholder window."""

    def __init__(
        self,
        on_trigger: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._rms: float = 0.0
        self._smoothed_rms: float = 0.0
        self._env_fast: float = 0.0
        self._state: str = "idle"
        self._on_trigger = on_trigger
        self._on_stop = on_stop

        # Timing / perf
        self._clock: pygame.time.Clock | None = None
        self._t0 = time.perf_counter()

        # Visual buffers
        self._trail: pygame.Surface | None = None   # persistent (filaments + tails)
        self._fx: pygame.Surface | None = None      # per-frame effects (smoke + core)
        self._glow_cache: dict[int, pygame.Surface] = {}

        # Style + knobs
        self._style: str = "nebula"  # "nebula" (favorite), "orbital" (ringy), "hybrid"
        self._density_1_to_10: int = 10
        self._orbit_1_to_10: int = 4

        # Smoke + filament systems
        self._smoke: list[dict] = []
        self._tendrils: list[dict] = []
        self._tendril_rebuild_cd: float = 0.0

        # Optional orbital particles (used in "orbital"/"hybrid")
        self._particles: list[dict] = []

    # --- Optional knobs you can call from the app later ---
    def set_density(self, density_1_to_10: int) -> None:
        self._density_1_to_10 = int(_clamp(float(density_1_to_10), 1, 10))

    def set_orbit(self, orbit_1_to_10: int) -> None:
        self._orbit_1_to_10 = int(_clamp(float(orbit_1_to_10), 1, 10))

    def set_style(self, style: str) -> None:
        # style in {"nebula","orbital","hybrid"}
        self._style = style

    async def run(self, stop_event: asyncio.Event) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("HA Ghost Assistant")
        self._font = pygame.font.SysFont(None, 48)
        self._clock = pygame.time.Clock()
        LOGGER.info("Fullscreen renderer started")

        try:
            while not stop_event.is_set():
                for event in pygame.event.get():
                    self._handle_event(event, stop_event)
                self._draw_frame()
                await asyncio.sleep(0)  # yield; frame pacing handled by pygame clock
        finally:
            self.close()

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        LOGGER.info("State set to %s", state)

    def set_trigger(self, on_trigger: Callable[[], None] | None) -> None:
        self._on_trigger = on_trigger

    def set_stop(self, on_stop: Callable[[], None] | None) -> None:
        self._on_stop = on_stop

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
        LOGGER.info("Keydown: %s", pygame.key.name(event.key))
        if event.key == pygame.K_ESCAPE:
            stop_event.set()
        elif event.key == pygame.K_RETURN:
            if self._state == "idle":
                self.set_state("listening")
                if self._on_trigger is not None:
                    self._on_trigger()
            elif self._state == "listening":
                self.set_state("idle")
                if self._on_stop is not None:
                    self._on_stop()
        elif event.key == pygame.K_w:
            self.set_state("listening")
        elif event.key == pygame.K_l:
            self.set_state("listening")
        elif event.key == pygame.K_t:
            self.set_state("thinking")
        elif event.key == pygame.K_s:
            self.set_state("responding")
        elif event.key == pygame.K_i:
            self.set_state("idle")
        # Quick style toggles for testing
        elif event.key == pygame.K_1:
            self.set_style("nebula")
        elif event.key == pygame.K_2:
            self.set_style("orbital")
        elif event.key == pygame.K_3:
            self.set_style("hybrid")

    def _draw_frame(self) -> None:
        if self._screen is None or self._font is None or self._clock is None:
            return

        dt_ms = self._clock.tick(60)
        dt = dt_ms / 1000.0
        t = time.perf_counter() - self._t0

        w, h = self._screen.get_size()
        cx, cy = w // 2, int(h // 2 + h * 0.02)

        # Audio envelope (fast + smoothed)
        rms = _clamp(self._rms, 0.0, 1.0)
        self._smoothed_rms = (self._smoothed_rms * 0.86) + (rms * 0.14)
        self._env_fast = (self._env_fast * 0.65) + (rms * 0.35)

        focus = 1.0 if self._state in ("listening", "thinking", "responding") else 0.0
        speak = 1.0 if self._state == "responding" else 0.0

        base = min(w, h) * 0.16
        breathe = 0.5 + 0.5 * math.sin(t * 2.2)
        pulse = (0.06 + 0.42 * self._env_fast + 0.18 * (self._env_fast ** 2)) if speak > 0 else (0.08 * self._env_fast)
        r = base * (1.0 + 0.05 * breathe * (1.0 - focus) + 0.08 * speak + pulse)
        r = float(_clamp(r, 40.0, min(w, h) * 0.33))
        radius = int(r)

        # Allocate buffers as needed
        if self._trail is None or self._trail.get_size() != (w, h):
            self._trail = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
            self._trail.fill((0, 0, 0, 255))
        if self._fx is None or self._fx.get_size() != (w, h):
            self._fx = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()

        # Background: deep space with subtle center glow
        self._screen.fill((0, 0, 0))
        self._draw_soft_bg(self._screen, w, h)

        # Fade persistent trail (filaments/tails)
        # Lower subtract => longer persistence (more "smokey")
        fade_amt = 16 if self._style == "nebula" else 20
        self._trail.fill((0, 0, 0, fade_amt), special_flags=pygame.BLEND_RGBA_SUB)

        # FX layer rebuilt every frame
        self._fx.fill((0, 0, 0, 0))

        # Smoke field + core bloom (favorite look)
        if self._style in ("nebula", "hybrid"):
            self._ensure_smoke(cx, cy, radius)
            self._draw_smoke(self._fx, cx, cy, radius, t, focus, speak)
            self._step_tendrils(cx, cy, radius, dt, t, focus, speak)
            self._draw_tendrils(self._trail, cx, cy, radius, t, focus, speak)

        # Orbital particles (ringy look)
        if self._style in ("orbital", "hybrid"):
            self._ensure_particles(cx, cy, radius)
            self._step_particles(self._trail, cx, cy, radius, dt, t, focus, speak)

        # Core always on top (white-hot center)
        self._draw_core(self._fx, cx, cy, radius, t, focus, speak)

        # Bloom pass: blur fx lightly then add to screen
        # factor=4 is a good balance; use 3 if you want fatter bloom.
        blurred = _cheap_blur(self._fx, factor=4)
        self._screen.blit(blurred, (0, 0), special_flags=pygame.BLEND_ADD)
        self._screen.blit(self._fx, (0, 0), special_flags=pygame.BLEND_ADD)

        # Add trails (filaments/tails)
        self._screen.blit(self._trail, (0, 0), special_flags=pygame.BLEND_ADD)

        # Optional debug label (keep for now)
        label = self._font.render(f"{self._state.upper()}  [{self._style}]", True, (255, 255, 255))
        rect = label.get_rect(center=(cx, cy + radius + 60))
        self._screen.blit(label, rect)
        pygame.display.flip()

    # ----------------- Background / glow helpers -----------------
    def _draw_soft_bg(self, surf: pygame.Surface, w: int, h: int) -> None:
        # Subtle, centered haze only (avoid large floating blobs).
        rr = int(min(w, h) * 0.45)
        g = self._radial_glow(rr, (90, 30, 130), alpha=18)
        surf.blit(g, g.get_rect(center=(w // 2, h // 2)), special_flags=pygame.BLEND_ADD)

    def _radial_glow(self, radius: int, color: tuple[int, int, int], alpha: int = 120) -> pygame.Surface:
        # Simple cache key
        key = (radius << 16) ^ (alpha << 8) ^ (color[0] << 2) ^ (color[1] << 1) ^ color[2]
        cached = self._glow_cache.get(key)
        if cached is not None:
            return cached

        size = radius * 2 + 2
        s = pygame.Surface((size, size), pygame.SRCALPHA).convert_alpha()
        cx = cy = radius + 1
        steps = 28
        for i in range(steps, 0, -1):
            rr = int(radius * (i / steps))
            a = int(alpha * ((i / steps) ** 2))
            pygame.draw.circle(s, (color[0], color[1], color[2], a), (cx, cy), rr)

        self._glow_cache[key] = s
        return s

    # ----------------- Core + smoke (favorite nebula look) -----------------
    def _draw_core(self, fx: pygame.Surface, cx: int, cy: int, r: int, t: float, focus: float, speak: float) -> None:
        # Outer purple halo
        halo_r = int(r * (1.55 + 0.18 * self._env_fast))
        halo = self._radial_glow(halo_r, (176, 108, 255), alpha=int(50 + 90 * speak + 60 * self._env_fast))
        fx.blit(halo, halo.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_ADD)

        # Inner lavender halo
        halo2_r = int(r * (1.10 + 0.24 * self._env_fast))
        halo2 = self._radial_glow(halo2_r, (210, 175, 255), alpha=int(35 + 70 * speak + 55 * self._env_fast))
        fx.blit(halo2, halo2.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_ADD)

        # White-hot core bloom (the key!)
        core_r = int(r * (0.30 + 0.10 * self._env_fast + 0.10 * speak))
        core = self._radial_glow(core_r, (255, 255, 255), alpha=int(120 + 90 * speak + 70 * self._env_fast))
        fx.blit(core, core.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_ADD)

        # Slight rim highlight so the "sphere" reads
        rim_alpha = int(18 + 40 * focus + 45 * speak + 30 * self._env_fast)
        pygame.draw.circle(fx, (255, 255, 255, rim_alpha), (cx, cy), int(r * 0.98), width=2)

    def _ensure_smoke(self, cx: int, cy: int, r: int) -> None:
        d = self._density_1_to_10
        target = int(250 + (d - 1) * (650 / 9))  # ~250..900
        if len(self._smoke) == target:
            return
        if len(self._smoke) < target:
            for _ in range(target - len(self._smoke)):
                self._smoke.append({
                    "ang": random.random() * math.tau,
                    "rad": random.random(),
                    "seed": random.random() * 999.0,
                    "size": random.uniform(r * 0.06, r * 0.26),
                    "shade": random.random(),
                })
        else:
            self._smoke = self._smoke[:target]

    def _draw_smoke(self, fx: pygame.Surface, cx: int, cy: int, r: int, t: float, focus: float, speak: float) -> None:
        # Volumetric-ish cloud made of many soft glow blobs.
        # The distribution is biased to fill a sphere with lumpy edges.
        env = self._env_fast
        swell = 1.0 + 0.12 * env + 0.15 * speak
        for b in self._smoke:
            wob = (
                math.sin(t * 0.75 + b["seed"]) * 0.55 +
                math.cos(t * 0.55 + b["seed"] * 1.7) * 0.45
            )
            # Bias radius (more mid-cloud), add wobble for turbulence
            rr = (r * swell) * (0.22 + 0.95 * (b["rad"] ** 0.55)) + wob * r * 0.07
            ang = b["ang"] + wob * 0.20
            x = cx + math.cos(ang) * rr
            y = cy + math.sin(ang) * rr

            # Color shift: mostly purple, some lavender
            if b["shade"] < 0.75:
                col = (176, 108, 255)
            else:
                col = (210, 175, 255)

            # Alpha is kept low so the cloud doesn't overpower the core
            a = int(10 + 14 * (0.6 + 0.8 * env) + 10 * speak)
            g = self._radial_glow(max(6, int(b["size"])), col, alpha=a)
            fx.blit(g, g.get_rect(center=(x, y)), special_flags=pygame.BLEND_ADD)

    # ----------------- Filaments / tendrils (branchy plasma) -----------------
    def _step_tendrils(self, cx: int, cy: int, r: int, dt: float, t: float, focus: float, speak: float) -> None:
        # Rebuild tendrils periodically for “crawling” effect
        self._tendril_rebuild_cd -= dt
        d = self._density_1_to_10
        rebuild_rate = 0.10 if self._state == "responding" else 0.16
        if self._tendril_rebuild_cd > 0:
            return
        self._tendril_rebuild_cd = rebuild_rate

        # Tendril count scales with density; more in responding
        count = int(10 + (d - 1) * (26 / 9)) + (8 if speak > 0 else 0)  # ~10..36 (+8)
        self._tendrils = []
        for i in range(count):
            seed = int((t * 1000) + i * 1337 + random.randint(0, 9999))
            self._tendrils.append(self._build_tendril(cx, cy, r, seed, hot=(random.random() < 0.30 + 0.25 * speak)))

    def _build_tendril(self, cx: int, cy: int, r: int, seed: int, hot: bool = False) -> dict:
        rnd = random.Random(seed)
        # Start near the core
        start_r = r * rnd.uniform(0.05, 0.18)
        start_a = rnd.random() * math.tau
        x = cx + math.cos(start_a) * start_r
        y = cy + math.sin(start_a) * start_r

        pts: list[tuple[float, float]] = [(x, y)]
        ang = start_a + rnd.uniform(-1.0, 1.0)

        steps = rnd.randint(18, 34)
        step_len = r * rnd.uniform(0.035, 0.060)
        outward_bias = rnd.uniform(0.55, 0.85)
        jitter = rnd.uniform(0.45, 0.75)

        for i in range(steps):
            # Encourage outward movement (radial) but keep it jagged
            dx = pts[-1][0] - cx
            dy = pts[-1][1] - cy
            base_ang = math.atan2(dy, dx)
            ang = (ang * 0.55) + (base_ang * 0.45) + rnd.uniform(-0.65, 0.65) * jitter

            # Move
            x = pts[-1][0] + math.cos(ang) * step_len * (0.75 + 0.65 * outward_bias)
            y = pts[-1][1] + math.sin(ang) * step_len * (0.75 + 0.65 * outward_bias)
            # Side jag
            x += (rnd.random() - 0.5) * step_len * 1.2 * jitter
            y += (rnd.random() - 0.5) * step_len * 1.2 * jitter
            pts.append((x, y))

            # Stop if too far
            if math.hypot(x - cx, y - cy) > r * 1.05:
                break

        return {"pts": pts, "hot": hot}

    def _draw_tendrils(self, trail: pygame.Surface, cx: int, cy: int, r: int, t: float, focus: float, speak: float) -> None:
        # Draw each tendril twice (fat faint + thin bright) for filament look
        env = self._env_fast
        for td in self._tendrils:
            pts = td["pts"]
            if len(pts) < 2:
                continue
            # Convert to int points
            ipts = [(int(x), int(y)) for (x, y) in pts]

            # Outer glow stroke (purple)
            a1 = int(40 + 80 * env + 65 * speak)
            pygame.draw.lines(trail, (176, 108, 255, a1), False, ipts, width=3)

            # Inner bright stroke (lavender/white)
            if td["hot"]:
                a2 = int(70 + 110 * env + 95 * speak)
                pygame.draw.lines(trail, (255, 255, 255, a2), False, ipts, width=1)
            else:
                a2 = int(65 + 105 * env + 85 * speak)
                pygame.draw.lines(trail, (210, 175, 255, a2), False, ipts, width=1)

    # ----------------- Optional orbital particles (ringy look) -----------------
    def _ensure_particles(self, cx: int, cy: int, r: int) -> None:
        d = self._density_1_to_10
        target = int(250 + (d - 1) * (1400 / 9))  # ~250..1650
        if len(self._particles) == target:
            return
        if len(self._particles) < target:
            for _ in range(target - len(self._particles)):
                self._particles.append(self._spawn_particle(cx, cy, r))
        else:
            self._particles = self._particles[:target]

    def _spawn_particle(self, cx: int, cy: int, r: int) -> dict:
        ang = random.random() * math.tau
        orbit = (self._orbit_1_to_10 - 1) / 9.0  # 0..1
        inner = 0.35 + 0.40 * (1.0 - orbit)
        outer = 0.95 + 0.70 * orbit
        rr = r * (inner + random.random() * (outer - inner))
        x = cx + math.cos(ang) * rr
        y = cy + math.sin(ang) * rr
        return {
            "x": x, "y": y,
            "px": x, "py": y,
            "z": random.random(),
            "seed": random.random() * 999.0,
            "life": 1.4 + random.random() * 3.6,
            "age": random.random() * 3.6,
        }

    def _step_particles(self, trail: pygame.Surface, cx: int, cy: int, r: int, dt: float, t: float, focus: float, speak: float) -> None:
        orbit = (self._orbit_1_to_10 - 1) / 9.0  # 0..1
        inner_keep = r * (0.35 + 0.45 * (1.0 - orbit))
        outer_keep = r * (1.05 + 0.75 * orbit)
        speed = (18.0 + 60.0 * self._env_fast) * (0.70 + 0.90 * speak + 0.35 * focus)

        for p in self._particles:
            p["age"] += dt
            if p["age"] > p["life"]:
                p.update(self._spawn_particle(cx, cy, r))
                continue

            dx = p["x"] - cx
            dy = p["y"] - cy
            dist = math.hypot(dx, dy) + 1e-6
            nx, ny = dx / dist, dy / dist
            tx, ty = -ny, nx

            wob = math.sin(t * 1.3 + p["seed"]) * 0.6 + math.cos(t * 0.9 + p["seed"] * 1.7) * 0.4
            mix = 0.55 + 0.35 * orbit
            vx = (tx * mix + nx * (0.10 * wob)) * speed * (0.35 + 0.9 * p["z"])
            vy = (ty * mix + ny * (0.10 * wob)) * speed * (0.35 + 0.9 * p["z"])

            ax = ay = 0.0
            if dist < inner_keep:
                push = (inner_keep - dist) / inner_keep
                ax += nx * speed * 0.55 * push
                ay += ny * speed * 0.55 * push
            elif dist > outer_keep:
                pull = (dist - outer_keep) / outer_keep
                ax -= nx * speed * 0.75 * pull
                ay -= ny * speed * 0.75 * pull

            p["px"], p["py"] = p["x"], p["y"]
            p["x"] += (vx + ax) * dt
            p["y"] += (vy + ay) * dt

            rr = math.hypot(p["x"] - cx, p["y"] - cy)
            if rr > r * 2.0:
                p.update(self._spawn_particle(cx, cy, r))
                continue

            life01 = p["age"] / p["life"]
            fade = math.sin(math.pi * (1.0 - life01))
            rim_boost = _clamp((rr - r * 0.70) / (r * 0.70), 0.0, 1.0)

            alpha = (18 + 120 * fade) * (0.55 + 0.85 * self._env_fast) * (0.55 + 0.55 * p["z"]) * (0.55 + 0.55 * rim_boost)
            alpha = int(_clamp(alpha, 0.0, 255.0))
            size = (1.0 + 2.2 * p["z"]) * (0.75 + 0.85 * self._env_fast)
            size_i = max(1, int(size))

            if (p["seed"] * 10 + t) % 2.0 < 1.0:
                col = (190, 110, 255, alpha)
            else:
                col = (124, 240, 255, int(alpha * 0.9))

            pygame.draw.aaline(trail, col, (p["px"], p["py"]), (p["x"], p["y"]))
            pygame.draw.circle(trail, col, (int(p["x"]), int(p["y"])), size_i)


def build_renderer() -> Renderer:
    """Create the renderer implementation based on environment."""
    if os.getenv("HA_GHOST_ASSISTANT_RENDERER", "gles").lower() == "pygame":
        return FullscreenRenderer()
    from ha_ghost_assistant.gles_renderer import GLESRenderer

    return GLESRenderer()
