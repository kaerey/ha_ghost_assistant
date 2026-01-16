"""OpenGL ES renderer for the HA Ghost Assistant visuals."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pygame
from OpenGL import GL

LOGGER = logging.getLogger(__name__)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class _ParticleField:
    positions: np.ndarray
    previous: np.ndarray
    seeds: np.ndarray
    ages: np.ndarray
    life: np.ndarray


class _ShaderProgram:
    def __init__(self, vertex_src: str, fragment_src: str) -> None:
        self.program = GL.glCreateProgram()
        self._uniform_locs: dict[str, int] = {}
        self._attrib_locs: dict[str, int] = {}
        vertex = self._compile(GL.GL_VERTEX_SHADER, vertex_src)
        fragment = self._compile(GL.GL_FRAGMENT_SHADER, fragment_src)
        GL.glAttachShader(self.program, vertex)
        GL.glAttachShader(self.program, fragment)
        GL.glLinkProgram(self.program)
        status = GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS)
        if status != GL.GL_TRUE:
            info = GL.glGetProgramInfoLog(self.program).decode("utf-8")
            raise RuntimeError(f"Shader link failed: {info}")
        GL.glDeleteShader(vertex)
        GL.glDeleteShader(fragment)

    @staticmethod
    def _compile(shader_type: int, source: str) -> int:
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        if status != GL.GL_TRUE:
            info = GL.glGetShaderInfoLog(shader).decode("utf-8")
            raise RuntimeError(f"Shader compile failed: {info}")
        return shader

    def use(self) -> None:
        GL.glUseProgram(self.program)

    def uniform(self, name: str) -> int:
        """Cached glGetUniformLocation."""
        loc = self._uniform_locs.get(name)
        if loc is not None:
            return loc
        loc = GL.glGetUniformLocation(self.program, name)
        self._uniform_locs[name] = loc
        return loc

    def attrib(self, name: str) -> int:
        """Cached glGetAttribLocation."""
        loc = self._attrib_locs.get(name)
        if loc is not None:
            return loc
        loc = GL.glGetAttribLocation(self.program, name)
        self._attrib_locs[name] = loc
        return loc


class GLESRenderer:
    """OpenGL ES renderer with GPU glow + dense particle field."""

    def __init__(
        self,
        on_trigger: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._t0 = time.perf_counter()
        self._state: str = "idle"
        self._rms: float = 0.0
        self._env_fast: float = 0.0
        self._env_slow: float = 0.0
        self._on_trigger = on_trigger
        self._on_stop = on_stop
        self._density_1_to_10: int = 10
        self._orbit_1_to_10: int = 6
        self._field: _ParticleField | None = None
        self._bg_shader: _ShaderProgram | None = None
        self._particle_shader: _ShaderProgram | None = None
        self._quad_vbo: int | None = None
        self._particle_vbo: int | None = None
        self._particle_trail_vbo: int | None = None
        self._bg_a_pos: int | None = None
        self._p_a_pos: int | None = None
        self._frame_index: int = 0
        self._responding_timeout: asyncio.Task[None] | None = None

    def set_density(self, density_1_to_10: int) -> None:
        self._density_1_to_10 = int(_clamp(float(density_1_to_10), 1, 10))
        self._field = None

    def set_orbit(self, orbit_1_to_10: int) -> None:
        self._orbit_1_to_10 = int(_clamp(float(orbit_1_to_10), 1, 10))

    def set_style(self, style: str) -> None:
        _ = style

    async def run(self, stop_event: asyncio.Event) -> None:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 0)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_ES)
        self._screen = pygame.display.set_mode(
            (0, 0), pygame.FULLSCREEN | pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("HA Ghost Assistant (GLES)")
        self._clock = pygame.time.Clock()
        self._setup_gl()
        LOGGER.info("OpenGL ES renderer started")

        try:
            while not stop_event.is_set():
                for event in pygame.event.get():
                    self._handle_event(event, stop_event)
                self._draw_frame()
                await asyncio.sleep(0)
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
            self._clock = None
        if self._responding_timeout is not None:
            self._responding_timeout.cancel()
            self._responding_timeout = None
            LOGGER.info("OpenGL ES renderer stopped")

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
            else:
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

    def _setup_gl(self) -> None:
        if self._screen is None:
            return
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        self._bg_shader = _ShaderProgram(_BG_VERTEX, _BG_FRAGMENT)
        self._particle_shader = _ShaderProgram(_PARTICLE_VERTEX, _PARTICLE_FRAGMENT)

        quad = np.array(
            [
                -1.0,
                -1.0,
                3.0,
                -1.0,
                -1.0,
                3.0,
            ],
            dtype=np.float32,
        )
        self._quad_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._quad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)

        self._particle_vbo = GL.glGenBuffers(1)
        self._particle_trail_vbo = GL.glGenBuffers(1)

        if self._bg_shader is not None:
            self._bg_a_pos = self._bg_shader.attrib("a_pos")
        if self._particle_shader is not None:
            self._p_a_pos = self._particle_shader.attrib("a_pos")

    def _draw_frame(self) -> None:
        if self._screen is None or self._clock is None:
            return
        width, height = self._screen.get_size()
        now = time.perf_counter() - self._t0
        dt = self._clock.tick(60) / 1000.0
        self._frame_index += 1

        rms = _clamp(self._rms, 0.0, 1.0)
        self._env_fast = (self._env_fast * 0.62) + (rms * 0.38)
        self._env_slow = (self._env_slow * 0.90) + (rms * 0.10)
        focus = 1.0 if self._state in ("listening", "thinking", "responding") else 0.0
        speak = 1.0 if self._state == "responding" else 0.0
        slow_tick = speak > 0.5 and (self._frame_index % 2 == 0)

        base = min(width, height) * 0.16
        pulse = (0.08 + 0.28 * self._env_fast + 0.12 * (self._env_fast**2)) if speak > 0 else (
            0.08 * self._env_fast
        )
        radius = base * (1.0 + 0.08 * focus + 0.10 * speak + pulse)
        radius = float(_clamp(radius, 40.0, min(width, height) * 0.33))

        cx, cy = width * 0.5, height * 0.5

        field = self._ensure_particles(radius, cx, cy, speak)
        if not slow_tick:
            self._update_particles(field, radius, cx, cy, dt * (2.0 if speak > 0.5 else 1.0), now)

        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Solid black background with a tight core glow.
        self._draw_background(width, height, radius, focus, speak, now)
        if not slow_tick:
            self._draw_particles(field, width, height, radius, speak)

        pygame.display.flip()

    def _draw_background(
        self, width: int, height: int, radius: float, focus: float, speak: float, t: float
    ) -> None:
        if self._bg_shader is None or self._quad_vbo is None:
            return
        self._bg_shader.use()
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._quad_vbo)
        pos_loc = self._bg_a_pos if self._bg_a_pos is not None else self._bg_shader.attrib("a_pos")
        GL.glEnableVertexAttribArray(pos_loc)
        GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glUniform2f(self._bg_shader.uniform("u_resolution"), width, height)
        GL.glUniform2f(self._bg_shader.uniform("u_center"), width * 0.5, height * 0.5)
        GL.glUniform1f(self._bg_shader.uniform("u_radius"), radius)
        GL.glUniform1f(self._bg_shader.uniform("u_focus"), focus)
        GL.glUniform1f(self._bg_shader.uniform("u_speak"), speak)
        GL.glUniform1f(self._bg_shader.uniform("u_time"), t)
        GL.glUniform1f(self._bg_shader.uniform("u_env"), self._env_fast)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def _draw_particles(
        self, field: _ParticleField, width: int, height: int, radius: float, speak: float
    ) -> None:
        if self._particle_shader is None or self._particle_vbo is None or self._particle_trail_vbo is None:
            return
        self._particle_shader.use()
        GL.glUniform2f(self._particle_shader.uniform("u_resolution"), width, height)
        GL.glUniform1f(self._particle_shader.uniform("u_radius"), radius)
        GL.glUniform1f(self._particle_shader.uniform("u_env"), self._env_fast)

        positions = field.positions
        prev = field.previous

        pos_loc = self._p_a_pos if self._p_a_pos is not None else self._particle_shader.attrib("a_pos")
        GL.glEnableVertexAttribArray(pos_loc)
        draw_trails = speak <= 0.5
        if draw_trails:
            trail = np.hstack([prev, positions]).reshape(-1, 2)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._particle_trail_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, trail.nbytes, trail, GL.GL_DYNAMIC_DRAW)
            GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glUniform1f(self._particle_shader.uniform("u_point_size"), 1.0)
            GL.glUniform1f(self._particle_shader.uniform("u_alpha"), 0.12)
            GL.glDrawArrays(GL.GL_LINES, 0, trail.shape[0])

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._particle_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, positions.nbytes, positions, GL.GL_DYNAMIC_DRAW)
        GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glUniform1f(self._particle_shader.uniform("u_point_size"), 2.4)
        GL.glUniform1f(self._particle_shader.uniform("u_alpha"), 0.72)
        GL.glDrawArrays(GL.GL_POINTS, 0, positions.shape[0])

    def _ensure_particles(self, radius: float, cx: float, cy: float, speak: float) -> _ParticleField:
        base_target = int(600 + (self._density_1_to_10 - 1) * (3600 / 9))
        if speak > 0.5:
            target_scale = 0.18
        elif speak > 0.2:
            target_scale = 0.55
        else:
            target_scale = 1.0
        target = max(200, int(base_target * target_scale))
        if self._field is not None and self._field.positions.shape[0] == target:
            return self._field

        ang = np.random.uniform(0.0, math.tau, target).astype(np.float32)
        orbit = (self._orbit_1_to_10 - 1) / 9.0
        inner = 0.25 + 0.50 * (1.0 - orbit)
        outer = 0.95 + 0.70 * orbit
        rr = radius * np.random.uniform(inner, outer, target).astype(np.float32)
        x = cx + np.cos(ang) * rr
        y = cy + np.sin(ang) * rr
        positions = np.stack([x, y], axis=1).astype(np.float32)
        previous = positions.copy()
        seeds = np.random.uniform(0.0, 1000.0, target).astype(np.float32)
        ages = np.random.uniform(0.0, 3.0, target).astype(np.float32)
        life = np.random.uniform(1.6, 4.6, target).astype(np.float32)
        self._field = _ParticleField(positions, previous, seeds, ages, life)
        return self._field

    def _update_particles(
        self, field: _ParticleField, radius: float, cx: float, cy: float, dt: float, t: float
    ) -> None:
        orbit = (self._orbit_1_to_10 - 1) / 9.0
        inner_keep = radius * (0.28 + 0.46 * (1.0 - orbit))
        outer_keep = radius * (1.02 + 0.65 * orbit)
        speed = (26.0 + 90.0 * self._env_fast) * (0.68 + 0.65 * orbit)

        pos = field.positions
        prev = field.previous
        seeds = field.seeds
        ages = field.ages
        life = field.life

        prev[:] = pos
        ages += dt

        dx = (pos[:, 0] - cx) / radius
        dy = (pos[:, 1] - cy) / radius
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        tang_x = -dy / dist
        tang_y = dx / dist

        flow = np.sin(dx * 2.4 + t * 0.7 + seeds * 0.002) + np.cos(dy * 2.0 - t * 0.6 + seeds * 0.0013)
        angle = flow * 1.4 + tang_x * 0.3
        fx = np.cos(angle)
        fy = np.sin(angle)
        mix = 0.55 + 0.35 * orbit
        vx = (fx * (1.0 - mix) + tang_x * mix) * speed
        vy = (fy * (1.0 - mix) + tang_y * mix) * speed

        ax = np.zeros_like(vx)
        ay = np.zeros_like(vy)
        radial = dist * radius
        inner_mask = radial < inner_keep
        outer_mask = radial > outer_keep
        inner_push = (inner_keep - radial) / inner_keep
        outer_pull = (radial - outer_keep) / outer_keep
        ax[inner_mask] += (dx[inner_mask] / dist[inner_mask]) * speed * 0.65 * inner_push[inner_mask]
        ay[inner_mask] += (dy[inner_mask] / dist[inner_mask]) * speed * 0.65 * inner_push[inner_mask]
        ax[outer_mask] -= (dx[outer_mask] / dist[outer_mask]) * speed * 0.70 * outer_pull[outer_mask]
        ay[outer_mask] -= (dy[outer_mask] / dist[outer_mask]) * speed * 0.70 * outer_pull[outer_mask]

        pos[:, 0] += (vx + ax) * dt
        pos[:, 1] += (vy + ay) * dt

        reset_mask = (ages > life) | (np.sqrt((pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2) > radius * 2.0)
        if np.any(reset_mask):
            count = reset_mask.sum()
            ang = np.random.uniform(0.0, math.tau, count)
            inner = 0.25 + 0.50 * (1.0 - orbit)
            outer = 0.95 + 0.70 * orbit
            rr = radius * np.random.uniform(inner, outer, count)
            pos[reset_mask, 0] = cx + np.cos(ang) * rr
            pos[reset_mask, 1] = cy + np.sin(ang) * rr
            prev[reset_mask] = pos[reset_mask]
            ages[reset_mask] = 0.0
            life[reset_mask] = np.random.uniform(1.6, 4.6, count)


_BG_VERTEX = """
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
    v_uv = (a_pos + 1.0) * 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_BG_FRAGMENT = """
precision mediump float;
varying vec2 v_uv;
uniform vec2 u_resolution;
uniform vec2 u_center;
uniform float u_radius;
uniform float u_focus;
uniform float u_speak;
uniform float u_time;
uniform float u_env;
void main() {
    vec2 frag = v_uv * u_resolution;
    vec2 d = (frag - u_center) / max(u_radius, 1.0);
    float r = length(d);
    float core = exp(-r * r * 6.0);
    float halo = exp(-r * r * 2.6);
    float fade = smoothstep(1.4, 0.6, r);
    vec3 core_col = vec3(0.95, 0.85, 1.0);
    vec3 halo_col = vec3(0.69, 0.42, 1.0);
    vec3 corona_col = vec3(0.23, 0.06, 0.45);
    vec3 color = vec3(0.0);
    color += core_col * core * (0.8 + 0.5 * u_env + 0.4 * u_speak);
    color += halo_col * halo * (0.25 + 0.4 * u_env + 0.2 * u_focus);
    color *= fade;
    gl_FragColor = vec4(color, 1.0);
}
"""

_PARTICLE_VERTEX = """
attribute vec2 a_pos;
uniform vec2 u_resolution;
uniform float u_point_size;
void main() {
    vec2 clip = (a_pos / u_resolution) * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
"""

_PARTICLE_FRAGMENT = """
precision mediump float;
uniform float u_alpha;
uniform float u_env;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float r = length(c) * 2.0;
    float glow = exp(-r * r * 2.2);
    vec3 col = mix(vec3(0.65, 0.35, 1.0), vec3(0.48, 0.95, 1.0), u_env);
    gl_FragColor = vec4(col * glow, glow * u_alpha);
}
"""
