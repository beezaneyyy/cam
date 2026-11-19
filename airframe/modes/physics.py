"""PHYSICS — neon balls you can create, grab, throw and bat around.

* Pinch in empty space and hold: a ball inflates; release to drop/throw it
* Pinch a ball: grab it · release while moving: throw it
* Open hand: solid paddle — hit the balls
* G: gravity on/off · Del: remove all
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from airframe.rendering import theme as T
from airframe.rendering.glow import GlowLayer
from airframe.rendering.viewport import Viewport

from .base import Experience, Pointer

COLORS_BGR = [(255, 230, 40), (220, 60, 255), (90, 255, 140), (40, 190, 255), (255, 120, 120)]
MAX_BALLS = 24
MAX_SPEED = 3200.0
GRAVITY = 1100.0
RESTITUTION = 0.82


@dataclass(eq=False)
class Ball:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    color: tuple[int, int, int]
    grabbed_by: int | None = None
    offset: np.ndarray | None = None
    hit_flash: float = 0.0

    @property
    def mass(self) -> float:
        return self.radius ** 2


class PhysicsMode(Experience):
    name, label, accent = "physics", "Physics", T.GREEN
    hints = [
        ("HOLD PINCH", "inflate ball"),
        ("PINCH BALL", "grab · throw"),
        ("OPEN HAND", "paddle"),
        ("G", "gravity"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.balls: list[Ball] = []
        self.inflating: dict[int, Ball] = {}
        self.gravity = True
        self.paddles: list[tuple[np.ndarray, float]] = []
        self._color_i = 0

    def _next_color(self):
        c = COLORS_BGR[self._color_i % len(COLORS_BGR)]
        self._color_i += 1
        return c

    def ball_at(self, p) -> Ball | None:
        best, best_d = None, float("inf")
        for b in self.balls:
            if b.grabbed_by is not None:
                continue
            d = float(np.linalg.norm(b.pos - p)) - b.radius
            if d < 18 and d < best_d:
                best, best_d = b, d
        return best

    def update(self, pointers, dt, now, size):
        super().update(pointers, dt, now, size)
        dt = min(dt, 1 / 20)
        self.paddles = []
        for p in pointers:
            if p.started:
                b = self.ball_at(p.pos)
                if b is not None:
                    b.grabbed_by, b.offset = p.pid, b.pos - p.pos
                else:
                    nb = Ball(p.pos.copy(), np.zeros(2), 18.0, self._next_color(), grabbed_by=p.pid, offset=np.zeros(2))
                    self.inflating[p.pid] = nb
                    self.balls.append(nb)
                    if len(self.balls) > MAX_BALLS:
                        self.balls.pop(0)
            if p.pinching:
                if p.pid in self.inflating:
                    self.inflating[p.pid].radius = min(95.0, self.inflating[p.pid].radius + 45.0 * dt)
            if p.ended:
                self.inflating.pop(p.pid, None)
                for b in self.balls:
                    if b.grabbed_by == p.pid:
                        b.grabbed_by = None
                        v = p.velocity.copy()
                        speed = float(np.linalg.norm(v))
                        b.vel = v / speed * MAX_SPEED if speed > MAX_SPEED else v
            if p.is_open and not p.pinching and p.palm is not None and not p.vanished:
                self.paddles.append((p.palm, p.palm_size * 0.9))
                self._paddle(p.palm, p.palm_size * 0.9, p.velocity)

        by_pid = {p.pid: p for p in pointers}
        for b in self.balls:
            if b.grabbed_by is not None:
                p = by_pid.get(b.grabbed_by)
                if p is not None:
                    target = p.pos + (b.offset if b.offset is not None else 0)
                    b.vel = (target - b.pos) / max(dt, 1e-3)
                    b.pos = target.copy()
                continue
            if self.gravity:
                b.vel[1] += GRAVITY * dt
            b.vel *= math.exp(-0.25 * dt)
            b.pos = b.pos + b.vel * dt
            b.hit_flash = max(0.0, b.hit_flash - dt * 3)
            self._walls(b)
        self._collide()
        self.balls = [b for b in self.balls if b.radius > 0]

    def _walls(self, b: Ball) -> None:
        w, h = self.size
        for axis, hi in ((0, w), (1, h)):
            if b.pos[axis] - b.radius < 0:
                b.pos[axis] = b.radius
                if b.vel[axis] < 0:
                    b.vel[axis] = -b.vel[axis] * RESTITUTION
                    b.hit_flash = min(1.0, abs(b.vel[axis]) / 900)
            elif b.pos[axis] + b.radius > hi:
                b.pos[axis] = hi - b.radius
                if b.vel[axis] > 0:
                    b.vel[axis] = -b.vel[axis] * RESTITUTION
                    b.hit_flash = min(1.0, abs(b.vel[axis]) / 900)
                    if axis == 1:
                        b.vel[0] *= 0.97  # floor friction

    def _paddle(self, center: np.ndarray, radius: float, hand_vel: np.ndarray) -> None:
        for b in self.balls:
            if b.grabbed_by is not None:
                continue
            d = b.pos - center
            dist = float(np.linalg.norm(d))
            overlap = b.radius + radius - dist
            if overlap > 0 and dist > 1e-6:
                n = d / dist
                b.pos = b.pos + n * overlap
                rel = b.vel - hand_vel
                vn = float(rel @ n)
                if vn < 0:
                    b.vel = b.vel - (1 + RESTITUTION) * vn * n
                b.vel = b.vel + n * max(0.0, float(hand_vel @ n)) * 0.3
                b.hit_flash = 1.0

    def _collide(self) -> None:
        balls = self.balls
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                a, b = balls[i], balls[j]
                d = b.pos - a.pos
                dist = float(np.linalg.norm(d))
                overlap = a.radius + b.radius - dist
                if overlap <= 0 or dist < 1e-6:
                    continue
                n = d / dist
                ga, gb = a.grabbed_by is not None, b.grabbed_by is not None
                ma = 1e9 if ga else a.mass
                mb = 1e9 if gb else b.mass
                total = ma + mb
                if not ga:
                    a.pos = a.pos - n * overlap * (mb / total)
                if not gb:
                    b.pos = b.pos + n * overlap * (ma / total)
                rel = float((b.vel - a.vel) @ n)
                if rel < 0:
                    jimp = -(1 + RESTITUTION) * rel / (1 / ma + 1 / mb)
                    if not ga:
                        a.vel = a.vel - jimp / ma * n
                    if not gb:
                        b.vel = b.vel + jimp / mb * n
                    a.hit_flash = b.hit_flash = min(1.0, abs(rel) / 700)

    def handle_key(self, key, mod):
        if key == pygame.K_g:
            self.gravity = not self.gravity
            return "GRAVITY ON" if self.gravity else "ZERO GRAVITY"
        return None

    def clear(self):
        self.balls, self.inflating = [], {}
        return "BALLS CLEARED"

    def render_light(self, glow: GlowLayer, now: float) -> None:
        for b in self.balls:
            k = 1.0 + b.hit_flash
            col = tuple(min(255, int(c * 0.9 * k)) for c in b.color)
            glow.circle(b.pos, b.radius, col, 3 + 3 * b.hit_flash)
            glow.circle(b.pos, b.radius * 0.35, tuple(int(c * 0.35) for c in b.color), -1)

    def render_overlay(self, layer: pygame.Surface, vp: Viewport, now: float) -> None:
        for b in self.balls:
            c = vp.to_screen(b.pos)
            r = vp.length(b.radius)
            rgb = (b.color[2], b.color[1], b.color[0])
            pygame.draw.circle(layer, T.with_alpha(rgb, 0.18), c, r)
            hl = (c[0] - r * 0.35, c[1] - r * 0.35)
            pygame.draw.circle(layer, T.with_alpha(T.INK, 0.35), hl, max(2, r * 0.18))
            if b.grabbed_by is not None:
                pygame.draw.circle(layer, T.with_alpha(T.INK, 0.9), c, r + 6, 1)
        for center, radius in self.paddles:
            pygame.draw.circle(layer, T.with_alpha(T.GREEN, 0.15), vp.to_screen(center), vp.length(radius))
            pygame.draw.circle(layer, T.with_alpha(T.GREEN, 0.8), vp.to_screen(center), vp.length(radius), 2)
        f = T.font(10, mono=True)
        s = f.render(f"{len(self.balls):2d} BALLS  ·  GRAVITY {'ON' if self.gravity else 'OFF'}", True, T.INK_FAINT)
        layer.blit(s, (layer.get_width() - s.get_width() - 18, 130))
