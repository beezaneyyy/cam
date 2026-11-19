"""PARTICLES — your fingertips shed light; your pinch is a black hole.

* Fingertips continuously emit particles (more when the hand moves fast)
* Pinch: gravity well — particles spiral into the pinch point
* Release the pinch: the gathered particles explode outward
* Open hand: repulsion field
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from airframe.rendering import theme as T
from airframe.rendering.glow import GlowLayer
from airframe.rendering.viewport import Viewport

from .base import Experience, Pointer

MAX_PARTICLES = 5000


class ParticleSystem:
    """Structure-of-arrays particle store, fully vectorised."""

    def __init__(self, capacity: int = MAX_PARTICLES, seed: int = 0):
        self.capacity = capacity
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)     # remaining seconds
        self.max_life = np.zeros(0, np.float32)
        self.hue = np.zeros(0, np.float32)      # 0..1 base hue
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pos)

    def emit(self, origin, n: int, base_vel=(0.0, 0.0), spread: float = 60.0, life: float = 2.2, hue: float = 0.5):
        if n <= 0:
            return
        rng = self.rng
        o = np.asarray(origin, np.float32).reshape(-1, 2)
        idx = rng.integers(0, len(o), n)
        p = o[idx] + rng.normal(0, 2.0, (n, 2)).astype(np.float32)
        ang = rng.uniform(0, math.tau, n)
        speed = rng.uniform(0.2, 1.0, n) * spread
        v = np.stack([np.cos(ang) * speed, np.sin(ang) * speed], 1).astype(np.float32)
        v += np.asarray(base_vel, np.float32) * 0.35
        lf = rng.uniform(0.6, 1.0, n).astype(np.float32) * life
        self.pos = np.concatenate([self.pos, p])
        self.vel = np.concatenate([self.vel, v])
        self.life = np.concatenate([self.life, lf])
        self.max_life = np.concatenate([self.max_life, lf])
        self.hue = np.concatenate([self.hue, np.full(n, hue, np.float32) + rng.normal(0, 0.03, n).astype(np.float32)])
        if len(self.pos) > self.capacity:
            k = len(self.pos) - self.capacity
            self.pos, self.vel = self.pos[k:], self.vel[k:]
            self.life, self.max_life, self.hue = self.life[k:], self.max_life[k:], self.hue[k:]

    def attract(self, center, strength: float, dt: float, swirl: float = 0.6, radius: float = 600.0) -> None:
        if not len(self):
            return
        d = np.asarray(center, np.float32) - self.pos
        r = np.linalg.norm(d, axis=1, keepdims=True) + 8.0
        inside = (r < radius).astype(np.float32)
        n = d / r
        tangent = np.concatenate([-n[:, 1:2], n[:, 0:1]], 1)
        acc = (n + tangent * swirl) * (strength / np.maximum(r, 30.0) * 60.0) * inside
        self.vel += acc * dt
        # particles reaching the core get trapped (keeps the ball tight)
        core = (r[:, 0] < 22)
        self.vel[core] *= 0.6
        self.life[core] = np.maximum(self.life[core], 0.5)

    def impulse(self, center, strength: float, radius: float) -> int:
        """Radial kick away from ``center`` (explosion). Returns affected count."""
        if not len(self):
            return 0
        d = self.pos - np.asarray(center, np.float32)
        r = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
        mask = (r[:, 0] < radius)
        falloff = np.clip(1.0 - r / radius, 0, 1)
        jitter = self.rng.uniform(0.6, 1.4, (len(self), 1)).astype(np.float32)
        self.vel += (d / r) * strength * falloff * jitter * mask[:, None]
        self.life[mask] = np.maximum(self.life[mask], self.max_life[mask] * 0.8)
        return int(mask.sum())

    def step(self, dt: float, size: tuple[int, int], drag: float = 1.4) -> None:
        if not len(self):
            return
        self.vel *= math.exp(-drag * dt)
        self.vel[:, 1] += 18.0 * dt  # a whisper of gravity
        self.pos += self.vel * dt
        self.life -= dt
        w, h = size
        alive = (self.life > 0) & (self.pos[:, 0] > -50) & (self.pos[:, 0] < w + 50) & (self.pos[:, 1] > -50) & (self.pos[:, 1] < h + 50)
        if not alive.all():
            self.pos, self.vel = self.pos[alive], self.vel[alive]
            self.life, self.max_life, self.hue = self.life[alive], self.max_life[alive], self.hue[alive]

    def colors(self) -> np.ndarray:
        """BGR uint8 per particle: hue shifts toward magenta with speed, fades with life."""
        if not len(self):
            return np.zeros((0, 3), np.uint8)
        speed = np.linalg.norm(self.vel, axis=1)
        h = (self.hue + np.clip(speed / 1500.0, 0, 0.35)) % 1.0
        fade = np.clip(self.life / np.maximum(self.max_life, 1e-3), 0, 1) ** 0.7
        # vectorised HSV→RGB (s=0.75, v=fade)
        s, v = 0.75, fade
        i = np.floor(h * 6).astype(np.int32) % 6
        f = h * 6 - np.floor(h * 6)
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        r = np.choose(i, [v, q, p, p, t, v])
        g = np.choose(i, [t, v, v, q, p, p])
        b = np.choose(i, [p, p, t, v, v, q])
        return (np.stack([b, g, r], 1) * 255).astype(np.uint8)


class ParticlesMode(Experience):
    name, label, accent = "particles", "Particles", T.CYAN
    hints = [
        ("MOVE HANDS", "shed particles"),
        ("PINCH", "black hole"),
        ("RELEASE", "explode"),
        ("OPEN HAND", "push away"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.ps = ParticleSystem()
        self.wells: dict[int, tuple[np.ndarray, float]] = {}   # pid → (pos, charge seconds)
        self.shockwaves: list[tuple[np.ndarray, float, float]] = []  # pos, start, power
        self.repellers: list[tuple[np.ndarray, float]] = []
        self.now = 0.0

    def update(self, pointers, dt, now, size):
        super().update(pointers, dt, now, size)
        self.now = now
        self.repellers = []
        for p in pointers:
            hue = 0.52 if p.handedness == "Right" else 0.78
            if p.pinching:
                pos, charge = self.wells.get(p.pid, (p.pos, 0.0))
                self.wells[p.pid] = (p.pos.copy(), charge + dt)
                self.ps.attract(p.pos, 900.0, dt)
                self.ps.emit(p.pos, 2, spread=20, life=1.2, hue=hue)
            else:
                if p.pid in self.wells:
                    pos, charge = self.wells.pop(p.pid)
                    power = min(1.0, 0.3 + charge / 2.0)
                    self.ps.impulse(pos, 900.0 * power, 380.0)
                    self.shockwaves.append((pos.copy(), now, power))
                if p.vanished:
                    continue
                speed = float(np.linalg.norm(p.velocity))
                n = int(2 + min(18, speed / 120))
                self.ps.emit(p.tips, n, base_vel=p.velocity, spread=45, hue=hue)
                if p.is_open and p.palm is not None:
                    radius = p.palm_size * 2.2
                    self.repellers.append((p.palm, radius))
                    self.ps.impulse(p.palm, 2400.0 * dt, radius)
        self.ps.step(dt, size)
        self.shockwaves = [s for s in self.shockwaves if now - s[1] < 0.8]

    def clear(self):
        self.ps = ParticleSystem()
        return "PARTICLES CLEARED"

    def render_light(self, glow: GlowLayer, now: float) -> None:
        glow.points(self.ps.pos, self.ps.colors())
        for pid, (pos, charge) in self.wells.items():
            r = 10 + 6 * math.sin(now * 12) + min(charge, 2.0) * 8
            glow.circle(pos, r + 14, (255, 140, 60), 2)
            glow.circle(pos, r * 0.6, (255, 255, 255), -1)
        for pos, start, power in self.shockwaves:
            age = (now - start) / 0.8
            glow.circle(pos, 20 + age * 380 * power, tuple(int(c * (1 - age)) for c in (255, 220, 120)), 4 * (1 - age) + 1)

    def render_overlay(self, layer: pygame.Surface, vp: Viewport, now: float) -> None:
        for pos, radius in self.repellers:
            pygame.draw.circle(layer, T.with_alpha(T.CYAN, 0.18), vp.to_screen(pos), vp.length(radius), 1)
        f = T.font(10, mono=True)
        s = f.render(f"{len(self.ps):5d} PARTICLES", True, T.INK_FAINT)
        layer.blit(s, (layer.get_width() - s.get_width() - 18, 130))
