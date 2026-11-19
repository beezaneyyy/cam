"""ENERGY — hold a ball of lightning between your hands.

* Two hands: an energy orb forms between the palms; lightning arcs from every
  fingertip into it. Its size follows the distance between your hands.
* Pinch with both hands: charge it (more arcs, bigger, brighter)
* Release after charging: BLAST — shockwave rings and a flash
* One hand: an electric arc jumps between thumb and index as they approach
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from airframe.rendering import theme as T
from airframe.rendering.glow import GlowLayer
from airframe.rendering.viewport import Viewport

from .base import Experience, Pointer

ARC_BGR = (255, 200, 120)        # electric blue
CORE_BGR = (255, 245, 230)
CHARGED_BGR = (255, 120, 210)    # violet when fully charged
CHARGE_SECONDS = 1.4


def lightning(a, b, rng: np.random.Generator, roughness: float = 0.22, depth: int = 5) -> np.ndarray:
    """Jagged bolt from ``a`` to ``b`` by recursive midpoint displacement. Returns (2**depth+1, 2)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    pts = np.array([a, b])
    length = float(np.linalg.norm(b - a))
    amp = length * roughness
    for _ in range(depth):
        mids = (pts[:-1] + pts[1:]) / 2
        seg = pts[1:] - pts[:-1]
        normal = np.stack([-seg[:, 1], seg[:, 0]], 1)
        norm = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
        mids = mids + normal / norm * rng.normal(0, amp, (len(mids), 1))
        out = np.empty((len(pts) + len(mids), 2))
        out[0::2], out[1::2] = pts, mids
        pts = out
        amp *= 0.5
    return pts


class EnergyMode(Experience):
    name, label, accent = "energy", "Energy", T.BLUE
    hints = [
        ("TWO HANDS", "summon orb"),
        ("HANDS APART", "bigger orb"),
        ("PINCH BOTH", "charge"),
        ("RELEASE", "blast"),
    ]

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.charge = 0.0
        self.orb: tuple[np.ndarray, float] | None = None
        self.orb_strength = 0.0             # fades the orb in / out
        self.tips: list[np.ndarray] = []
        self.single_arcs: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.blasts: list[tuple[np.ndarray, float, float]] = []   # center, start time, power
        self.flash = 0.0
        self.now = 0.0

    def update(self, pointers, dt, now, size):
        super().update(pointers, dt, now, size)
        self.now = now
        hands = [p for p in pointers if p.hand is not None and not p.vanished]
        self.single_arcs = []
        self.flash = max(0.0, self.flash - dt * 2.5)

        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            center = (a.palm + b.palm) / 2
            dist = float(np.linalg.norm(a.palm - b.palm))
            both = a.pinching and b.pinching
            if both:
                self.charge = min(1.0, self.charge + dt / CHARGE_SECONDS)
            radius = float(np.clip(dist * 0.11, 14, 120)) * (1 + 0.6 * self.charge)
            self.orb = (center, radius)
            self.orb_strength = min(1.0, self.orb_strength + dt * 4)
            self.tips = [t for h in (a, b) for t in h.tips]
            if not both and self.charge > 0:
                if self.charge > 0.35:
                    self._blast(center, self.charge)
                self.charge = 0.0
        else:
            if self.orb is not None and self.charge > 0.35:
                self._blast(self.orb[0], self.charge)
            self.charge = 0.0
            self.orb_strength = max(0.0, self.orb_strength - dt * 3)
            if self.orb_strength <= 0:
                self.orb = None
            self.tips = []
            for h in hands:
                thumb, index = h.tips[0], h.tips[1]
                gap = float(np.linalg.norm(thumb - index)) / max(h.palm_size, 1)
                if 0.05 < gap < 0.9:
                    self.single_arcs.append((thumb, index, 1.0 - gap / 0.9))
        self.blasts = [bl for bl in self.blasts if now - bl[1] < 1.0]

    def _blast(self, center, power: float) -> None:
        self.blasts.append((np.asarray(center).copy(), self.now, power))
        self.flash = 0.5 * power

    def clear(self):
        self.blasts, self.charge = [], 0.0
        return None

    # -- rendering --------------------------------------------------------------------------
    def render_light(self, glow: GlowLayer, now: float) -> None:
        rng = self.rng
        if self.orb is not None and self.orb_strength > 0:
            center, radius = self.orb
            s = self.orb_strength
            col = tuple(int(ARC_BGR[i] + (CHARGED_BGR[i] - ARC_BGR[i]) * self.charge) for i in range(3))
            col = tuple(int(c * s) for c in col)
            pulse = 1 + 0.08 * math.sin(now * 20) + 0.15 * self.charge * math.sin(now * 37)
            glow.circle(center, radius * pulse, tuple(int(c * 0.45) for c in col), -1)   # plasma body
            glow.circle(center, radius * pulse, col, 3)                                  # rim
            glow.circle(center, radius * 0.3 * pulse, tuple(int(c * s * 0.8) for c in CORE_BGR), -1)
            # orbiting rings
            for k in range(2 + int(self.charge * 3)):
                ang = now * (2.5 + k) + k * 1.7
                rr = radius * (1.35 + 0.25 * k)
                p = center + rr * np.array([math.cos(ang), math.sin(ang) * 0.45])
                glow.circle(p, 3 + self.charge * 3, col, -1)
            # arcs from fingertips (only some each frame → crackle)
            p_arc = 0.35 + 0.6 * self.charge
            for tip in self.tips:
                if rng.random() > p_arc:
                    continue
                d = center - tip
                n = float(np.linalg.norm(d))
                if n < radius:
                    continue
                end = center - d / n * radius * 0.8
                bolt = lightning(tip, end, rng, 0.12 + 0.08 * self.charge, depth=5)
                glow.polyline(bolt, tuple(int(c * 0.7) for c in col), 2 + self.charge)
                glow.polyline(bolt, tuple(int(c * s * 0.9) for c in CORE_BGR), 1)
        for a, b, strength in self.single_arcs:
            if rng.random() < 0.3 + 0.6 * strength:
                bolt = lightning(a, b, rng, 0.3, depth=4)
                glow.polyline(bolt, ARC_BGR, 1 + 2 * strength)
                glow.polyline(bolt, CORE_BGR, 1)
            glow.circle(a, 3 + 4 * strength, ARC_BGR, -1)
            glow.circle(b, 3 + 4 * strength, ARC_BGR, -1)
        for center, start, power in self.blasts:
            age = (now - start) / 1.0
            fade = (1 - age) ** 1.5
            for k in range(3):
                r = (60 + 900 * power) * min(1.0, age * (1.0 + 0.35 * k))
                glow.circle(center, r, tuple(int(c * fade) for c in (CHARGED_BGR if k == 1 else ARC_BGR)), 6 * fade + 1)
            # radial bolts on the shockwave front
            if age < 0.4:
                for _ in range(6):
                    ang = rng.uniform(0, math.tau)
                    end = center + (80 + 500 * age * power) * np.array([math.cos(ang), math.sin(ang)])
                    glow.polyline(lightning(center, end, rng, 0.25, 4), CORE_BGR, 2)
        if self.flash > 0.01:
            glow.canvas[...] = np.maximum(glow.canvas, np.array([int(c * self.flash * 0.5) for c in ARC_BGR], np.uint8))
            glow.dirty = True

    def render_overlay(self, layer: pygame.Surface, vp: Viewport, now: float) -> None:
        if self.orb is None:
            return
        center, radius = self.orb
        c = vp.to_screen(center)
        r = vp.length(radius) * 1.9
        rect = pygame.Rect(0, 0, r * 2, r * 2)
        rect.center = (c[0], c[1])
        pygame.draw.circle(layer, T.with_alpha(T.INK_FAINT, 0.6 * self.orb_strength), c, r, 1)
        if self.charge > 0:
            pygame.draw.arc(layer, T.with_alpha(T.VIOLET, 0.95), rect, math.pi / 2, math.pi / 2 + self.charge * math.tau, 3)
            txt = T.font(10, mono=True).render(f"CHARGE {self.charge * 100:3.0f}%", True, T.VIOLET)
            layer.blit(txt, txt.get_rect(midtop=(c[0], c[1] + r + 6)))
