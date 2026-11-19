"""DRAW — paint neon light in the air.

* Pinch and move: draw a stroke (each hand draws its own stroke at the same time)
* Open hand over strokes: erase (like wiping a glass board)
* Open-hand swipe / ← →: change colour · [ ]: brush size · Z: undo · Del: clear
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field

import numpy as np
import pygame

from airframe.rendering import theme as T
from airframe.rendering.glow import GlowLayer
from airframe.rendering.viewport import Viewport

from .base import Experience, Pointer

PALETTE = [
    ("CYAN", (255, 230, 40)),
    ("MAGENTA", (220, 60, 255)),
    ("LIME", (90, 255, 140)),
    ("AMBER", (40, 190, 255)),
    ("WHITE", (245, 245, 245)),
    ("RAINBOW", None),
]
MIN_SEGMENT = 3.0      # px between stored points
MAX_POINTS = 30000


@dataclass(eq=False)
class Stroke:
    points: list[np.ndarray] = field(default_factory=list)
    color: tuple[int, int, int] | None = (255, 230, 40)   # None = rainbow
    width: float = 6.0
    born: float = 0.0

    def add(self, p: np.ndarray) -> bool:
        if not self.points or np.linalg.norm(p - self.points[-1]) >= MIN_SEGMENT:
            self.points.append(p.copy())
            return True
        return False


def _rainbow(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.006) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class DrawMode(Experience):
    name, label, accent = "draw", "Draw", T.MAGENTA
    hints = [
        ("PINCH + MOVE", "draw"),
        ("OPEN HAND", "erase"),
        ("SWIPE  ← →", "colour"),
        ("[ ]  Z  DEL", "size · undo · clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.strokes: list[Stroke] = []
        self.active: dict[int, Stroke] = {}
        self.color_index = 0
        self.width = 7.0
        self.erasers: list[tuple[np.ndarray, float]] = []
        self.cursors: list[Pointer] = []

    @property
    def color_name(self) -> str:
        return PALETTE[self.color_index][0]

    def cycle_color(self, d: int) -> str:
        self.color_index = (self.color_index + d) % len(PALETTE)
        return f"COLOUR · {self.color_name}"

    def update(self, pointers, dt, now, size):
        super().update(pointers, dt, now, size)
        self.erasers = []
        self.cursors = [p for p in pointers if not p.vanished]
        for p in pointers:
            if p.pinching:
                stroke = self.active.get(p.pid)
                if stroke is None or p.started:
                    stroke = Stroke(color=PALETTE[self.color_index][1], width=self.width, born=now)
                    self.active[p.pid] = stroke
                    self.strokes.append(stroke)
                stroke.add(p.pos)
            else:
                if p.pid in self.active:
                    s = self.active.pop(p.pid)
                    if len(s.points) < 2:  # a tap leaves a dot
                        s.points.append(s.points[-1] + 0.5)
                if p.is_open and p.palm is not None and not p.vanished:
                    radius = p.palm_size * 0.75
                    self.erasers.append((p.palm, radius))
                    self.erase(p.palm, radius)
        self._enforce_budget()

    def erase(self, center: np.ndarray, radius: float) -> int:
        """Remove points within ``radius`` of ``center``, splitting strokes. Returns removed count."""
        removed = 0
        out: list[Stroke] = []
        active_ids = {id(s): pid for pid, s in self.active.items()}
        for s in self.strokes:
            if id(s) in active_ids:
                out.append(s)
                continue
            pts = np.array(s.points)
            if len(pts) == 0:
                continue
            keep = np.linalg.norm(pts - center, axis=1) > radius
            if keep.all():
                out.append(s)
                continue
            removed += int((~keep).sum())
            # split into contiguous runs of kept points
            run: list[np.ndarray] = []
            for pt, k in zip(s.points, keep):
                if k:
                    run.append(pt)
                elif run:
                    if len(run) >= 2:
                        out.append(Stroke(run, s.color, s.width, s.born))
                    run = []
            if len(run) >= 2:
                out.append(Stroke(run, s.color, s.width, s.born))
        self.strokes = out
        return removed

    def _enforce_budget(self) -> None:
        total = sum(len(s.points) for s in self.strokes)
        while total > MAX_POINTS and len(self.strokes) > 1:
            total -= len(self.strokes.pop(0).points)

    def _active_ids(self) -> set[int]:
        return {id(s) for s in self.active.values()}  # identity: strokes hold arrays, == is ambiguous

    def undo(self):
        active = self._active_ids()
        for i in range(len(self.strokes) - 1, -1, -1):
            if id(self.strokes[i]) not in active:
                del self.strokes[i]
                return "UNDO"
        return None

    def clear(self):
        active = self._active_ids()
        self.strokes = [s for s in self.strokes if id(s) in active]
        return "CANVAS CLEARED"

    def handle_key(self, key, mod):
        if key == pygame.K_RIGHTBRACKET:
            self.width = min(30.0, self.width + 2)
            return f"BRUSH · {self.width:.0f}px"
        if key == pygame.K_LEFTBRACKET:
            self.width = max(2.0, self.width - 2)
            return f"BRUSH · {self.width:.0f}px"
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            return self.cycle_color(1 if key == pygame.K_RIGHT else -1)
        return None

    def on_swipe(self, direction: int) -> str:
        return self.cycle_color(direction)

    # -- rendering --------------------------------------------------------------------------
    def render_light(self, glow: GlowLayer, now: float) -> None:
        for s in self.strokes:
            if len(s.points) < 2:
                continue
            fresh = max(0.0, 1.0 - (now - s.born) / 0.4)  # brief flash when a stroke starts
            if s.color is None:
                for i in range(len(s.points) - 1):
                    glow.line(s.points[i], s.points[i + 1], _rainbow(i), s.width)
            else:
                c = tuple(min(255, int(v * (1.0 + 0.3 * fresh))) for v in s.color)
                glow.polyline(s.points, c, s.width)

    def render_overlay(self, layer: pygame.Surface, vp: Viewport, now: float) -> None:
        col = PALETTE[self.color_index][1]
        rgb = (col[2], col[1], col[0]) if col else T.INK
        for center, radius in self.erasers:
            c = vp.to_screen(center)
            pygame.draw.circle(layer, T.with_alpha(T.INK, 0.12), c, vp.length(radius))
            pygame.draw.circle(layer, T.with_alpha(T.INK, 0.6), c, vp.length(radius), 1)
        for p in self.cursors:
            if p.is_open:
                continue
            c = vp.to_screen(p.pos)
            r = vp.length(self.width / 2 + 3)
            if p.pinching:
                pygame.draw.circle(layer, T.with_alpha(rgb, 0.9), c, r)
            else:
                ring = 10 + 12 * (1 - (p.hand.pinch_progress if p.hand else 0))
                pygame.draw.circle(layer, T.with_alpha(rgb, 0.7), c, ring, 1)
                pygame.draw.circle(layer, T.with_alpha(rgb, 0.9), c, max(2, r * 0.5))
        # swatch
        f = T.font(10, mono=True)
        W, H = layer.get_size()
        x, y = W - 30, H // 2 - len(PALETTE) * 13
        for i, (name, c) in enumerate(PALETTE):
            cc = (c[2], c[1], c[0]) if c else None
            cy = y + i * 26
            if cc is None:
                for k in range(6):
                    a0 = k / 6 * math.tau
                    hue = _rainbow(k * 28)
                    pygame.draw.arc(layer, (hue[2], hue[1], hue[0]), pygame.Rect(x - 7, cy - 7, 14, 14), a0, a0 + math.tau / 6, 3)
            else:
                pygame.draw.circle(layer, cc, (x, cy), 6)
            if i == self.color_index:
                pygame.draw.circle(layer, T.INK, (x, cy), 10, 1)
                lbl = f.render(name, True, T.INK)
                layer.blit(lbl, (x - 18 - lbl.get_width(), cy - lbl.get_height() // 2))
